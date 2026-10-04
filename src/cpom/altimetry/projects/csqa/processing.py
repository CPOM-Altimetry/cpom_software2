"""cpom.altimetry.projects.csqa.processing

Process one CSQA cycle for one product baseline: find the input files, load each configured
parameter, and write statistics and map plots per area, variant and acquisition mode.

Parameters with a grid (ie freeboard) also have gridded maps and statistics: the measurements
of each selection are gridded into polar stereographic cells, and statistics of the cell values
(ie the median freeboard of each 10 km cell) are mapped and summarised ('grid_rows' of the
statistics file).

Map plots can be rendered in parallel by a pool of plot worker processes (plot_workers > 1).
Each parameter is loaded in turn and its maps are submitted to the pool as soon as its
statistics are calculated, so rendering overlaps with the loading of the next parameter. The
statistics file of each parameter is written when all of its maps are complete.

See outputs.py for the output directory layout.
"""

import contextlib
import hashlib
import logging
import multiprocessing
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from cpom.altimetry.projects.csqa import __version__
from cpom.altimetry.projects.csqa.csqa_config import (
    AreaConfig,
    CsqaConfig,
    ParameterConfig,
    VariantDef,
)
from cpom.altimetry.projects.csqa.gridding import grid_measurements
from cpom.altimetry.projects.csqa.loader import bit_values, load_parameter_data
from cpom.altimetry.projects.csqa.log_setup import current_logging_config, setup_logging
from cpom.altimetry.projects.csqa.outputs import (
    cycle_dir,
    plots_dir,
    read_json,
    stats_path,
    utc_now_str,
    write_json_atomic,
)
from cpom.altimetry.projects.csqa.plotting import (
    PlotJob,
    plot_filename,
    prepare_grid_plot_job,
    prepare_plot_job,
    render_plot_job,
    thumbnail_path,
)
from cpom.altimetry.projects.csqa.product_files import (
    ProductFile,
    coverage_days,
    find_product_files,
)
from cpom.altimetry.projects.csqa.stats import bit_flag_stats, flag_stats, float_stats

log = logging.getLogger(__name__)

TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class CycleResult:
    """Outcome of processing a cycle for a baseline"""

    cycle: int
    baseline: str
    status: str  # 'processed', 'unchanged', 'no_data' or 'error'
    params_processed: list[str] = field(default_factory=list)
    message: str = ""


@dataclass
class _PendingParameter:  # pylint: disable=too-many-instance-attributes
    """A parameter whose statistics are calculated, waiting for its map plots"""

    param: ParameterConfig
    signature: str
    rows: dict[tuple[str, str, str], dict]  # (area, variant, mode) -> statistics row
    # (area, variant, mode, colour scale file suffix) -> plot step (or its future)
    plots: dict[tuple[str, str, str, str], "Future[int] | int"]
    summary: dict  # statistics file fields other than the rows
    t_start: float
    make_plots: bool
    # (area, variant, mode, grid statistic) -> gridded statistics row / its map's plot step
    grid_rows: dict[tuple[str, str, str, str], dict] = field(default_factory=dict)
    grid_plots: dict[tuple[str, str, str, str], "Future[int] | int"] = field(default_factory=dict)


@dataclass
class _PlotContext:
    """What the maps of a parameter's selections need besides the selection"""

    cycle: int
    bounds: tuple
    baseline: str
    pdir: str
    make_plots: bool
    submit_plot: Callable[[PlotJob], "Future[int] | int"]


def input_signature(files: list[ProductFile]) -> str:
    """signature of a set of input files, used to detect changed cycle inputs"""
    names = "\n".join(sorted(f.name for f in files))
    return hashlib.sha1(names.encode("utf-8")).hexdigest()


def find_cycle_files(
    cfg: CsqaConfig, cycle: int, baseline: str, sources: set[str]
) -> dict[str, list[ProductFile]]:
    """Find the input files of each product source for a cycle and baseline

    Args:
        cfg (CsqaConfig): CSQA config
        cycle (int): cycle number
        baseline (str): product baseline
        sources (set[str]): product ids

    Returns:
        dict[str, list[ProductFile]]: files per product id
    """
    start, end = cfg.calendar().cycle_bounds(cycle)
    return {
        src: find_product_files(cfg.products[src], start, end, baseline, cfg.stage_preference)
        for src in sorted(sources)
    }


def _info_path(cfg: CsqaConfig, baseline: str, cycle: int) -> str:
    """path of a cycle's cycle_info.json file"""
    return os.path.join(cycle_dir(cfg.output_dir, baseline, cycle), "cycle_info.json")


def has_maps(param: ParameterConfig, row: dict) -> bool:
    """True if a statistics row's selection has maps: its mode has maps, it has valid values
    and, for a bit of a flag word, the bit is set in some records"""
    if row.get("mode", "") not in param.map_modes or row.get("n_valid", 0) == 0:
        return False
    if param.is_bit_flag:
        return (row.get("counts") or {}).get("set", 0) > 0
    return True


def grid_plot_filename(param: ParameterConfig, row: dict, fmt: str) -> str:
    """name of the gridded map of a grid statistics row, ie
    freeboard_filtered_all_north_polar_grid10km_median.webp"""
    assert param.grid is not None
    return plot_filename(
        param.id,
        row["variant"],
        row["mode"],
        row["area"],
        fmt,
        param.grid.file_suffix(row["statistic"]),
    )


def expected_plots(param: ParameterConfig, stats: dict, fmt: str) -> set[str]:
    """names of the maps (of every colour scale, and gridded maps) that a parameter's cycle
    statistics should have"""
    names = {
        plot_filename(param.id, r["variant"], r["mode"], r["area"], fmt, s.file_suffix)
        for r in stats.get("rows", [])
        if has_maps(param, r)
        for s in param.colour_scales
    }
    if param.grid is not None:
        names |= {
            grid_plot_filename(param, r, fmt)
            for r in stats.get("grid_rows", [])
            if r.get("n_cells", 0) > 0
        }
    return names


def _outputs_exist(cfg: CsqaConfig, baseline: str, cycle: int, param_id: str, make_plots: bool):
    """True if the stats file (and plots of every colour scale and grid if required) of a
    parameter exist for a cycle"""
    stats = read_json(stats_path(cfg.output_dir, baseline, cycle, param_id))
    if stats is None:
        return False
    param = cfg.parameters[param_id]
    if param.grid is not None and "grid_rows" not in stats:
        return False  # processed before the parameter was gridded
    if make_plots:
        pdir = plots_dir(cfg.output_dir, baseline, cycle, param_id)
        for fname in expected_plots(param, stats, cfg.image_format):
            if not os.path.isfile(os.path.join(pdir, fname)):
                return False
    return True


def params_to_process(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: CsqaConfig,
    cycle: int,
    baseline: str,
    param_ids: list[str] | None,
    make_plots: bool,
    update: bool,
    cycle_files: dict[str, list[ProductFile]] | None = None,
) -> list[str]:
    """Parameters of a cycle and baseline that need processing: those with input files and,
    in update mode, whose input files changed since last processed (or outputs are missing)

    Args:
        cfg (CsqaConfig): CSQA config
        cycle (int): cycle number
        baseline (str): product baseline
        param_ids (list[str]|None): parameters requested (default all)
        make_plots (bool): True if map plots are produced
        update (bool): update mode
        cycle_files (dict|None): input files per product, from find_cycle_files()

    Returns:
        list[str]: parameter ids (excluding parameters not in the baseline's products)
    """
    params = [cfg.parameters[pid] for pid in (param_ids or list(cfg.parameters))]
    if cycle_files is None:
        cycle_files = find_cycle_files(cfg, cycle, baseline, {p.source for p in params})
    previous = (read_json(_info_path(cfg, baseline, cycle)) or {}).get("parameters", {})

    needed = []
    for param in params:
        files = cycle_files.get(param.source, [])
        if not files or not param.in_baseline(baseline):
            continue
        if (
            update
            and previous.get(param.id, {}).get("input_signature") == input_signature(files)
            and _outputs_exist(cfg, baseline, cycle, param.id, make_plots)
        ):
            continue
        needed.append(param.id)
    return needed


def plan_cycle(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: CsqaConfig,
    cycle: int,
    baseline: str,
    param_ids: list[str] | None,
    make_plots: bool,
    update: bool,
) -> tuple[str, list[str]]:
    """What processing a cycle and baseline needs (a quick check of input files and outputs)

    Returns:
        tuple[str, list[str]]: status ('no_data', 'unchanged' or 'process') and the ids of
                               the parameters to process
    """
    params = [cfg.parameters[pid] for pid in (param_ids or list(cfg.parameters))]
    cycle_files = find_cycle_files(cfg, cycle, baseline, {p.source for p in params})
    if not any(cycle_files.values()):
        return "no_data", []
    needed = params_to_process(cfg, cycle, baseline, param_ids, make_plots, update, cycle_files)
    return ("process" if needed else "unchanged"), needed


def _remove_plot(plot_path: str):
    """remove a plot and its thumbnail if they exist"""
    for path in (plot_path, thumbnail_path(plot_path)):
        if os.path.isfile(path):
            os.remove(path)


def _grid_selection(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: CsqaConfig,
    param: ParameterConfig,
    variant: VariantDef,
    mode: str,
    area: AreaConfig,
    lats: np.ndarray,
    lons: np.ndarray,
    vals: np.ndarray,
    ctx: _PlotContext,
    pending: _PendingParameter,
):
    """Grid the measurements of a selection, adding a gridded statistics row for each grid
    statistic and submitting their maps (or, when not plotting, keeping existing maps)

    Args:
        cfg (CsqaConfig): CSQA config
        param (ParameterConfig): parameter (with a grid)
        variant (VariantDef): variant gridded
        mode (str): mode selection
        area (AreaConfig): area gridded (with a grid_area)
        lats (np.ndarray): latitudes of the selection's measurements
        lons (np.ndarray): longitudes of the selection's measurements
        vals (np.ndarray): values of the selection's measurements (NaN for missing)
        ctx (_PlotContext): plot context
        pending (_PendingParameter): parameter whose grid rows and plots are added to
    """
    grid = param.grid
    assert grid is not None
    gridded = grid_measurements(
        area.grid_area,
        int(round(grid.binsize_km * 1000.0)),
        lats,
        lons,
        vals,
        [s.id for s in grid.statistics],
        grid.min_count,
    )
    for stat in grid.statistics:
        cell_stats = float_stats(gridded.values[stat.id])
        row: dict = {
            "area": area.id,
            "variant": variant.id,
            "mode": mode,
            "statistic": stat.id,
            "n_records": gridded.n_records,  # valid measurements gridded
            "n_cells": cell_stats.pop("n_valid"),  # cells with data
            **cell_stats,  # statistics of the cell values
            "plot": None,
        }
        key = (area.id, variant.id, mode, stat.id)
        pending.grid_rows[key] = row
        fname = grid_plot_filename(param, row, cfg.image_format)
        fpath = os.path.join(ctx.pdir, fname)
        if row["n_cells"] == 0:
            _remove_plot(fpath)  # left from an earlier run with different inputs
        elif ctx.make_plots:
            job = prepare_grid_plot_job(
                cfg,
                param,
                variant,
                mode,
                area,
                gridded,
                stat,
                row,
                ctx.cycle,
                ctx.bounds,
                ctx.baseline,
                fpath,
            )
            pending.grid_plots[key] = ctx.submit_plot(job)
        elif os.path.isfile(fpath):  # keep the existing map when not plotting
            row["plot"] = fname


def _prepare_parameter(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: CsqaConfig,
    param: ParameterConfig,
    files: list[ProductFile],
    cycle: int,
    baseline: str,
    area_ids: list[str],
    make_plots: bool,
    submit_plot: Callable[[PlotJob], "Future[int] | int"],
) -> _PendingParameter:
    """Load a parameter, calculate its statistics and submit its map plots

    Args:
        cfg (CsqaConfig): CSQA config
        param (ParameterConfig): parameter
        files (list[ProductFile]): input files
        cycle (int): cycle number
        baseline (str): product baseline
        area_ids (list[str]): areas to process (subset of param.areas)
        make_plots (bool): plot maps if True
        submit_plot (Callable): renders a plot job, returning its plot step or a future of it

    Returns:
        _PendingParameter
    """
    # pylint: disable=too-many-locals
    t_start = time.time()
    bounds = cfg.calendar().cycle_bounds(cycle)
    data = load_parameter_data(files, param, cfg, *bounds)
    log.info(
        "cycle %d baseline %s %s: %d records from %d files",
        cycle,
        baseline,
        param.id,
        data.n_records,
        len(data.files_used),
    )

    pdir = plots_dir(cfg.output_dir, baseline, cycle, param.id)

    # keep rows of areas not processed in this run (when processing a subset of areas)
    previous = read_json(stats_path(cfg.output_dir, baseline, cycle, param.id)) or {}
    rows = {
        (r["area"], r["variant"], r["mode"]): r
        for r in previous.get("rows", [])
        if r.get("area") not in area_ids and r.get("area") in param.areas
    }
    plots: dict[tuple[str, str, str, str], "Future[int] | int"] = {}

    def plot_selection(variant, mode, area, sel, get_sel_vals, row):
        """submit the maps of a selection in each colour scale (or when not plotting, keep
        existing maps, and remove maps of selections without maps). get_sel_vals() returns
        the selection's values (only called when they are plotted)"""
        with_maps = has_maps(param, row)
        plotting = make_plots and with_maps
        sel_lats = data.lat(variant.id)[sel] if plotting else None
        sel_lons = data.lon(variant.id)[sel] if plotting else None
        sel_vals = get_sel_vals() if plotting else None
        for scale in param.colour_scales:
            fname = plot_filename(
                param.id, variant.id, mode, area.id, cfg.image_format, scale.file_suffix
            )
            fpath = os.path.join(pdir, fname)
            if not with_maps:
                _remove_plot(fpath)  # left from an earlier run with different inputs
            elif make_plots:
                job = prepare_plot_job(
                    cfg,
                    param,
                    variant,
                    mode,
                    area,
                    sel_lats,
                    sel_lons,
                    sel_vals,
                    row,
                    cycle,
                    bounds,
                    baseline,
                    fpath,
                    scale,
                )
                plots[(area.id, variant.id, mode, scale.file_suffix)] = submit_plot(job)
            elif os.path.isfile(fpath):  # keep the existing map when not plotting
                if scale.file_suffix:
                    row["extra_plots"][scale.id] = fname
                else:
                    row["plot"] = fname

    # area/mode selections, shared by the variants with the same coordinates
    selections: dict[tuple, np.ndarray] = {}

    def selection(variant, mode, area):
        key = (data.coord_names[variant.id], mode, area.id)
        if key not in selections:
            lats = data.lat(variant.id)
            sel = (lats >= area.lat_min) & (lats <= area.lat_max)
            mode_surface = cfg.mode_surfaces.get(mode)
            if mode_surface is not None and data.modes is not None:
                # an acquisition mode over surface types (ie LRM over ice)
                sel &= data.modes == cfg.mode_values[mode_surface.mode]
                if data.surfaces is not None:
                    sel &= np.isin(
                        data.surfaces, [cfg.surface_values[s] for s in mode_surface.surfaces]
                    )
            elif mode not in ("", "all") and data.modes is not None:
                sel &= data.modes == cfg.mode_values[mode]
            selections[key] = sel
        return selections[key]

    def add_row(variant, mode, area, n_records, stats, sel, get_sel_vals):
        """add the statistics row of a selection, and handle its maps"""
        row: dict = {
            "area": area.id,
            "variant": variant.id,
            "mode": mode,
            "n_records": n_records,
            **stats,
            "plot": None,  # map of the default colour scale
            "plot_step": None,
            "extra_plots": {},  # colour scale id -> map of other colour scales
        }
        plot_selection(variant, mode, area, sel, get_sel_vals, row)
        rows[(area.id, variant.id, mode)] = row

    if param.is_bit_flag:
        # bits of a flag word: count every bit in each selection of the words
        words = data.bit_words[param.variants[0].variable]
        for mode in param.mode_options:
            for area_id in area_ids:
                area = cfg.areas[area_id]
                sel = selection(param.variants[0], mode, area)
                sel_words = words[sel]
                valid = sel_words >= 0
                n_valid = int(np.count_nonzero(valid))
                valid_or_none = None if n_valid == sel_words.size else valid
                for variant in param.variants:
                    mask = int(variant.bit_mask or 0)
                    add_row(
                        variant,
                        mode,
                        area,
                        int(sel_words.size),
                        bit_flag_stats(sel_words, valid_or_none, n_valid, mask, param.flags),
                        sel,
                        lambda w=sel_words, m=mask: bit_values(w, m),
                    )
    else:
        for variant in param.variants:
            vals = data.variant_values(variant)
            for mode in param.mode_options:
                for area_id in area_ids:
                    area = cfg.areas[area_id]
                    sel = selection(variant, mode, area)
                    sel_vals = vals[sel]
                    stats = (
                        flag_stats(sel_vals, param.flags)
                        if param.type == "flag"
                        else float_stats(sel_vals)
                    )
                    add_row(
                        variant, mode, area, int(sel_vals.size), stats, sel, lambda v=sel_vals: v
                    )

    summary = {
        "parameter": param.id,
        "type": param.type,
        "baseline": baseline,
        "cycle": cycle,
        "start": bounds[0].strftime(TIME_FMT),
        "end": bounds[1].strftime(TIME_FMT),
        "n_files": len(data.files_used),
        "n_records": data.n_records,
        "first_record_time": data.first_time.strftime(TIME_FMT) if data.first_time else None,
        "last_record_time": data.last_time.strftime(TIME_FMT) if data.last_time else None,
        "bad_files": data.bad_files,
        "missing_variables": data.missing_variables,
    }
    pending = _PendingParameter(
        param, input_signature(files), rows, plots, summary, t_start, make_plots
    )
    if param.grid is not None:
        # keep the grid rows of areas not processed in this run
        pending.grid_rows = {
            (r["area"], r["variant"], r["mode"], r["statistic"]): r
            for r in previous.get("grid_rows", [])
            if r.get("area") not in area_ids and r.get("area") in param.grid.areas
        }
        ctx = _PlotContext(cycle, bounds, baseline, pdir, make_plots, submit_plot)
        for variant in param.variants:
            vals = data.variant_values(variant)
            for mode in param.grid.modes:
                for area in [cfg.areas[a] for a in param.grid.areas if a in area_ids]:
                    sel = selection(variant, mode, area)
                    _grid_selection(
                        cfg,
                        param,
                        variant,
                        mode,
                        area,
                        data.lat(variant.id)[sel],
                        data.lon(variant.id)[sel],
                        vals[sel],
                        ctx,
                        pending,
                    )
    return pending


def _remove_stale_plots(
    cfg: CsqaConfig, param: ParameterConfig, baseline: str, cycle: int, stats: dict
):
    """Remove maps (and thumbnails) of a parameter for a cycle that none of its selections,
    colour scales and grid statistics produce any more (ie after a colour scale is changed or
    removed)

    Args:
        cfg (CsqaConfig): CSQA config
        param (ParameterConfig): parameter
        baseline (str): product baseline
        cycle (int): cycle number
        stats (dict): statistics of the parameter for the cycle
    """
    pdir = plots_dir(cfg.output_dir, baseline, cycle, param.id)
    expected = expected_plots(param, stats, cfg.image_format)
    for directory in (pdir, os.path.join(pdir, "thumbs")):
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if (
                name.startswith(f"{param.id}_")
                and name.endswith(f".{cfg.image_format}")
                and name not in expected
            ):
                os.remove(os.path.join(directory, name))
                log.info("removed stale map %s", os.path.join(directory, name))


def _finish_parameter(cfg: CsqaConfig, pending: _PendingParameter) -> dict:
    """Wait for a parameter's map plots, then write its statistics file

    Args:
        cfg (CsqaConfig): CSQA config
        pending (_PendingParameter): parameter prepared by _prepare_parameter()

    Returns:
        dict: statistics of the parameter for the cycle (as written to the stats json file)
    """
    param = pending.param
    scale_ids = {s.file_suffix: s.id for s in param.colour_scales}
    for (area_id, variant_id, mode, suffix), result in pending.plots.items():
        step = result.result() if isinstance(result, Future) else result
        row = pending.rows[(area_id, variant_id, mode)]
        fname = plot_filename(param.id, variant_id, mode, area_id, cfg.image_format, suffix)
        if suffix:
            row["extra_plots"][scale_ids[suffix]] = fname
        else:
            row["plot"] = fname
            row["plot_step"] = step

    # order rows as in the parameter definition
    order = {
        key: i
        for i, key in enumerate(
            (a, v.id, m) for v in param.variants for m in param.mode_options for a in param.areas
        )
    }
    stats = {
        **pending.summary,
        "processed_at": utc_now_str(),
        "software_version": __version__,
        "rows": sorted(
            pending.rows.values(),
            key=lambda r: order.get((r["area"], r["variant"], r["mode"]), len(order)),
        ),
    }
    if param.grid is not None:
        for key, result in pending.grid_plots.items():
            if isinstance(result, Future):
                result.result()
            row = pending.grid_rows[key]
            row["plot"] = grid_plot_filename(param, row, cfg.image_format)
        grid_order = {
            key: i
            for i, key in enumerate(
                (a, v.id, m, s.id)
                for v in param.variants
                for m in param.grid.modes
                for a in param.grid.areas
                for s in param.grid.statistics
            )
        }
        stats["grid"] = {
            "binsize_km": param.grid.binsize_km,
            "min_count": param.grid.min_count,
            "grid_areas": {a: cfg.areas[a].grid_area for a in param.grid.areas},
        }
        stats["grid_rows"] = sorted(
            pending.grid_rows.values(),
            key=lambda r: grid_order.get(
                (r["area"], r["variant"], r["mode"], r["statistic"]), len(grid_order)
            ),
        )
    baseline, cycle = pending.summary["baseline"], pending.summary["cycle"]
    write_json_atomic(stats_path(cfg.output_dir, baseline, cycle, param.id), stats)
    if pending.make_plots:
        _remove_stale_plots(cfg, param, baseline, cycle, stats)
    return stats


def plot_pool(plot_workers: int) -> ProcessPoolExecutor:
    """A pool of plot worker processes, logging like the current process"""
    return ProcessPoolExecutor(
        max_workers=plot_workers,
        # spawn: fresh processes (safe with matplotlib/netCDF threads on all platforms)
        mp_context=multiprocessing.get_context("spawn"),
        initializer=setup_logging,
        initargs=current_logging_config(),
    )


def process_cycle(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: CsqaConfig,
    cycle: int,
    baseline: str,
    param_ids: list[str] | None = None,
    area_ids: list[str] | None = None,
    make_plots: bool = True,
    update: bool = False,
    plot_workers: int = 1,
) -> CycleResult:
    """Process the configured parameters for a cycle and baseline

    Args:
        cfg (CsqaConfig): CSQA config
        cycle (int): cycle number
        baseline (str): product baseline, ie 'F'
        param_ids (list[str]|None): parameters to process (default all)
        area_ids (list[str]|None): areas to process (default all of each parameter's areas)
        make_plots (bool): plot maps if True, otherwise only calculate statistics
        update (bool): only process parameters whose input files have changed since they
                       were last processed (or whose outputs are missing)
        plot_workers (int): number of processes rendering the map plots in parallel
                            (1 = render in this process)

    Returns:
        CycleResult
    """
    # pylint: disable=too-many-locals
    t_start = time.time()
    params = [cfg.parameters[pid] for pid in (param_ids or list(cfg.parameters))]
    start, end = cfg.calendar().cycle_bounds(cycle)

    cycle_files = find_cycle_files(cfg, cycle, baseline, {p.source for p in params})
    if not any(cycle_files.values()):
        return CycleResult(cycle, baseline, "no_data", message="no input files")

    info_file = _info_path(cfg, baseline, cycle)
    info = read_json(info_file) or {}
    info.update(
        {
            "cycle": cycle,
            "baseline": baseline,
            "start": start.strftime(TIME_FMT),
            "end": end.strftime(TIME_FMT),
            "cycle_length_days": cfg.cycle_length_days,
        }
    )
    info.setdefault("products", {})
    info.setdefault("parameters", {})
    for src, files in cycle_files.items():
        info["products"][src] = {
            "n_files": len(files),
            "coverage_days": round(coverage_days(files, start, end), 3),
            "first_file": files[0].name if files else None,
            "last_file": files[-1].name if files else None,
            "input_signature": input_signature(files),
        }

    needed = params_to_process(cfg, cycle, baseline, param_ids, make_plots, update, cycle_files)
    for param in params:
        if param.id not in needed and param.in_baseline(baseline):
            log.info("cycle %d baseline %s %s: nothing to do", cycle, baseline, param.id)

    result = CycleResult(cycle, baseline, "unchanged")
    use_pool = make_plots and plot_workers > 1 and bool(needed)
    pool_context = plot_pool(plot_workers) if use_pool else contextlib.nullcontext()
    with pool_context as pool:

        def submit_plot(job: PlotJob) -> "Future[int] | int":
            if pool is None:
                return render_plot_job(job)
            return pool.submit(render_plot_job, job)

        pending = []
        for pid in needed:
            param = cfg.parameters[pid]
            areas = [a for a in param.areas if area_ids is None or a in area_ids]
            pending.append(
                _prepare_parameter(
                    cfg,
                    param,
                    cycle_files[param.source],
                    cycle,
                    baseline,
                    areas,
                    make_plots,
                    submit_plot,
                )
            )

        for item in pending:
            stats = _finish_parameter(cfg, item)
            info["parameters"][item.param.id] = {
                "input_signature": item.signature,
                "n_files": stats["n_files"],
                "n_records": stats["n_records"],
                "first_record_time": stats["first_record_time"],
                "last_record_time": stats["last_record_time"],
                "processed_at": stats["processed_at"],
                "processing_seconds": round(time.time() - item.t_start, 1),
            }
            result.params_processed.append(item.param.id)

    if result.params_processed:
        result.status = "processed"
    info["processed_at"] = utc_now_str()
    info["software_version"] = __version__
    write_json_atomic(info_file, info)
    result.message = f"{time.time() - t_start:.0f}s"
    if use_pool:
        result.message += f" ({plot_workers} plot workers)"
    return result

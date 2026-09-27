"""cpom.altimetry.projects.csqa.processing

Process one CSQA cycle for one product baseline: find the input files, load each configured
parameter, and write statistics and map plots per area, variant and acquisition mode.

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
from cpom.altimetry.projects.csqa.csqa_config import CsqaConfig, ParameterConfig
from cpom.altimetry.projects.csqa.loader import load_parameter_data
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
    prepare_plot_job,
    render_plot_job,
    thumbnail_path,
)
from cpom.altimetry.projects.csqa.product_files import (
    ProductFile,
    coverage_days,
    find_product_files,
)
from cpom.altimetry.projects.csqa.stats import flag_stats, float_stats

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


def _outputs_exist(cfg: CsqaConfig, baseline: str, cycle: int, param_id: str, make_plots: bool):
    """True if the stats file (and plots of every colour scale if required) of a parameter
    exist for a cycle"""
    stats = read_json(stats_path(cfg.output_dir, baseline, cycle, param_id))
    if stats is None:
        return False
    if make_plots:
        param = cfg.parameters[param_id]
        pdir = plots_dir(cfg.output_dir, baseline, cycle, param_id)
        for row in stats.get("rows", []):
            if row.get("n_valid", 0) == 0:
                continue
            for scale in param.colour_scales:
                fname = plot_filename(
                    param_id,
                    row["variant"],
                    row["mode"],
                    row["area"],
                    cfg.image_format,
                    scale.file_suffix,
                )
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
        list[str]: parameter ids
    """
    params = [cfg.parameters[pid] for pid in (param_ids or list(cfg.parameters))]
    if cycle_files is None:
        cycle_files = find_cycle_files(cfg, cycle, baseline, {p.source for p in params})
    previous = (read_json(_info_path(cfg, baseline, cycle)) or {}).get("parameters", {})

    needed = []
    for param in params:
        files = cycle_files.get(param.source, [])
        if not files:
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

    def plot_selection(variant, mode, area, lats, lons, idx, sel_vals, row):
        """submit the maps of a selection in each colour scale (or when not plotting, keep
        existing maps, and remove maps of selections without valid values)"""
        sel_lats = lats[idx] if make_plots and row["n_valid"] > 0 else None
        sel_lons = lons[idx] if make_plots and row["n_valid"] > 0 else None
        for scale in param.colour_scales:
            fname = plot_filename(
                param.id, variant.id, mode, area.id, cfg.image_format, scale.file_suffix
            )
            fpath = os.path.join(pdir, fname)
            if row["n_valid"] == 0:
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

    for variant in param.variants:
        vals = data.values[variant.id]
        lats = data.lat(variant.id)
        lons = data.lon(variant.id)
        for mode in param.mode_options:
            mode_sel = None
            if mode not in ("", "all") and data.modes is not None:
                mode_sel = data.modes == cfg.mode_values[mode]
            for area_id in area_ids:
                area = cfg.areas[area_id]
                sel = (lats >= area.lat_min) & (lats <= area.lat_max)
                if mode_sel is not None:
                    sel &= mode_sel
                idx = np.flatnonzero(sel)
                sel_vals = vals[idx]

                key = (area_id, variant.id, mode)
                row: dict = {
                    "area": area_id,
                    "variant": variant.id,
                    "mode": mode,
                    "n_records": int(idx.size),
                }
                if param.type == "flag":
                    row.update(flag_stats(sel_vals, param.flags))
                else:
                    row.update(float_stats(sel_vals))
                row["plot"] = None  # map of the default colour scale
                row["plot_step"] = None
                row["extra_plots"] = {}  # colour scale id -> map of other colour scales
                plot_selection(variant, mode, area, lats, lons, idx, sel_vals, row)
                rows[key] = row

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
    return _PendingParameter(param, input_signature(files), rows, plots, summary, t_start)


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
    baseline, cycle = pending.summary["baseline"], pending.summary["cycle"]
    write_json_atomic(stats_path(cfg.output_dir, baseline, cycle, param.id), stats)
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
        if param.id not in needed:
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

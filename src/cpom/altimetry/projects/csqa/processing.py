"""cpom.altimetry.projects.csqa.processing

Process one CSQA cycle for one product baseline: find the input files, load each configured
parameter, and write statistics and map plots per area, variant and acquisition mode.

See outputs.py for the output directory layout.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

from cpom.altimetry.projects.csqa import __version__
from cpom.altimetry.projects.csqa.csqa_config import CsqaConfig, ParameterConfig
from cpom.altimetry.projects.csqa.cycles import CycleCalendar
from cpom.altimetry.projects.csqa.loader import load_parameter_data
from cpom.altimetry.projects.csqa.outputs import (
    cycle_dir,
    plots_dir,
    read_json,
    stats_path,
    utc_now_str,
    write_json_atomic,
)
from cpom.altimetry.projects.csqa.plotting import (
    plot_filename,
    plot_parameter_map,
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
    start, end = CycleCalendar(cfg.mission_start_date, cfg.cycle_length_days).cycle_bounds(cycle)
    return {
        src: find_product_files(cfg.products[src], start, end, baseline, cfg.stage_preference)
        for src in sorted(sources)
    }


def _outputs_exist(cfg: CsqaConfig, baseline: str, cycle: int, param_id: str, make_plots: bool):
    """True if the stats file (and plots if required) of a parameter exist for a cycle"""
    stats = read_json(stats_path(cfg.output_dir, baseline, cycle, param_id))
    if stats is None:
        return False
    if make_plots:
        pdir = plots_dir(cfg.output_dir, baseline, cycle, param_id)
        for row in stats.get("rows", []):
            if row.get("n_valid", 0) > 0 and not (
                row.get("plot") and os.path.isfile(os.path.join(pdir, row["plot"]))
            ):
                return False
    return True


def _remove_plot(plot_path: str):
    """remove a plot and its thumbnail if they exist"""
    for path in (plot_path, thumbnail_path(plot_path)):
        if os.path.isfile(path):
            os.remove(path)


def process_parameter(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: CsqaConfig,
    param: ParameterConfig,
    files: list[ProductFile],
    cycle: int,
    baseline: str,
    area_ids: list[str],
    make_plots: bool = True,
) -> dict:
    """Calculate the statistics and plot the maps of a parameter for a cycle

    Args:
        cfg (CsqaConfig): CSQA config
        param (ParameterConfig): parameter
        files (list[ProductFile]): input files
        cycle (int): cycle number
        baseline (str): product baseline
        area_ids (list[str]): areas to process (subset of param.areas)
        make_plots (bool): plot maps if True

    Returns:
        dict: statistics of the parameter for the cycle (as written to the stats json file)
    """
    bounds = CycleCalendar(cfg.mission_start_date, cfg.cycle_length_days).cycle_bounds(cycle)
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
    stats_file = stats_path(cfg.output_dir, baseline, cycle, param.id)

    # keep rows of areas not processed in this run (when processing a subset of areas)
    previous = read_json(stats_file) or {}
    rows = {
        (r["area"], r["variant"], r["mode"]): r
        for r in previous.get("rows", [])
        if r.get("area") not in area_ids and r.get("area") in param.areas
    }

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

                fname = plot_filename(param.id, variant.id, mode, area_id, cfg.image_format)
                fpath = os.path.join(pdir, fname)
                row["plot"] = None
                row["plot_step"] = None
                if row["n_valid"] == 0:
                    _remove_plot(fpath)  # left from an earlier run with different inputs
                elif make_plots:
                    row["plot_step"] = plot_parameter_map(
                        cfg,
                        param,
                        variant,
                        mode,
                        area,
                        lats[idx],
                        lons[idx],
                        sel_vals,
                        cycle,
                        bounds,
                        baseline,
                        fpath,
                    )
                    row["plot"] = fname
                elif os.path.isfile(fpath):
                    row["plot"] = fname  # keep the existing plot when not plotting
                rows[(area_id, variant.id, mode)] = row

    # order rows as in the parameter definition
    order = {
        key: i
        for i, key in enumerate(
            (a, v.id, m) for v in param.variants for m in param.mode_options for a in param.areas
        )
    }
    stats = {
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
        "processed_at": utc_now_str(),
        "software_version": __version__,
        "rows": sorted(
            rows.values(), key=lambda r: order.get((r["area"], r["variant"], r["mode"]), len(order))
        ),
    }
    write_json_atomic(stats_file, stats)
    return stats


def process_cycle(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: CsqaConfig,
    cycle: int,
    baseline: str,
    param_ids: list[str] | None = None,
    area_ids: list[str] | None = None,
    make_plots: bool = True,
    update: bool = False,
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

    Returns:
        CycleResult
    """
    t_start = time.time()
    params = [cfg.parameters[pid] for pid in (param_ids or list(cfg.parameters))]
    start, end = CycleCalendar(cfg.mission_start_date, cfg.cycle_length_days).cycle_bounds(cycle)

    cycle_files = find_cycle_files(cfg, cycle, baseline, {p.source for p in params})
    if not any(cycle_files.values()):
        return CycleResult(cycle, baseline, "no_data", message="no input files")

    info_file = os.path.join(cycle_dir(cfg.output_dir, baseline, cycle), "cycle_info.json")
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

    result = CycleResult(cycle, baseline, "unchanged")
    for param in params:
        files = cycle_files[param.source]
        if not files:
            log.info(
                "cycle %d baseline %s: no %s files for %s", cycle, baseline, param.source, param.id
            )
            continue
        signature = input_signature(files)
        previous = info["parameters"].get(param.id, {})
        if (
            update
            and previous.get("input_signature") == signature
            and _outputs_exist(cfg, baseline, cycle, param.id, make_plots)
        ):
            log.info(
                "cycle %d baseline %s %s: inputs unchanged, skipping", cycle, baseline, param.id
            )
            continue

        areas = [a for a in param.areas if area_ids is None or a in area_ids]
        t_param = time.time()
        stats = process_parameter(cfg, param, files, cycle, baseline, areas, make_plots)
        info["parameters"][param.id] = {
            "input_signature": signature,
            "n_files": stats["n_files"],
            "n_records": stats["n_records"],
            "first_record_time": stats["first_record_time"],
            "last_record_time": stats["last_record_time"],
            "processed_at": stats["processed_at"],
            "processing_seconds": round(time.time() - t_param, 1),
        }
        result.params_processed.append(param.id)

    if result.params_processed:
        result.status = "processed"
    info["processed_at"] = utc_now_str()
    info["software_version"] = __version__
    write_json_atomic(info_file, info)
    result.message = f"{time.time() - t_start:.0f}s"
    return result

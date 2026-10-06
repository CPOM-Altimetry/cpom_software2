#!/usr/bin/env python3
"""cpom.altimetry.projects.csqa.build_portal_index

Build the CSQA portal index from the processed cycle outputs:

    <output_dir>/manifest.json
        parameter definitions, areas, modes and the processed cycles of each baseline
    <output_dir>/baseline_<B>/timeseries/<param>.csv
        statistics of every processed cycle, one row per cycle/area/variant/mode
    <output_dir>/availability.json
        files per acquisition day of the most recent input products (Data Availability page)
    <output_dir>/baseline_<B>/timeseries/<param>_grid.csv
        gridded statistics (of parameters with a grid), one row per
        cycle/area/variant/mode/grid statistic

Run automatically at the end of process_cycles.py, or standalone:

    python build_portal_index.py [--config csqa_config.yaml]
"""

import argparse
import csv
import glob
import io
import logging
import os
import re
import sys

from cpom.altimetry.projects.csqa import __version__
from cpom.altimetry.projects.csqa.availability import availability
from cpom.altimetry.projects.csqa.csqa_config import (
    CsqaConfig,
    ParameterConfig,
    load_config,
)
from cpom.altimetry.projects.csqa.outputs import (
    read_json,
    timeseries_path,
    utc_now_str,
    write_json_atomic,
    write_text_atomic,
)
from cpom.altimetry.projects.csqa.stats import FLOAT_STATS

log = logging.getLogger(__name__)

ROW_KEYS = ("cycle", "start_date", "end_date", "area", "variant", "mode", "n_records", "n_valid")
GRID_ROW_KEYS = (
    "cycle",
    "start_date",
    "end_date",
    "area",
    "variant",
    "mode",
    "statistic",
    "n_records",
    "n_cells",
)


def parameter_manifest(param: ParameterConfig, cfg: CsqaConfig) -> dict:
    """portal description of a parameter"""
    return {
        "id": param.id,
        "long_name": param.long_name,
        "description": param.description,
        "source": param.source,
        "type": param.type,
        "units": param.units,
        "variant_label": param.variant_label,
        "variants": [
            {
                "id": v.id,
                "name": v.name,
                "variable": v.display_variable,
                # input variables of a derived variant (ie the mispointing angle)
                "inputs": list(v.inputs),
                "units": param.variant_units(v),
                "valid_modes": list(v.valid_modes),
                "invalid_values": list(v.invalid_values),
                "plot_log": v.plot_log,
                "mode_descriptions": v.mode_descriptions,
                "bit_mask": v.bit_mask,
                "bit_name": v.bit_name,
                "plot_range": list(v.plot_range) if v.plot_range else None,
                # values are rejected where this bit is set
                "reject_bit": (
                    {"variable": v.reject_variable, "mask": v.reject_mask, "name": v.reject_name}
                    if v.reject_mask is not None
                    else None
                ),
            }
            for v in param.variants
        ],
        "default_variant": param.default_variant,
        "mode_default_variants": param.mode_default_variants,
        "bit_flag": param.is_bit_flag,
        "map_modes": param.map_modes,
        "modes": param.modes,
        "areas": param.areas,
        "flags": [
            {"value": f.value, "name": f.name, "key": f.key, "color": f.color} for f in param.flags
        ],
        "plot_range": list(param.plot_range) if param.plot_range else None,
        # colour scales of the maps (the first is the default)
        "colour_scales": [
            {
                "id": s.id,
                "name": s.name,
                "range": list(s.range) if s.range else None,
                "file_suffix": s.file_suffix,
                "log": s.log,
            }
            for s in param.colour_scales
        ],
        "grid": grid_manifest(param),
        "valid_modes": param.valid_modes,
        "value_scale": param.value_scale,
        # what each value is (ie crossover), and the crossover settings
        "record_name": param.record_name,
        "crossover": (
            {
                "max_abs_difference": param.crossover.max_abs_difference,
                "max_arc_length_m": param.crossover.max_arc_length_m,
                "location": [param.crossover.lat, param.crossover.lon],
                "one_per_pass_pair": param.crossover.one_per_pass_pair,
            }
            if param.crossover is not None
            else None
        ),
        "first_baseline": param.first_baseline,
        "image_format": cfg.image_format,
    }


def mode_manifest(cfg: CsqaConfig, mode: str, label: str) -> dict:
    """portal description of a mode selection: its kind (mode, mode_surface or pass), and the
    mode and surface types of mode surface selections"""
    if mode in cfg.mode_surfaces:
        sel = cfg.mode_surfaces[mode]
        return {
            "id": mode,
            "label": label,
            "kind": "mode_surface",
            "mode": sel.mode,
            "surfaces": list(sel.surfaces),
        }
    if mode in cfg.pass_selections:
        return {"id": mode, "label": label, "kind": "pass"}
    return {"id": mode, "label": label, "kind": "mode"}


def grid_manifest(param: ParameterConfig) -> dict | None:
    """portal description of a parameter's grid (None if it is not gridded)"""
    grid = param.grid
    if grid is None:
        return None
    return {
        "label": grid.label,
        "binsize_km": grid.binsize_km,
        "min_count": grid.min_count,
        "smooth_radius_km": grid.smooth_radius_km,
        "cell_text": grid.cell_text,
        "areas": grid.areas,
        "modes": grid.modes,
        # each statistic's maps have the file name suffix <file_suffix> (the first statistic is
        # the default)
        "statistics": [
            {
                "id": s.id,
                "name": s.name,
                "units": s.units,
                "range": list(s.range) if s.range else None,
                "log": s.log,
                "file_suffix": grid.file_suffix(s.id),
            }
            for s in grid.statistics
        ],
    }


def timeseries_columns(param: ParameterConfig) -> list[str]:
    """csv columns of a parameter's timeseries file"""
    if param.type == "flag":
        return [*ROW_KEYS, "n_other", *[f"pct_{f.key}" for f in param.flags]]
    return [*ROW_KEYS, *FLOAT_STATS]


def timeseries_csv(param: ParameterConfig, stats_files: list[dict]) -> str:
    """csv text of a parameter's statistics for a list of cycle stats (sorted by cycle)"""
    columns = timeseries_columns(param)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    for stats in stats_files:
        for row in stats.get("rows", []):
            values = {
                "cycle": stats["cycle"],
                "start_date": stats["start"][:10],
                "end_date": stats["end"][:10],
                **row,
            }
            if param.type == "flag":
                for key, pct in (row.get("pct") or {}).items():
                    values[f"pct_{key}"] = pct
            writer.writerow(["" if values.get(c) is None else values.get(c) for c in columns])
    return out.getvalue()


def grid_timeseries_csv(stats_files: list[dict]) -> str:
    """csv text of a parameter's gridded statistics for a list of cycle stats (sorted by
    cycle)"""
    columns = [*GRID_ROW_KEYS, *FLOAT_STATS]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    for stats in stats_files:
        for row in stats.get("grid_rows", []):
            values = {
                "cycle": stats["cycle"],
                "start_date": stats["start"][:10],
                "end_date": stats["end"][:10],
                **row,
            }
            writer.writerow(["" if values.get(c) is None else values.get(c) for c in columns])
    return out.getvalue()


def write_availability(cfg: CsqaConfig):
    """write the availability of the most recent input products (portal Data Availability
    page) to <output_dir>/availability.json"""
    write_json_atomic(os.path.join(cfg.output_dir, "availability.json"), availability(cfg))
    log.info("wrote %s", os.path.join(cfg.output_dir, "availability.json"))


def build_portal_index(cfg: CsqaConfig) -> dict:
    """Build the portal manifest and statistics timeseries files

    Args:
        cfg (CsqaConfig): CSQA config

    Returns:
        dict: the manifest written to <output_dir>/manifest.json
    """
    baselines = []
    for bdir in sorted(glob.glob(os.path.join(cfg.output_dir, "baseline_*"))):
        match = re.fullmatch(r"baseline_([A-Z])", os.path.basename(bdir))
        if match is None:
            continue
        baseline = match[1]

        cycles = []
        stats_by_param: dict[str, list[dict]] = {pid: [] for pid in cfg.parameters}
        for info_file in sorted(
            glob.glob(os.path.join(bdir, "cycles", "cycle_*", "cycle_info.json"))
        ):
            info = read_json(info_file)
            if info is None:
                log.warning("unreadable %s", info_file)
                continue
            cdir = os.path.dirname(info_file)
            params_available = []
            for pid, stats_list in stats_by_param.items():
                stats = read_json(os.path.join(cdir, "stats", f"{pid}.json"))
                if stats is not None:
                    stats_list.append(stats)
                    params_available.append(pid)
            if not params_available:
                continue
            cycles.append(
                {
                    "cycle": info["cycle"],
                    "start": info["start"],
                    "end": info["end"],
                    "products": {
                        src: {k: p.get(k) for k in ("n_files", "coverage_days")}
                        for src, p in info.get("products", {}).items()
                    },
                    "parameters": params_available,
                    "processed_at": info.get("processed_at"),
                }
            )

        if not cycles:
            continue
        cycles.sort(key=lambda c: c["cycle"])

        for pid, stats_list in stats_by_param.items():
            if not stats_list:
                continue
            stats_list.sort(key=lambda s: s["cycle"])
            write_text_atomic(
                timeseries_path(cfg.output_dir, baseline, pid),
                timeseries_csv(cfg.parameters[pid], stats_list),
            )
            if cfg.parameters[pid].grid is not None:
                write_text_atomic(
                    timeseries_path(cfg.output_dir, baseline, f"{pid}_grid"),
                    grid_timeseries_csv(stats_list),
                )

        baselines.append(
            {
                "id": baseline,
                "default": baseline in cfg.baselines,
                "cycles": cycles,
            }
        )
        log.info("baseline %s: %d cycles indexed", baseline, len(cycles))

    manifest = {
        "generated_at": utc_now_str(),
        "software_version": __version__,
        "mission_start_date": cfg.mission_start_date.strftime("%Y-%m-%d"),
        "cycle_length_days": cfg.cycle_length_days,
        "data_latency_days": cfg.data_latency_days,
        "image_format": cfg.image_format,
        "areas": [{"id": a.id, "long_name": a.long_name} for a in cfg.areas.values()],
        # modes, 'all' and the mode surface selections (with their mode and surface types)
        "modes": [mode_manifest(cfg, k, v) for k, v in cfg.mode_labels.items()],
        "products": {p.id: p.long_name for p in cfg.products.values()},
        "parameters": [parameter_manifest(p, cfg) for p in cfg.parameters.values()],
        # newest baseline first
        "baselines": sorted(baselines, key=lambda b: b["id"], reverse=True),
    }
    write_json_atomic(os.path.join(cfg.output_dir, "manifest.json"), manifest)
    log.info("wrote %s", os.path.join(cfg.output_dir, "manifest.json"))
    write_availability(cfg)
    return manifest


def main(args: list[str] | None = None):
    """main function of build_portal_index tool"""
    parser = argparse.ArgumentParser(description="Build the CSQA portal manifest and timeseries")
    parser.add_argument(
        "--config", help="CSQA config file (default: $CSQA_CONFIG or config/csqa_config.yaml)"
    )
    parsed = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = load_config(parsed.config)
    if not os.path.isdir(cfg.output_dir):
        log.error("output directory %s does not exist", cfg.output_dir)
        sys.exit(1)
    build_portal_index(cfg)


if __name__ == "__main__":
    main()

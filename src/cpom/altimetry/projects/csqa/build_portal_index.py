#!/usr/bin/env python3
"""cpom.altimetry.projects.csqa.build_portal_index

Build the CSQA portal index from the processed cycle outputs:

    <output_dir>/manifest.json
        parameter definitions, areas, modes and the processed cycles of each baseline
    <output_dir>/baseline_<B>/timeseries/<param>.csv
        statistics of every processed cycle, one row per cycle/area/variant/mode

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
        "variants": [{"id": v.id, "name": v.name, "variable": v.variable} for v in param.variants],
        "modes": param.modes,
        "areas": param.areas,
        "flags": [
            {"value": f.value, "name": f.name, "key": f.key, "color": f.color} for f in param.flags
        ],
        "plot_range": list(param.plot_range) if param.plot_range else None,
        "image_format": cfg.image_format,
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
        "image_format": cfg.image_format,
        "areas": [{"id": a.id, "long_name": a.long_name} for a in cfg.areas.values()],
        "modes": [{"id": k, "label": v} for k, v in cfg.mode_labels.items()],
        "products": {p.id: p.long_name for p in cfg.products.values()},
        "parameters": [parameter_manifest(p, cfg) for p in cfg.parameters.values()],
        # newest baseline first
        "baselines": sorted(baselines, key=lambda b: b["id"], reverse=True),
    }
    write_json_atomic(os.path.join(cfg.output_dir, "manifest.json"), manifest)
    log.info("wrote %s", os.path.join(cfg.output_dir, "manifest.json"))
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

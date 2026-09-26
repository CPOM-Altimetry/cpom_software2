#!/usr/bin/env python3
"""cpom.altimetry.projects.csqa.process_cycles

CSQA QCV processing tool: generate the CryoSat-2 performance monitoring portal content (map
plots and statistics) for 30-day cycles and product baselines, then update the portal index.

Examples:

    # process cycle 193 for the default baselines (csqa_config.yaml:baselines)
    python process_cycles.py -c 193

    # list the input files that would be used for cycles 3 and 193, without processing
    python process_cycles.py -c 3 193 --dry_run

    # the cycle containing a date, baseline F, backscatter only
    python process_cycles.py -d 2026-08-01 -b F -p backscatter

    # routine update (ie from cron): reprocess the latest 3 cycles if their input files changed
    python process_cycles.py --latest 3 --update --workers 3

    # full mission reprocessing using 8 parallel processes
    python process_cycles.py --all --workers 8
"""

import argparse
import logging
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from cpom.altimetry.projects.csqa.build_portal_index import build_portal_index
from cpom.altimetry.projects.csqa.csqa_config import CsqaConfig, load_config
from cpom.altimetry.projects.csqa.cycles import CycleCalendar
from cpom.altimetry.projects.csqa.processing import (
    CycleResult,
    find_cycle_files,
    process_cycle,
)
from cpom.altimetry.projects.csqa.product_files import coverage_days

log = logging.getLogger(__name__)

LOG_FORMAT = "[%(levelname)s] %(asctime)s %(processName)s %(name)s: %(message)s"


def setup_logging(level: int, log_file: str | None = None):
    """configure logging to stderr and optionally a log file"""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)
    # quieten verbose third party / plotting modules
    for name in ("matplotlib", "PIL", "cpom.areas", "cpom.backgrounds", "fiona", "rasterio"):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate CryoSat-2 performance monitoring (CSQA) plots and statistics "
        "for 30-day cycles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument("-c", "--cycles", type=int, nargs="+", help="cycle number(s)")
    select.add_argument(
        "-r",
        "--cycle_range",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="range of cycle numbers (inclusive)",
    )
    select.add_argument("-d", "--date", help="process the cycle containing this date (YYYY-MM-DD)")
    select.add_argument(
        "-l", "--latest", type=int, metavar="N", help="the latest N cycles (up to today)"
    )
    select.add_argument("-a", "--all", action="store_true", help="all cycles up to today")

    parser.add_argument(
        "--config", help="CSQA config file (default: $CSQA_CONFIG or config/csqa_config.yaml)"
    )
    parser.add_argument(
        "-b",
        "--baselines",
        nargs="+",
        help="product baseline(s), ie E F (default: csqa_config.yaml baselines)",
    )
    parser.add_argument("-p", "--params", nargs="+", help="parameter id(s) (default: all)")
    parser.add_argument(
        "--areas",
        nargs="+",
        help="area id(s) (default: all). Mainly for testing: statistics of other areas "
        "are kept from earlier runs",
    )
    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="only process parameters whose input files changed since last processed",
    )
    parser.add_argument(
        "--no_plots", action="store_true", help="only calculate statistics (no map plots)"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=1, help="number of cycles processed in parallel"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="list the cycles and input files that would be processed, then exit",
    )
    parser.add_argument(
        "--no_index", action="store_true", help="do not update the portal index when finished"
    )
    parser.add_argument("--log_file", help="also write log messages to this file")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(args)


def select_cycles(parsed: argparse.Namespace, calendar: CycleCalendar) -> list[int]:
    """cycle numbers selected by the command line options"""
    if parsed.cycles:
        return sorted(set(parsed.cycles))
    if parsed.cycle_range:
        first, last = parsed.cycle_range
        return list(range(first, last + 1))
    if parsed.date:
        return [calendar.cycle_for_datetime(datetime.strptime(parsed.date, "%Y-%m-%d"))]
    if parsed.latest:
        return calendar.latest_cycles(parsed.latest)
    return list(range(1, calendar.current_cycle() + 1))


def validate_selection(parsed: argparse.Namespace, cfg: CsqaConfig) -> list[str]:
    """check the parameter, area and baseline options against the config

    Returns:
        list[str]: error messages
    """
    errors = []
    for pid in parsed.params or []:
        if pid not in cfg.parameters:
            errors.append(f"unknown parameter {pid}: choose from {list(cfg.parameters)}")
    for area in parsed.areas or []:
        if area not in cfg.areas:
            errors.append(f"unknown area {area}: choose from {list(cfg.areas)}")
    for baseline in parsed.baselines or []:
        if len(baseline) != 1 or not baseline.isalpha():
            errors.append(f"invalid baseline {baseline}: must be a single letter")
    return errors


def dry_run(cfg: CsqaConfig, cycles: list[int], baselines: list[str], param_ids: list[str]):
    """print the input files that each cycle would use"""
    calendar = CycleCalendar(cfg.mission_start_date, cfg.cycle_length_days)
    sources = {cfg.parameters[p].source for p in param_ids}
    for cycle in cycles:
        start, end = calendar.cycle_bounds(cycle)
        for baseline in baselines:
            files = find_cycle_files(cfg, cycle, baseline, sources)
            for src, src_files in files.items():
                cover = coverage_days(src_files, start, end)
                print(
                    f"cycle {cycle:3d} ({start:%Y-%m-%d} to {end:%Y-%m-%d}) baseline {baseline} "
                    f"{src}: {len(src_files)} files, {cover:.1f} days coverage"
                )
                for pfile in src_files:
                    print(f"    {pfile.path}")


def run_task(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config_file: str,
    cycle: int,
    baseline: str,
    param_ids: list[str],
    area_ids: list[str] | None,
    make_plots: bool,
    update: bool,
    log_level: int,
    log_file: str | None,
) -> CycleResult:
    """process one cycle/baseline (run in a worker process when workers > 1)"""
    if multiprocessing.current_process().name != "MainProcess":
        setup_logging(log_level, log_file)
    try:
        cfg = load_config(config_file)
        return process_cycle(cfg, cycle, baseline, param_ids, area_ids, make_plots, update)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).exception("cycle %d baseline %s failed", cycle, baseline)
        return CycleResult(cycle, baseline, "error", message=str(exc))


def main(args: list[str] | None = None) -> int:
    """main function of the process_cycles tool

    Returns:
        int: exit status (0 = success, 1 = one or more cycles failed)
    """
    parsed = parse_args(args)
    log_level = logging.DEBUG if parsed.verbose else logging.INFO
    setup_logging(log_level, parsed.log_file)

    cfg = load_config(parsed.config)
    errors = validate_selection(parsed, cfg)
    if errors:
        for error in errors:
            log.error(error)
        return 2

    calendar = CycleCalendar(cfg.mission_start_date, cfg.cycle_length_days)
    cycles = select_cycles(parsed, calendar)
    baselines = [b.upper() for b in (parsed.baselines or cfg.baselines)]
    param_ids = parsed.params or list(cfg.parameters)

    if parsed.dry_run:
        dry_run(cfg, cycles, baselines, param_ids)
        return 0

    tasks = [(cycle, baseline) for cycle in cycles for baseline in baselines]
    log.info(
        "processing %d cycles (%d-%d) for baselines %s, parameters %s, %d workers",
        len(cycles),
        cycles[0],
        cycles[-1],
        baselines,
        param_ids,
        parsed.workers,
    )
    t_start = time.time()
    task_args = (
        param_ids,
        parsed.areas,
        not parsed.no_plots,
        parsed.update,
        log_level,
        parsed.log_file,
    )

    results: list[CycleResult] = []
    if parsed.workers > 1 and len(tasks) > 1:
        # spawn: fresh worker processes (safe with matplotlib/netCDF on all platforms)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=parsed.workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(run_task, cfg.config_file, cycle, baseline, *task_args)
                for cycle, baseline in tasks
            ]
            for future in as_completed(futures):
                results.append(future.result())
                _log_result(results[-1])
    else:
        for cycle, baseline in tasks:
            results.append(run_task(cfg.config_file, cycle, baseline, *task_args))
            _log_result(results[-1])

    n_status = {
        s: sum(r.status == s for r in results)
        for s in ("processed", "unchanged", "no_data", "error")
    }
    log.info("finished in %.0fs: %s", time.time() - t_start, n_status)

    if not parsed.no_index and (n_status["processed"] > 0 or not parsed.update):
        build_portal_index(cfg)

    return 1 if n_status["error"] > 0 else 0


def _log_result(result: CycleResult):
    """log the outcome of a cycle"""
    if result.status == "error":
        log.error("cycle %d baseline %s: FAILED %s", result.cycle, result.baseline, result.message)
    elif result.status == "no_data":
        log.info("cycle %d baseline %s: no input data", result.cycle, result.baseline)
    else:
        log.info(
            "cycle %d baseline %s: %s %s %s",
            result.cycle,
            result.baseline,
            result.status,
            result.params_processed,
            result.message,
        )


if __name__ == "__main__":
    sys.exit(main())

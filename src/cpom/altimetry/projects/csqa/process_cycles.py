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

    # routine update (ie from cron): reprocess the latest 3 cycles that can have data
    # (the latest being the cycle containing today - data_latency_days) if their input files
    # changed, using up to 64 processes
    python process_cycles.py --latest 3 --update --workers 64
    # (csqa_daily.sh runs this from cron, with logging: see the script's header)

    # full mission reprocessing using 128 processes
    python process_cycles.py --all --workers 128

Parallel processing:

    --workers is the total number of processes used. Cycles (per baseline) are processed in
    parallel, and within each cycle the map plots are rendered in parallel by plot worker
    processes. By default the workers are shared between the cycles that need processing:
    ie with --workers 64 and 2 cycles to process, each cycle uses 32 plot workers, while with
    200 cycles to process, 64 cycles are processed at a time, each rendering its plots in turn.
    Use --plot_workers to set the plot workers per cycle explicitly.
    Memory: each cycle process holds one parameter of a full cycle (up to ~2-3 GB), each plot
    worker ~0.5 GB.
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
from cpom.altimetry.projects.csqa.log_setup import init_worker, setup_logging
from cpom.altimetry.projects.csqa.processing import (
    CycleResult,
    find_cycle_files,
    plan_cycle,
    process_cycle,
)
from cpom.altimetry.projects.csqa.product_files import coverage_days

log = logging.getLogger(__name__)

# during long runs, update the portal index at this interval so progress appears in the portal
INDEX_UPDATE_SECONDS = 600


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
        "-l",
        "--latest",
        type=int,
        metavar="N",
        help="the latest N cycles, ending with the latest cycle that can have data: the cycle "
        "containing (today - csqa_config.yaml cycles:data_latency_days)",
    )
    select.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="all cycles up to the latest cycle that can have data",
    )

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
        "-w",
        "--workers",
        type=int,
        default=1,
        help="total number of processes, shared between cycles processed in parallel and the "
        "plot workers of each cycle (see Parallel processing below)",
    )
    parser.add_argument(
        "--plot_workers",
        type=int,
        help="plot worker processes per cycle (default: workers / number of cycles to process)",
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
    return list(range(1, calendar.latest_available_cycle() + 1))


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
    if parsed.workers < 1:
        errors.append("--workers must be at least 1")
    if parsed.plot_workers is not None and parsed.plot_workers < 1:
        errors.append("--plot_workers must be at least 1")
    return errors


def max_plots_per_cycle(cfg: CsqaConfig, param_ids: list[str], area_ids: list[str] | None):
    """the largest number of map plots a cycle can have (along-track maps of every colour
    scale, and gridded maps)"""

    def n_areas(areas: list[str]) -> int:
        return len([a for a in areas if area_ids is None or a in area_ids])

    n_plots = 0
    for param in (cfg.parameters[pid] for pid in param_ids):
        n_plots += (
            len(param.variants)
            * len(param.map_modes)
            * n_areas(param.areas)
            * len(param.colour_scales)
        )
        if param.grid is not None:
            n_plots += (
                len(param.variants)
                * len(param.grid.modes)
                * n_areas(param.grid.areas)
                * len(param.grid.statistics)
            )
    return n_plots


def allocate_workers(
    workers: int, n_tasks: int, plot_workers: int | None, max_plots: int
) -> tuple[int, int]:
    """Share the worker processes between cycles and their plot workers

    Args:
        workers (int): total number of processes
        n_tasks (int): number of cycle/baseline tasks to process
        plot_workers (int|None): plot workers per cycle requested, or None to share
        max_plots (int): the largest number of map plots of a cycle

    Returns:
        tuple[int, int]: (number of cycles processed in parallel, plot workers per cycle)
    """
    n_cycle_procs = max(1, min(workers, n_tasks))
    if plot_workers is None:
        plot_workers = min(max(1, workers // n_cycle_procs), max(1, max_plots))
    return n_cycle_procs, plot_workers


def dry_run(cfg: CsqaConfig, cycles: list[int], baselines: list[str], param_ids: list[str]):
    """print the input files that each cycle would use"""
    calendar = cfg.calendar()
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
    plot_workers: int,
    log_level: int,
    log_file: str | None,
) -> CycleResult:
    """process one cycle/baseline (run in a worker process when cycles run in parallel)"""
    if multiprocessing.current_process().name != "MainProcess":
        setup_logging(log_level, log_file)
    try:
        cfg = load_config(config_file)
        return process_cycle(
            cfg, cycle, baseline, param_ids, area_ids, make_plots, update, plot_workers
        )
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

    calendar = cfg.calendar()
    cycles = select_cycles(parsed, calendar)
    baselines = [b.upper() for b in (parsed.baselines or cfg.baselines)]
    param_ids = parsed.params or list(cfg.parameters)
    make_plots = not parsed.no_plots
    if parsed.latest or parsed.all:
        log.info(
            "latest cycle that can have data (today - %g days): %d",
            cfg.data_latency_days,
            calendar.latest_available_cycle(),
        )

    if parsed.dry_run:
        dry_run(cfg, cycles, baselines, param_ids)
        return 0

    # plan: skip cycles without input files, or unchanged in update mode, without starting
    # worker processes for them
    t_start = time.time()
    results: list[CycleResult] = []
    tasks = []
    for cycle in cycles:
        for baseline in baselines:
            status, _ = plan_cycle(cfg, cycle, baseline, param_ids, make_plots, parsed.update)
            if status == "process":
                tasks.append((cycle, baseline))
            else:
                results.append(CycleResult(cycle, baseline, status))
                _log_result(results[-1])

    n_cycle_procs, plot_workers = allocate_workers(
        parsed.workers,
        len(tasks),
        parsed.plot_workers,
        max_plots_per_cycle(cfg, param_ids, parsed.areas),
    )
    if not tasks:
        log.info("nothing to process")
    else:
        log.info(
            "processing %d of %d cycle/baseline selections (cycles %d-%d, baselines %s, "
            "parameters %s): %d in parallel, %d plot workers each",
            len(tasks),
            len(cycles) * len(baselines),
            cycles[0],
            cycles[-1],
            baselines,
            param_ids,
            n_cycle_procs,
            plot_workers if make_plots else 0,
        )
    task_args = (
        param_ids,
        parsed.areas,
        make_plots,
        parsed.update,
        plot_workers,
        log_level,
        parsed.log_file,
    )

    last_index = time.time()
    if n_cycle_procs > 1:
        # spawn: fresh worker processes (safe with matplotlib/netCDF on all platforms)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_cycle_procs,
            mp_context=ctx,
            # workers log like this process, and end if it is killed
            initializer=init_worker,
            initargs=(log_level, parsed.log_file),
        ) as pool:
            futures = {
                pool.submit(run_task, cfg.config_file, cycle, baseline, *task_args): (
                    cycle,
                    baseline,
                )
                for cycle, baseline in tasks
            }
            for future in as_completed(futures):
                cycle, baseline = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    # ie BrokenProcessPool when a worker process is killed (out of memory):
                    # record the failure and carry on, so the portal index is still updated
                    result = CycleResult(
                        cycle, baseline, "error", message=f"{type(exc).__name__}: {exc}"
                    )
                results.append(result)
                _log_result(result)
                last_index = _update_index_periodically(cfg, parsed, results, last_index)
    else:
        for cycle, baseline in tasks:
            results.append(run_task(cfg.config_file, cycle, baseline, *task_args))
            _log_result(results[-1])
            last_index = _update_index_periodically(cfg, parsed, results, last_index)

    n_status = {
        s: sum(r.status == s for r in results)
        for s in ("processed", "unchanged", "no_data", "error")
    }
    log.info("finished in %.0fs: %s", time.time() - t_start, n_status)
    if n_status["error"] > 0:
        failed = sorted((r.cycle, r.baseline) for r in results if r.status == "error")
        log.error(
            "%d cycle/baseline selections failed: %s. Re-run with --update to process them "
            "(completed cycles are skipped)%s",
            len(failed),
            ", ".join(f"{c}{b}" for c, b in failed[:20]) + (" ..." if len(failed) > 20 else ""),
            (
                ", with fewer --workers if workers ran out of memory"
                if any("BrokenProcessPool" in r.message for r in results)
                else ""
            ),
        )

    if not parsed.no_index and (n_status["processed"] > 0 or not parsed.update):
        build_portal_index(cfg)

    return 1 if n_status["error"] > 0 else 0


def _update_index_periodically(
    cfg: CsqaConfig, parsed: argparse.Namespace, results: list[CycleResult], last_index: float
) -> float:
    """rebuild the portal index if INDEX_UPDATE_SECONDS have passed since it was last built
    and cycles have been processed, so a long run's progress appears in the portal

    Returns:
        float: time the index was last built
    """
    if parsed.no_index or time.time() - last_index < INDEX_UPDATE_SECONDS:
        return last_index
    if not any(r.status == "processed" for r in results):
        return last_index
    try:
        build_portal_index(cfg)
        log.info("portal index updated (%d cycle/baseline selections done)", len(results))
    except Exception:  # pylint: disable=broad-exception-caught
        log.exception("failed to update the portal index")
    return time.time()


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

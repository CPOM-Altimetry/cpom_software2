"""
cpom.altimetry.tools.sec_tools.calculate_dhdt

Purpose:
    Compute surface elevation change rates (dh/dt) on a grid from epoch-averaged
    altimetry data using linear regression within configurable time windows.

    Uncertainty Modes:
        - Single-mission:  input_uncertainty + model_uncertainty
        - Multimission:    input_uncertainty + model_uncertainty + xcal_uncertainty
                       (requires --multi_mission flag and xcal std error column)

    Processing scope:
    - Icesheet-wide:   single root directory (--basin_structure False)
    - Basin-structured: subdirectories processed independently (--basin_structure True)
"""

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
from scipy.stats import linregress

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
    get_basins_to_process,
)
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    elapsed,
    get_algo_name,
    write_metadata,
)
from cpom.logging_funcs.logging import set_loggers


def parse_arguments(args: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for dh/dt calculation."""
    parser = argparse.ArgumentParser(
        description="Processor to calculate surface elevation change rate (dh/dt)"
    )
    # I/O Arguments
    parser.add_argument(
        "--in_step", type=str, required=False, help="Input algorithm step to source metadata from"
    )
    parser.add_argument(
        "--in_dir", type=str, required=True, help="Input data directory (epoch_average)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory Path",
    )
    parser.add_argument(
        "--parquet_glob",
        type=str,
        default="*/epoch_average.parquet",
        help="File glob pattern for selecting input files, relative to --in_dir.",
    )
    # dh/dt Calculation Parameters
    parser.add_argument(
        "--dhdt_start",
        help="Start time of the dh/dt period YYYY/MM/DD, Defaults to earliest date in dataset.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dhdt_end",
        help="End time of the dh/dt period YYYY/MM/DD, Defaults to latest date in dataset.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dhdt_period",
        help="Length of each dh/dt window. e.g. '2y', '2y1m', '2y1m12d'."
        "If not set, a single window spanning the full time range is used.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--step_length",
        help="Spacing between successive dh/dt window start times. Same format as --dhdt_period.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--min_pts_in_period",
        help="Minimum observations per grid cell and period to compute dh/dt.",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--min_period_coverage",
        help="Minimum %% of the dh/dt period spanned by data in a grid cell.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--max_allowed_dhdt",
        help="Maximum absolute dh/dt value allowed per period and grid cell.",
        type=int,
        default=30,
    )
    # Column Name Arguments
    parser.add_argument(
        "--dh_avg_varname", type=str, default="dh_ave", help="Column name for the dh average"
    )
    parser.add_argument(
        "--dh_stddev_varname",
        type=str,
        default="dh_stddev",
        help="Column name for the dh stddev",
    )
    parser.add_argument(
        "--dh_time_varname",
        type=str,
        default="epoch_midpoint",
        help="Column name for the epoch midpoint datetime",
    )
    parser.add_argument(
        "--dh_time_fractional_varname",
        type=str,
        default="epoch_midpoint_fractional_yr",
        help="Column name for the epoch midpoint as a fractional year",
    )
    # Multimission cross-calibration arguments
    parser.add_argument(
        "--multi_mission",
        help="Include cross-calibration uncertainty in the total uncertainty estimate.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--xcal_stderr_varname",
        help="Column name for cross-calibration standard error (multi-mission only).",
        type=str,
        default="biased_dh_xcal_stderr",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging")
    # Shared basin/region selection arguments
    add_basin_selection_arguments(parser)
    return parser.parse_args(args)


# ----------------------------------------------
# Setup Functions - Dates, Periods, Input Data
# ----------------------------------------------


def _has_glob_magic(path_pattern: str) -> bool:
    """Return True if the path pattern contains shell-style glob syntax."""
    return any(char in path_pattern for char in "*?[]")


def resolve_paths(
    params: argparse.Namespace,
    logger: logging.Logger,
    basin: str | None = None,
) -> tuple[str, Path]:
    """
    Build the input parquet glob path and output directory for a basin or the dataset root.

    Args:
        params: Parsed command-line parameters.
        logger: Logger object.
        basin: Basin subdirectory name, or None for ice-sheet-wide processing.

    Returns:
        tuple[str, Path]: (input glob pattern string, output directory Path)
    """
    in_base = Path(params.in_dir) / basin if basin else Path(params.in_dir)
    in_path = in_base / params.parquet_glob

    out_path = Path(params.out_dir) / basin if basin is not None else Path(params.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if _has_glob_magic(params.parquet_glob):
        logger.info(
            "Using glob pattern '%s' to select input files in %s", params.parquet_glob, in_base
        )
    else:
        logger.info("Using input file path '%s'", in_path)

    return str(in_path), out_path


def get_start_end_dates_for_calculation(
    input_df: pl.LazyFrame, dhdt_start: str | None, dhdt_end: str | None, dh_time_var: str
) -> tuple[datetime, datetime]:
    """
    Return the time range for dh/dt calculation.
    Uses provided start/end if given, otherwise infers from the dataset.
    Supports formats: YYYY/MM/DD and YYYY.MM.DD.

    Args:
        input_df (pl.LazyFrame): Polars LazyFrame of epoch-averaged elevation data.
        dhdt_start (str | None): Calculation start date (YYYY/MM/DD | YYYY.MM.DD) or None to infer.
        dhdt_end (str | None): Calculation end date (YYYY/MM/DD | YYYY.MM.DD) or None to infer.
        dh_time_var (str): Name of the datetime column in the input_df.

    Returns:
        tuple[datetime, datetime]:(start, end) datetimes for the calculation period.
    """

    def _get_date(date_str: str) -> datetime:
        if "/" in date_str:
            return datetime.strptime(date_str, "%Y/%m/%d")
        if "." in date_str:
            return datetime.strptime(date_str, "%Y.%m.%d")

        raise ValueError(f"Unrecognized date format: {date_str}, pass as YYYY/MM/DD or YYYY.MM.DD ")

    if dhdt_start is not None and dhdt_end is not None:
        return _get_date(dhdt_start), _get_date(dhdt_end)

    result = input_df.select(
        [
            pl.col(dh_time_var).min().alias("min_time"),
            pl.col(dh_time_var).max().alias("max_time"),
        ]
    ).collect()

    start_time, end_time = result["min_time"][0], result["max_time"][0]
    if start_time is None or end_time is None:
        raise ValueError(
            f"No non-null values found in '{dh_time_var}'. "
            "Check --in_dir/--parquet_glob and whether the selected files contain data."
        )

    return start_time, end_time


def get_period_limits_df(
    dhdt_period: str, step_length: str, dhdt_start: datetime, dhdt_end: datetime
) -> tuple[pl.DataFrame, float]:
    """
    Build a DataFrame of dh/dt calculation windows, given in fractional years.

    If step_length is provided, generates multiple overlapping or sequential windows
    spaced by step_length. Otherwise, returns a single window from dhdt_start: dhdt_end.
    Args:
        dhdt_period (str): Length of each window. Format 'ymd' e.g. '2y1m'.
        step_length (str):  Temporal spacing between successive window start times.
                            Format ymd e.g. "1y"
        dhdt_start (datetime): Start datetime for dh/dt calculation.
        dhdt_end (datetime): End datetime for dh/dt calculation.

    Returns:
        tuple[pl.DataFrame, float]:
            - DataFrame of windows with columns [period_lo, period_hi, period_id].
            - Window length in fractional years.
    """

    def _parse_period_string(period_str: str, as_string: bool = False) -> float | str:
        matches = re.findall(r"(\d+)([ymd])", period_str)
        tv = {"y": 0, "m": 0, "d": 0}
        for value, unit in matches:
            tv[unit] += int(value)
        if as_string:
            return f"{tv['y']} year {tv['m']} months {tv['d']} days"
        return tv["y"] + tv["m"] / 12 + tv["d"] / 365.25

    if step_length is None:
        usable_periods = pl.DataFrame(
            {"period_lo": dhdt_start, "period_hi": dhdt_end, "period_id": 1}
        )
        period_length = (dhdt_end - dhdt_start).days / 365.25
    else:
        con = duckdb.connect()
        usable_periods = con.execute(f"""
            WITH periods AS (
                SELECT 
                    period_lo, 
                    period_lo + INTERVAL '{_parse_period_string(dhdt_period, True)}' AS period_hi,
                    ROW_NUMBER() OVER (ORDER BY period_lo) AS period_id
                FROM 
                generate_series(DATE '{dhdt_start.strftime("%Y-%m-%d")}',
                DATE '{dhdt_end.strftime("%Y-%m-%d")}', INTERVAL
                '{_parse_period_string(step_length, True)}' ) AS t(period_lo)
            )
            SELECT * FROM periods
        """).pl()

        period_length = float(_parse_period_string(dhdt_period))
        con.close()
    return usable_periods, period_length


def get_input_df(
    input_df: pl.LazyFrame, dhdt_periods_df: pl.DataFrame, dh_time_var: str
) -> pl.DataFrame:
    """
    Join epoch-level data to dh/dt windows. Exclude observations that do not fall within a
    window.

    Args:
        input_df (pl.LazyFrame): Epoch average elevation data.
        dhdt_periods_df (pl.DataFrame): dh/dt windows with columns [period_lo, period_hi, period_id]
        dh_time_var (str): Name of the datetime variable in input_df.

    Returns (pl.DataFrame): Input data appended to corrosponding dh/dt window.
    """
    con = duckdb.connect()
    con.register("epoch_avg_df", input_df)
    con.register("usable_periods", dhdt_periods_df)
    dh_input = con.execute(f"""
        SELECT a.*, b.*,
        FROM epoch_avg_df a
        LEFT JOIN usable_periods b 
            ON a.{dh_time_var} BETWEEN b.period_lo AND b.period_hi
        WHERE b.period_lo IS NOT NULL 
        AND b.period_hi IS NOT NULL
        AND b.period_id IS NOT NULL
    """).pl()
    con.close()
    return dh_input


# ----------------------------------------------
# DH/DT Calculation
# ----------------------------------------------


def get_uncertainty(
    group: pl.DataFrame,
    params: argparse.Namespace,
    dhdt_period: float,
    input_uncertainty: float,
    model_uncertainty: float,
) -> tuple[float | None, float]:
    """
    Calculate total dh/dt uncertainty. Optionally include cross-calibration uncertainty.

    For single-mission data, combines input and model uncertainties.
    For multi-mission data, also includes the RMS cross-calibration standard error,
    normalised by the window length.

    Args:
        group (pl.DataFrame): Observations for a single grid cell and period.
        params (argparse.Namespace): Command line parameters.
        dhdt_period (float): Length of the dh/dt period in fractional years.
        input_uncertainty (float): Uncertainty derived from input elevation stddev.
        model_uncertainty (float): Uncertainty from the linear regression model.

    Returns:
        tuple[float | None, float]: (xcal_uncertainty, total_uncertainty).
            xcal_uncertainty is None for single-mission data.
    """
    xcal_uncertainty = None

    if params.multi_mission:
        xcal_arr = group[params.xcal_stderr_varname].to_numpy()
        xcal_uncertainty = (
            float(np.sqrt(np.mean(xcal_arr**2))) / dhdt_period if np.sum(xcal_arr) > 0.0 else 0.0
        )
        components = [xcal_uncertainty**2, model_uncertainty**2]
        if not np.isnan(input_uncertainty):
            components.append(input_uncertainty**2)
        total_uncertainty = float(np.sqrt(sum(components)))
    else:
        total_uncertainty = (
            model_uncertainty
            if np.isnan(input_uncertainty)
            else float(np.sqrt(input_uncertainty**2 + model_uncertainty**2))
        )

    return xcal_uncertainty, total_uncertainty


# pylint: disable=R0914
def get_dhdt(
    params: argparse.Namespace, dh_input: pl.DataFrame, dhdt_period: float
) -> tuple[list[dict], dict]:
    """
    Compute dh/dt and uncertainties for each grid cell and time window.

    For each (x_bin, y_bin, period_id) group:
        1. QC: minimum point count and temporal coverage checks
        2. Linear regression to estimate dh/dt (slope) and model uncertainty
        3. Outlier rejection: discard results exceeding max_allowed_dhdt
        4. Uncertainty: input, model, and (optionally) cross-calibration components

    Args:
        params (argparse.Namespace): Command Line Parameters
        dh_input (pl.DataFrame): Epoch-averaged data joined to dh/dt windows.
        dhdt_period (float): Window length in fractional years.

    Returns:
        [(list[dict], dict)]:
            -record_dhdt (list[dict]): List of dicts including: dh/dt, uncertainties and periods.
            -status (dict):  Metadata on processing outcomes.
    """

    record_dhdt = []
    status = {
        "no_input_data": 0,
        "fewer_datapoints_than_min_pts_in_period": 0,
        "time_coverage_less_than_min_period_coverage": 0,
        "calculated_dhdt_outside_max_allowed_dhdt_range": 0,
        "calculation_successful": 0,
    }

    # Loop through each grid cell and period
    for (x_bin, y_bin, period), group in dh_input.group_by(["x_bin", "y_bin", "period_id"]):

        # 1. Quality Control
        if len(group) == 0:
            status["no_input_data"] += 1
            continue

        if len(group) < params.min_pts_in_period:
            status["fewer_datapoints_than_min_pts_in_period"] += 1
            continue

        coverage_fraction = (
            group[params.dh_time_fractional_varname].max()
            - group[params.dh_time_fractional_varname].min()
        ) / dhdt_period
        if coverage_fraction * 100 < params.min_period_coverage:
            status["time_coverage_less_than_min_period_coverage"] += 1
            continue

        # 2. Perform linear regression to calculate dh/dt
        slope, icept, _, _, std_err = linregress(
            group[params.dh_time_fractional_varname], group[params.dh_avg_varname]
        )

        # 3. Outlier rejection: Check for dhdt outside allowed range
        if abs(slope) > params.max_allowed_dhdt:
            status["calculated_dhdt_outside_max_allowed_dhdt_range"] += 1
            continue

        # 4. Calculate uncertainties
        input_uncertainty = np.sqrt(np.nanmean(group[params.dh_stddev_varname] ** 2)) / dhdt_period
        model_uncertainty = std_err
        total_uncertainty = np.nan

        xcal_uncertainty, total_uncertainty = get_uncertainty(
            group, params, dhdt_period, input_uncertainty, model_uncertainty
        )

        record_dhdt.append(
            {
                "period": period,
                "x_bin": x_bin,
                "y_bin": y_bin,
                "dhdt": slope,
                "dhdt_incept": icept,
                "input_uncertainty": input_uncertainty,
                "model_uncertainty": model_uncertainty,
                "total_uncertainty": total_uncertainty,
                "input_dh_start_time": group[params.dh_time_varname].min(),
                "input_dh_end_time": group[params.dh_time_varname].max(),
                "period_lo": group["period_lo"].min(),
                "period_hi": group["period_hi"].max(),
                "num_pts_in_dhdt": len(group),
            }
        )
        if xcal_uncertainty is not None:
            record_dhdt.append({"xcal_uncertainty": xcal_uncertainty})

        status["calculation_successful"] += 1
    return record_dhdt, status


# ------------------------
# Metadata JSON
# ------------------------


def get_metadata_json(
    params: argparse.Namespace,
    start_time: float,
    basin: str | None,
    status: dict[str, Any] | None,
    logger,
) -> None:
    """
    Write processing metadata to JSON using the shared SEC entry-store format.

    Args:
        params (argparse.Namespace): Command line parameters.
        start_time (float): Processing start time.
        basin (str | None): Name of the basin being processed.
        status (dict[str, Any] | None): Metadata on processing outcomes from get_dhdt.
        logger: Logger object.
    """
    metadata: dict[str, Any] = {
        **{k: v for k, v in vars(params).items() if k != "algo"},
        "processed_basin": basin,
        "status": status,
        "execution_time": elapsed(start_time),
        "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_meta_dir = Path(params.out_dir) / basin if basin is not None else Path(params.out_dir)
    logger.info("Writing metadata to %s", out_meta_dir / f"{get_algo_name(__file__)}_meta.json")
    write_metadata(
        params,
        get_algo_name(__file__),
        out_meta_dir,
        metadata,
        basin_name=basin,
        logger=logger,
    )


# --------------------------
# Main Processing Workflow
# --------------------------
def process_dhdt(
    input_df: pl.LazyFrame,
    params: argparse.Namespace,
    out_path: Path,
    logger,
) -> dict[str, Any]:
    """
    Run the full dh/dt pipeline for one basin or the dataset root.
    Steps:
        1. Determine temporal range
        2. Generate dh/dt windows
        3. Assign observations to windows and compute dh/dt
        4. Write results to Parquet

    Args:
        input_df (pl.LazyFrame): Epoch-averaged input data for the basin/root.
        params (argparse.Namespace): Command line parameters.
        out_path (Path): Output directory path for the basin/root.
        logger: Logger object.
    """

    # 2. Determine the temporal range for dh/dt calculation
    dhdt_start, dhdt_end = get_start_end_dates_for_calculation(
        input_df, params.dhdt_start, params.dhdt_end, params.dh_time_varname
    )
    logger.info("dh/dt calculation time range: %s to %s", dhdt_start, dhdt_end)

    # 3. Generate one or more dh/dt calculation windows
    usable_periods, dhdt_period = get_period_limits_df(
        params.dhdt_period, params.step_length, dhdt_start, dhdt_end
    )
    logger.info("Calculated %d dh/dt periods for calculation.", len(usable_periods))

    # 4-5. Assign observations to windows and compute dh/dt with uncertainties
    record_dhdt, status = get_dhdt(
        params, get_input_df(input_df, usable_periods, params.dh_time_varname), dhdt_period
    )

    out_file = out_path / "dhdt.parquet"
    logger.info("Writing %d dh/dt records to: %s", len(record_dhdt), out_file)
    pl.DataFrame(record_dhdt).write_parquet(out_file, compression="zstd")

    return status


def calculate_dhdt(args: list[str]) -> None:
    """
    Main entry point for dh/dt calculation.

    Iterates over basin subdirectories or the full dataset root, and for each:
        - Reads epoch-averaged elevation data
        - Computes dh/dt per grid cell
        - Writes results and metadata to disk

    Uncertainty modes:
        - Single-mission: combines input measurement and regression model uncertainty.
        - Multi-mission:  also incorporates cross-calibration uncertainty.

    Args:
        args: List of command-line arguments.

    Output:
        - dhdt.parquet: Grid cell dh/dt values with uncertainty components
    """

    start_time = time.time()
    params = parse_arguments(args)

    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )

    # Get required columns
    base_columns = [
        "x_bin",
        "y_bin",
        params.dh_avg_varname,
        params.dh_stddev_varname,
        params.dh_time_varname,
        params.dh_time_fractional_varname,
    ]
    # Add cross-calibration uncertainty column if in multimission mode
    if params.multi_mission:
        base_columns.append(params.xcal_stderr_varname)

    # Get processing structure: basin-level or ice sheet-wide
    if params.basin_structure is True:
        for basin in get_basins_to_process(params, Path(params.in_dir), logger):
            basin_start = time.time()
            input_path, basin_out_dir = resolve_paths(
                params,
                logger,
                basin=basin,
            )
            logger.info("Processing basin '%s' from %s", basin, input_path)
            logger.info("Output directory: %s", basin_out_dir)

            input_df = pl.scan_parquet(input_path).select(base_columns)
            status = process_dhdt(input_df, params, basin_out_dir, logger)
            get_metadata_json(params, basin_start, basin, status, logger)
    else:
        input_path, out_dir_root = resolve_paths(params, logger)
        logger.info("Processing data in %s", input_path)
        input_df = pl.scan_parquet(input_path).select(base_columns)

        out_dir_root.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory:  %s ", out_dir_root)
        status = process_dhdt(input_df, params, out_dir_root, logger)
        get_metadata_json(params, start_time, None, status, logger)


if __name__ == "__main__":
    calculate_dhdt(sys.argv[1:])

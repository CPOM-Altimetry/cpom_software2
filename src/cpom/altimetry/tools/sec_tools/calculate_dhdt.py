"""
cpom.altimetry.tools.sec_tools.calculate_dhdt

Purpose:
    Calculate surface elevation change rates (dh/dt) from epoch-averaged elevation data.

    For each grid cell, computes dh/dt using linear regression within a moving time window.
    The window slides through the time series based on specified start/stop times,
    step length, and window size parameters.

Supported Structures:
    - root: Process all data at root level without subdirectories
    - single-tier: Direct basin subdirectories (e.g., basin1/, basin2/)
    - two-tier: Region/subregion hierarchy (e.g., West/H-Hp/, East/A-Ap/)

Output:
    - dh/dt data: <out_dir>/<basin>/dhdt.parquet
    - Per-basin stats: <out_dir>/<basin>/dhdt_statistics.json
    - Central metadata: <out_dir>/dhdt_metadata.json
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy.stats import linregress

from cpom.altimetry.tools.sec_tools.cross_calibrate_missions import (
    get_basins_to_process,
)
from cpom.logging_funcs.logging import set_loggers


def parse_arguments(args: list[str]) -> argparse.Namespace:
    """
    Parse command line arguments for dh/dt calculation.

    Args:
        args: List of command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            """
            Processor to take an epoch-averaged dh and 
            calculate surface elevation change rate in each grid cell, for given time limits.
            """
        )
    )

    parser.add_argument(
        "--in_dir",
        help=("Path of the epoch average dir containing parquet files"),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        help=("Path of the directory containing dh/dt output files"),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dhdt_start",
        help="Start time of first dh/dt period, format YYYY/MM/DD",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dhdt_end",
        help="End time of last dh/dt period, format YYYY/MM/DD",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dhdt_period",
        help="Time period for dhdt calculation format (e.g. 2y1m)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--step_length",
        help="amount of time to step each succeeding dh/dt period forward by format (e.g. 1y)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--min_pts_in_period",
        help="minimum number of datapoints needed in a cell/epoch for calculation to be performed",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--min_period_coverage",
        help="minimum percentage of the dh/dt period that must be spanned by data for calculation."
        "Eg If min_period_coverage is 60%, a 5 year dh/dt period must have at least 3 years diff,"
        " between dhdt_start/dhdt_end in each epoch/cell",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--max_allowed_dhdt",
        help="maximum allowed value for dh/dt in each epoch/cell",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--dh_avg_varname", help="Variable name for the dh average", type=str, default="dh_ave"
    )
    parser.add_argument(
        "--dh_stddev_varname",
        help="Variable name for the dh standard deviation",
        type=str,
        default="dh_stddev",
    )
    parser.add_argument(
        "--dh_time_varname",
        help="Variable name for the dh_time (The midpoint of each epoch)",
        type=str,
        default="epoch_midpoint",
    )
    parser.add_argument(
        "--dh_time_fractional_varname",
        help="Variable name for the dh_time as a fractional year",
        type=str,
        default="epoch_midpoint_fractional_yr",
    )
    parser.add_argument(
        "--multi_mission",
        help="Flag to enable multimission cross-calibration uncertainty calculation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--xcal_stderr_varname",
        help="Variable name for cross-calibration standard error (required for multimission)",
        type=str,
        default="biased_dh_xcal_stderr",
    )
    parser.add_argument(
        "--parquet_glob",
        help="Glob pattern to match parquet files within each sub-basin directory.",
        type=str,
        default="*/epoch_average.parquet",
    )
    return parser.parse_args(args)


def process_basin(
    basin_path: str,
    params: argparse.Namespace,
    base_columns: list[str],
    logger,
):
    """
    Process a single basin to calculate dh/dt.

    Steps :
        1. Load input data for the basin.
        2. Get start and end dates for dh/dt calculation.
        3. Get period limits
        4. Join input data with period limits
        5. Calculate dh/dt for each grid cell and period
        6. Write output parquet files and metadata json.
    Args:
        basin_path: Path to basin (or "None" for root level).
        params: Command line parameters.
        base_columns: List of required column names.
        logger: Logger object.

    """
    basin_start_time = time.time()

    if basin_path in ["None", ["None"], None]:
        logger.info("Loading root-level data from: %s", Path(params.in_dir) / params.parquet_glob)
        input_df = pl.scan_parquet(Path(params.in_dir) / params.parquet_glob).select(base_columns)
        output_path = Path(params.out_dir)
    else:
        input_path = Path(params.in_dir) / basin_path / params.parquet_glob
        logger.info("Loading data from: %s", input_path)
        input_df = pl.scan_parquet(input_path).select(base_columns)
        output_path = Path(params.out_dir) / basin_path

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory:  %s ", output_path)

    dhdt_start, dhdt_end = get_start_end_dates_for_calculation(
        input_df, params.dhdt_start, params.dhdt_end, params.dh_time_varname
    )

    usable_periods, dhdt_period = get_period_limits_df(
        params.dhdt_period, params.step_length, dhdt_start, dhdt_end
    )

    record_dhdt, status = get_dhdt(
        get_input_df(input_df, usable_periods, params.dh_time_varname), dhdt_period, params
    )

    logger.info("Writing %d dh/dt records to: %s", len(record_dhdt), output_path / "dhdt.parquet")
    pl.DataFrame(record_dhdt).write_parquet(output_path / "dhdt.parquet", compression="zstd")

    elapsed = int(time.time() - basin_start_time)
    status["basin_execution_time"] = (
        f"{elapsed // 3600:02}:" f"{(elapsed % 3600) // 60:02}:{elapsed % 60:02}"
    )

    write_basin_statistics(status, output_path, basin_path, logger)


def write_basin_statistics(
    status: dict,
    output_path: Path,
    basin_name: str,
    logger,
) -> None:
    """
    Write processing statistics for a specific basin to its output directory.

    Args:
        status (dict): Processing statistics (successful, failures, etc.).
        output_path (Path): Basin-specific output directory.
        basin_name (str): Name of the basin being processed.
        logger: Logger object.
    """
    stats_file = output_path / "dhdt_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    logger.info(f"Wrote statistics for {basin_name} to {stats_file}")


def write_central_metadata(
    params: argparse.Namespace,
    start_time: float,
    processed_basins: list[str],
    logger,
) -> None:
    """
    Write central metadata file with processing parameters and timing.

    Args:
        params (argparse.Namespace): Command line parameters.
        start_time (float): Processing start time.
        processed_basins (list): List of basins that were successfully processed.
        logger: Logger object.
    """
    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    metadata_file = Path(params.out_dir) / "dhdt_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f_meta:
        json.dump(
            {
                **vars(params),
                "processed_basins": processed_basins,
                "total_basins_processed": len(processed_basins),
                "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f_meta,
            indent=2,
        )
    logger.info(f"Wrote central metadata to {metadata_file}")


def get_start_end_dates_for_calculation(
    input_df: pl.LazyFrame, dhdt_start: str | None, dhdt_end: str | None, dh_time_var: str
) -> tuple[datetime, datetime]:
    """
    Get time range to use for dh/dt calculation.
    If time range start and/or end are provided, use them, otherwise use dataset time limits.
    Return the start and end dates as datetime objects.

    Args:
        input_df (pl.LazyFrame): Polars LazyFrame of epoch-averaged data.
        dhdt_start (str | None): Start time of first dh/dt period, format (YYYY/MM/DD | YYYY.MM.DD.)
        dhdt_end (str | None): End time of first dh/dt period, format (YYYY/MM/DD | YYYY.MM.DD.)
        dh_time_var (str): Name of the time variable in the dataset.

    Returns:
        tuple[datetime, datetime]: (dhdt_start, dhdt_end)
    """

    def _get_date(timedt: str) -> datetime:
        if "/" in timedt:
            time_dt = datetime.strptime(timedt, "%Y/%m/%d")
            return time_dt

        if "." in timedt:
            time_dt = datetime.strptime(timedt, "%Y.%m.%d")
            return time_dt

        raise ValueError(f"Unrecognized date format: {timedt}, pass as YYYY/MM/DD or YYYY.MM.DD ")

    if dhdt_start is not None and dhdt_end is not None:
        return _get_date(dhdt_start), _get_date(dhdt_end)
    result = input_df.select(
        [
            pl.col(dh_time_var).min().alias("min_time"),
            pl.col(dh_time_var).max().alias("max_time"),
        ]
    ).collect()

    return result["min_time"][0], result["max_time"][0]


def get_period_limits_df(
    dhdt_period: str, step_length: str, dhdt_start: datetime, dhdt_end: datetime
) -> tuple[pl.DataFrame, float]:
    """
    Calculates the period limits and the dhdt_period (in fractional years) for dh/dt calculation.

    Takes a dh/dt window size (dhdt_period, e.g. "2y1m"), and step size (step_length, e.g. "1y")
    and calculates a sequence of time intervals between `dhdt_start` and `dhdt_end`.
    Each interval is assigned period_id.

    If no dhdt_period and step_length are given,
    use the full time range in the dataset as a single period.

    Args:
        dhdt_period (str): Duration of dh/dt window, e.g. '2y1m'.
        step_length (str): The spacing between window start times, e.g. "1y"
        dhdt_start (datetime): Start datetime for dh/dt calculation.
        dhdt_end (datetime): End datetime for dh/dt calculation.

    Returns:
        tuple[pl.DataFrame, float]:
            usable_periods (pl.DataFrame): DataFrame with columns [period_lo, period_hi, period_id].
            dhdt_period (float): Length of dh/dt period in fractional years
    """

    def _parse_period_string(period_str: str, as_string: bool = False) -> float | str:
        matches = re.findall(r"(\d+)([ymd])", period_str)
        time_values = {"y": 0, "m": 0, "d": 0}
        for value, unit in matches:
            if unit in time_values:
                time_values[unit] += int(value)
        if as_string is False:
            return (
                int(time_values["y"])
                + int(time_values["m"]) / 12
                + int(time_values["d"]) / 365.25  # Julian year
            )
        return f"'{time_values['y']} year {time_values['m']} months {time_values['d']} days'"

    if step_length is None:
        usable_periods = pl.DataFrame(
            {"period_lo": dhdt_start, "period_hi": dhdt_end, "period_id": 1}
        )
        dhdt_period_value = (dhdt_end - dhdt_start).days / 365.25
    else:
        con = duckdb.connect()
        usable_periods = con.execute(
            f"""
            WITH periods AS (
                SELECT 
                    period_lo, 
                    period_lo + INTERVAL {_parse_period_string(dhdt_period, True)} AS period_hi,
                    ROW_NUMBER() OVER (ORDER BY period_lo) AS period_id
                FROM 
                generate_series(DATE '{dhdt_start.strftime("%Y-%m-%d")}',
                DATE '{dhdt_end.strftime("%Y-%m-%d")}', INTERVAL
                {_parse_period_string(step_length, True)} ) AS t(period_lo)
            )
            SELECT * FROM periods
        """
        ).pl()

        dhdt_period_parsed = _parse_period_string(dhdt_period)
        assert isinstance(dhdt_period_parsed, float)  # Type narrowing for mypy
        dhdt_period_value = dhdt_period_parsed
        con.close()
    return usable_periods, dhdt_period_value


def get_input_df(
    input_df: pl.LazyFrame, dhdt_periods_df: pl.DataFrame, dh_time_var: str
) -> pl.DataFrame:
    """
    Join epoch-level input data with pre-computed dh/dt period windows.

    Takes an input dataset (the output from epoch_average or interpolate_grids_of_dh)
    and assigns each record to a dh/dt period based on its epoch midpoint time.

    Filters out rows that do not fall inside a valid dh/dt period.

    Args:
        input_df (pl.LazyFrame): Input data.
        dhdt_periods_df (pl.DataFrame): DataFrame defining the dh/dt windows.
                                        Columsns: [period_lo, period_hi, period_id].
        dh_time_var (str): Name of the time variable in input_df.

    Returns (pl.DataFrame): DataFrame with input data joined to periods.
    """
    con = duckdb.connect()
    con.register("epoch_avg_df", input_df)
    con.register("usable_periods", dhdt_periods_df)
    dh_input = con.execute(
        f"""
        SELECT 
            a.*, 
            b.*,
        FROM epoch_avg_df a
        LEFT JOIN usable_periods b 
            ON a.{dh_time_var} BETWEEN b.period_lo AND b.period_hi
        WHERE b.period_lo IS NOT NULL AND b.period_hi IS NOT NULL
        AND b.period_id IS NOT NULL
    """
    ).pl()
    con.close()
    return dh_input


# pylint: disable=R0914
def get_dhdt(
    dh_input: pl.DataFrame, dhdt_period: float, params: argparse.Namespace
) -> tuple[list[dict], dict]:
    """
    Calculates dh/dt and uncertainties for each grid cell and dh/dt period.

    Steps:
    For each grid cell and period :
        1. Validate input data against criteria (min points, time coverage)
        2. Perform linear regression, to get :
            - dh/dt (slope)
            - intercept
            - model uncertainty (standard error of the regression)
        3. Calculate uncertainties :
            - input uncertainty - based on input dh stddev
            - cross-calibration uncertainty (if multi-mission)
            - total uncertainty

    Args:
        dh_input (pl.DataFrame): Joined epoch level data assigned to a period.
            Includes:
                - params.dh_time_fractional_varname (float)
                - params.dh_avg_varname (float)
                - params.dh_stddev_varname (float)
                - params.dh_time_varname (datetime)
                - x_bin, y_bin, period_id
                - period_lo, period_hi

        dhdt_period (float): Length of dh/dt period in fractional years.
        params (argparse.Namespace): Command Line Parameters

    Returns:
        [(list[dict], dict)]:
            -record_dhdt (list[dict]): List of dictionaries,
              one per successful (x_bin, y_bin, period_id).
                With, dh/dt results, uncertainties, period bounds, and metadata.
            -status (dict): Dictionary counting the outcome of each group:
    """

    def _get_uncertainty(group, params, dhdt_period, input_uncertainty, model_uncertainty):
        xcal_uncertainty = None
        total_uncertainty = None
        if params.multi_mission:
            xcal_stderr_array = group[params.xcal_stderr_varname].to_numpy()
            if xcal_stderr_array is not None and len(xcal_stderr_array) > 0:
                # Remove reference point (first element) as in original implementation
                # xcal_stderr_array = xcal_stderr - xcal_stderr[0] # Remove first data point
                if np.sum(xcal_stderr_array) > 0.0:
                    xcal_uncertainty = np.sqrt(np.mean((xcal_stderr_array) ** 2)) / dhdt_period
                else:
                    xcal_uncertainty = 0.0

                # Calculate total uncertainty with cross-calibration component
                if np.isnan(input_uncertainty):
                    total_uncertainty = np.sqrt(xcal_uncertainty**2 + model_uncertainty**2)
                else:
                    total_uncertainty = np.sqrt(
                        input_uncertainty**2 + xcal_uncertainty**2 + model_uncertainty**2
                    )
        else:
            # Single mission uncertainty calculation
            if np.isnan(input_uncertainty):
                total_uncertainty = model_uncertainty
            else:
                total_uncertainty = np.sqrt(input_uncertainty**2 + model_uncertainty**2)

        return xcal_uncertainty, total_uncertainty

    record_dhdt = []
    status = {
        "no_input_data": 0,
        "fewer_datapoints_than_min_pts_in_period": 0,
        "time_coverage_less_than_min_period_coverage": 0,
        "calculated_dhdt_outside_max_allowed_dhdt_range": 0,
        "calculation_successful": 0,
    }

    # Loop through each grid cell and period
    grouped = dh_input.group_by(["x_bin", "y_bin", "period_id"])
    for (x_bin, y_bin, period), group in grouped:

        # Check for no input data
        if len(group) == 0:
            status["no_input_data"] += 1
            continue

        num_pts_in_dhdt = len(group)
        # Check for fewer datapoints than minimum
        if num_pts_in_dhdt < params.min_pts_in_period:
            status["fewer_datapoints_than_min_pts_in_period"] += 1
            continue

        # Check for time coverage less than minimum
        coverage_fraction = (
            group[params.dh_time_fractional_varname].max()
            - group[params.dh_time_fractional_varname].min()
        ) / dhdt_period
        if coverage_fraction * 100 < params.min_period_coverage:
            status["time_coverage_less_than_min_period_coverage"] += 1
            continue

        # Perform linear regression to calculate dh/dt
        slope, icept, _, _, std_err = linregress(
            group[params.dh_time_fractional_varname], group[params.dh_avg_varname]
        )

        # Check for dhdt outside allowed range
        if abs(slope) > params.max_allowed_dhdt:
            status["calculated_dhdt_outside_max_allowed_dhdt_range"] += 1
            continue

        # Calculate uncertainties
        input_uncertainty = np.sqrt(np.nanmean(group[params.dh_stddev_varname] ** 2)) / dhdt_period
        model_uncertainty = std_err
        total_uncertainty = np.nan

        xcal_uncertainty, total_uncertainty = _get_uncertainty(
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
                "num_pts_in_dhdt": num_pts_in_dhdt,
            }
        )
        if xcal_uncertainty is not None:
            record_dhdt.append({"xcal_uncertainty": xcal_uncertainty})

        status["calculation_successful"] += 1
    return record_dhdt, status


def main(args: list[str]) -> None:
    """
    Main entry point for single mission dh/dt calculation with optional multimission support.

    Loads command-line arguments, reads metadata,
    performs dh/dt calculation, and writes output parquet and metadata JSON files.

    Supports two processing modes:
        1. Single mission: Uses input measurement and model fit uncertainties
        2. Multimission: Adds cross-calibration uncertainty component for bias-corrected data

    Args:
        args: List of command-line arguments.

    Output:
        - dhdt.parquet: Grid cell dh/dt values with uncertainty components
        - dhdt_statistics.json: Processing metadata and quality statistics per basin
        - dhdt_metadata.json: Central metadata with processing parameters
    """
    start_time = time.time()
    params = parse_arguments(args)

    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_file_info=str(Path(params.out_dir) / "info.log"),
        log_file_error=str(Path(params.out_dir) / "errors.log"),
        log_file_warning=str(Path(params.out_dir) / "warnings.log"),
    )

    # Determine required columns
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

    processed_basins = []

    if params.structure == "root":
        # Process root-level data without subdirectories
        logger.info("Processing root-level data")
        process_basin("None", params, base_columns, logger)
        processed_basins.append("root")
    else:
        basins_to_process = get_basins_to_process(params, Path(params.in_dir), logger)

        for basin_path in basins_to_process:
            process_basin(basin_path, params, base_columns, logger)
            processed_basins.append(basin_path)

    write_central_metadata(params, start_time, processed_basins, logger)


if __name__ == "__main__":
    main(sys.argv[1:])

"""
cpom.altimetry.tools.sec_tools.calculate_dhdt

Purpose:
    Calculate surface elevation change rates (dh/dt) on a grid, from epoch averaged altimetry
    elevation data.
    For each grid cell, dh/dt is estimated using linear regression within specified time windows.

    This tool supports single mission and multimission (cross-calibrated) datasets as input:
    - Single-mission: Processes data from one satellite mission
        Uncertainty: input_uncertainty + model_uncertainty

    - Multimission: Processes bias-corrected data from multiple missions (--multi_mission flag)
        Uncertainty: input_uncertainty + model_uncertainty + xcal_uncertainty
        Requires cross-calibration standard error column in input data

    And icesheet wide or basin_level processing:
    - Icesheet-wide: Processes all data in a single root directory (--basin_structure False)
    - Basin-structured: Processes subdirectories independently (--basin_structure True,
        --region_selector defined )
"""

import argparse
import json
import logging
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

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
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
    # I/O Arguments
    parser.add_argument(
        "--in_dir",
        help="Path of the epoch average data directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        help="Path of the directory to save dh/dt output files",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--parquet_glob",
        help="File glob pattern for selecting input files."
        "The pattern is applied relative to the input directory (--in_dir).",
        type=str,
        default="*/epoch_average.parquet",
    )
    # dh/dt Calculation Parameters
    parser.add_argument(
        "--dhdt_start",
        help="Start time of first dh/dt period, if not provided the earliest date in the dataset "
        "is used. Format YYYY/MM/DD",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dhdt_end",
        help="End time of last dh/dt period, if not provided the latest date in the dataset "
        "is used. Format YYYY/MM/DD",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dhdt_period",
        help="Length of each dh/dt calculation window. If not set a single dh/dt value is computed"
        "Format years(y), months(m) and days(d), e.g. '2y', '2y1m','2y1m12d'",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--step_length",
        help="Temporal spacing between successive dh/dt windows," "Format as --dhdt_period",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--min_pts_in_period",
        help="Minimum number of datapoints needed within a grid cell and dh/dt period"
        " to calculate dh/dt.",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--min_period_coverage",
        help="Minimum percentage of the dh/dt period that must be spanned by data in a grid cell."
        "E.g. Within a 5 year window, if min_period_coverage is 60%,the time difference must be at "
        "least 3 years between the earliest and latest data points to calculate dh/dt.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--max_allowed_dhdt",
        help="Maximum allowed value for dh/dt in each period and grid cell.",
        type=int,
        default=30,
    )
    # Column Name Arguments
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
    # Multimission cross-calibration arguments
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
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging",
    )
    # Shared basin/region selection arguments
    add_basin_selection_arguments(parser)
    return parser.parse_args(args)


# ----------------------------------------------
# Setup Functions - Dates, Periods, Input Data
# ----------------------------------------------


def get_start_end_dates_for_calculation(
    input_df: pl.LazyFrame, dhdt_start: str | None, dhdt_end: str | None, dh_time_var: str
) -> tuple[datetime, datetime]:
    """
    Get time range to use for dh/dt calculation.

    If dhdt_start or dhdt_end are provided they are used.
    Otherwise,the dataset temporal extent is used.

    Supported input date formats are ``YYYY/MM/DD`` and ``YYYY.MM.DD``.

    Args:
        input_df (pl.LazyFrame): Polars LazyFrame containing epoch-averaged elevation data.
        dhdt_start (str | None): Start time of dh/dt calculation period (YYYY/MM/DD | YYYY.MM.DD)
        dhdt_end (str | None): End time of dh/dt calculation period (YYYY/MM/DD | YYYY.MM.DD)
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
    Calculates the time intervals 'periods' over which dh/dt will be calculated. Defined by a fixed
    window length (dhdt_period) and optional step size (step_length) spanning from (dhdt_start) to
    (dhdt_end).Each interval is assigned period_id.

    If dhdt_period and step_length is provided multiple windows are generated.
    If step_length is not provided a single window spanning the full time range is used.

    Args:
        dhdt_period (str): Length of each dh/dt calculation window. Format 'ymd' e.g. '2y1m'.
        step_length (str):  Temporal spacing between successive window start times.
                            Format ymd e.g. "1y"
        dhdt_start (datetime): Start datetime for dh/dt calculation.
        dhdt_end (datetime): End datetime for dh/dt calculation.

    Returns:
        tuple[pl.DataFrame, float]:
            usable_periods (pl.DataFrame): DataFrame defining the dh/dt windows.
                                            Columns [period_lo, period_hi, period_id].
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
    Assign input data to dh/dt calculation windows.

    This function takes epoch-level input data and assigns each record to a dh/dt period based
    on its (dh_time_var).

    Observations that do not fall within any valid dh/dt window are excluded
    from the output.

    Args:
        input_df (pl.LazyFrame): Input data.
        dhdt_periods_df (pl.DataFrame): DataFrame defining the dh/dt windows.
                                        Columns: [period_lo, period_hi, period_id].
        dh_time_var (str): Name of the datetime variable in input_df.

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


# ----------------------------------------------
# DH/DT Calculation
# ----------------------------------------------


# pylint: disable=R0914
def get_dhdt(
    dh_input: pl.DataFrame, dhdt_period: float, params: argparse.Namespace
) -> tuple[list[dict], dict]:
    """
    Compute dh/dt and uncertainties for each grid cell and dh/dt period.

    Calculates dhdt for each grid cell and dh/dt period (period_id) using linear regression of
    epoch averaged elevation over time.

    For each grid cell and period :
        1. Quality control:
        - Require a minimum number of observations
        - Require a minimum temporal coverage relative to the window length
        2. Linear regression:
        - Estimate dh/dt (slope), intercept, and model uncertainty
        3. Outlier rejection:
        - Exclude results exceeding a maximum allowed absolute dh/dt
        4. Uncertainty estimation:
        - Input uncertainty derived from elevation standard deviations
        - Regression model uncertainty
        - Optional cross-calibration uncertainty for multi-mission data
        - Total uncertainty combined in quadrature

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
            -record_dhdt (list[dict]): List of dicts, one per successful (x_bin, y_bin, period_id).
            Includes : dh/dt results, uncertainties, period bounds, and metadata.
            -status (dict):  Dictionary summarizing processing outcome counts.
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

        # 1. Quality Control
        if len(group) == 0:
            status["no_input_data"] += 1
            continue

        num_pts_in_dhdt = len(group)
        if num_pts_in_dhdt < params.min_pts_in_period:
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


# ------------------------
# Metadata JSON
# ------------------------


def get_metadata_json(
    params: argparse.Namespace,
    start_time: float,
    processed_basins: list[str],
    logger,
) -> None:
    """
    Write metadata file with processing parameters and timing.

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
    logger.info("Wrote central metadata to %s", metadata_file)


# --------------------------
# Main Processing Workflow
# --------------------------


def main(args: list[str]) -> None:
    """
    Main processing workflow for dh/dt calculation.

    Parses command-line arguments, configures logging, iterates over one or more input datasets
    (either a single root dataset or multiple basin subdirectories), and writes
    dh/dt outputs and metadata to disk.

    For each basin or root dataset:
        1. Read epoch-averaged elevation data from Parquet files.
        2. Determine the temporal range for dh/dt calculation.
        3. Generate one or more dh/dt calculation windows.
        4. Assign observations to windows.
        5. Compute dh/dt and uncertainties per grid cell and window.
        6. Write dh/dt results and statistics to Parquet and JSON files.

    The function supports both single-mission and multi-mission processing:
    - Single-mission mode combines input measurement uncertainty and regression
      model uncertainty.
    - Multi-mission mode additionally incorporates cross-calibration uncertainty
      for bias-corrected datasets.

    Args:
        args: List of command-line arguments.

    Output:
        - dhdt.parquet: Grid cell dh/dt values with uncertainty components
        - dhdt_statistics.json: Processing metadata and quality statistics per basin
        - dhdt_metadata.json: Central metadata with processing parameters
    """

    def _process_dhdt(
        input_df: pl.LazyFrame,
        params: argparse.Namespace,
        out_path: Path,
        logger,
    ):
        """
        Main processing workflow for dh/dt calculation.

        Args:
            input_df (pl.LazyFrame): Epoch-averaged input data for the basin/root.
            params (argparse.Namespace): Command line parameters.
            out_path (Path): Output directory path for the basin/root.
            logger: Logger object.
        """
        basin_start_time = time.time()

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
            get_input_df(input_df, usable_periods, params.dh_time_varname), dhdt_period, params
        )

        logger.info("Writing %d dh/dt records to: %s", len(record_dhdt), out_path / "dhdt.parquet")
        pl.DataFrame(record_dhdt).write_parquet(out_path / "dhdt.parquet", compression="zstd")

        elapsed = int(time.time() - basin_start_time)
        status["basin_execution_time"] = (
            f"{elapsed // 3600:02}:" f"{(elapsed % 3600) // 60:02}:{elapsed % 60:02}"
        )

        stats_file = out_path / "dhdt_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        logger.info("Wrote statistics for %s to %s", out_path, stats_file)

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

    processed_basins = []

    # Get processing structure: basin-level or ice sheet-wide
    in_dir_root = Path(params.in_dir)
    out_dir_root = Path(params.out_dir)

    if params.basin_structure is True:
        paths = [
            (in_dir_root / basin, out_dir_root / basin, basin)
            for basin in get_basins_to_process(params, in_dir_root, logger)
        ]
    else:
        paths = [(in_dir_root, out_dir_root, "root")]

    # Loop through each basin/root dataset and execute processing workflow
    for in_path, out_path, basin in paths:
        # 1. Read epoch-averaged elevation data from Parquet files
        logger.info("Processing data in %s", in_path / params.parquet_glob)
        input_df = pl.scan_parquet(in_path / params.parquet_glob).select(base_columns)

        out_path.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory:  %s ", out_path)

        # 2-6. Execute the main dh/dt calculation workflow
        _process_dhdt(input_df, params, out_path, logger)
        processed_basins.append(basin)

    # Write central metadata file with processing parameters and timing
    get_metadata_json(params, start_time, processed_basins, logger)


if __name__ == "__main__":
    main(sys.argv[1:])

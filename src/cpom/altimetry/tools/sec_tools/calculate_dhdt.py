"""
cpom.altimetry.tools.sec_tools.single_mission_dhdt.py

Purpose:
Take epoch-averaged dh file and calculate surface elevation change rate in each grid cell,
for given time limits.
In each grid cell calculates dh/dt in a moving window defined by overall start and stop times,
step length and window size (the latter called dhdt_period).
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy.stats import linregress


def get_sub_basins(params: argparse.Namespace) -> list[str]:
    """
    Get available sub-basins by scanning the directory structure.

    - If params.sub_basins is None or ["None"], returns ["None"] and logs
        that root-level data will be processed.
    - If params.sub_basins is "all" or ["all"], returns a sorted list of
        sub-basins that exist.
        input_directory/
            sub_basin_1/
                *.parquet
            sub_basin_2/
                *.parquet
    - Otherwise, returns the list provided in params.sub_basins as-is.

    Args:
        params (argparse.Namespace): Command Line Parameters
            Includes:
                - in_dir (str): Path to the top-level directory that contains sub-basin
                    subdirectories.
                - sub_basins (list[str] | str): List of sub-basins to process,
                     "all" to auto-discover, 'None' to process root-level data.

    Returns:
        list[str]: list: Sorted list of sub-basin directory names to process.
    """

    if params.sub_basins in ("None", ["None"]):
        return ["None"]

    if params.sub_basins in ("all", ["all"]):
        all_sub_basins = set()

        if Path(params.in_dir).exists():
            # Look for subdirectories that contain parquet files
            for subdir in Path(params.in_dir).iterdir():
                if subdir.is_dir():
                    # Check if this subdirectory contains parquet files
                    parquet_files = list(subdir.glob("*.parquet"))
                    if parquet_files:
                        all_sub_basins.add(subdir.name)
        return sorted(all_sub_basins)
    return params.sub_basins


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


# pylint: disable=R0914
def main(args: list[str]) -> None:
    """
    Main entry point for single mission dh/dt calculation with optional multimission support.

    Loads command-line arguments, reads metadata,
    performs dh/dt calculation, and writes output parquet and metadata JSON files.

        Supports two processing modes:
        1. Single mission: Uses input measurement and model fit uncertainties
        2. Multimission: Adds cross-calibration uncertainty component for bias-corrected data

    Steps:
    1. Load command line arguments
    2. Load input fit dataframe and get start and end dates for calculation
    3. Calculate period limits
    4. Join periods with input data
    5. Calculate dh/dt for each grid cell and period with appropriate uncertainties
    6. Write output parquet file and metadata JSON

    Args:
        args: List of command-line arguments.

    Output:
        - dhdt.parquet: Grid cell dh/dt values with uncertainty components
        - dhdt_meta.json: Processing metadata and quality statistics
    """
    # ------------------------------#
    # 1.Load command line arguments#
    # ------------------------------#

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

    parser.add_argument(
        "--sub_basins",
        nargs="+",
        default=["all"],
        help="Specific list of sub-basins to process."
        "If not provided : Will calculate for entire directory. "
        "If set to 'all' will auto-discover sub-basins."
        "To process root-level data, set to [None].",
    )

    start_time = time.time()
    # -----------------------------------------#
    # 2. Load input dataframe and get start
    # and end dates for calculation
    # -----------------------------------------#
    params = parser.parse_args(args)
    # Load required variables as a dataframe
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

    sub_basins_to_process = get_sub_basins(params)

    for sub_basin in sub_basins_to_process:
        if sub_basin == "None":
            input_df = pl.scan_parquet(Path(params.in_dir) / params.parquet_glob).select(
                base_columns
            )
            output_path = Path(params.out_dir)
        else:
            input_df = pl.scan_parquet(
                Path(params.in_dir) / sub_basin / params.parquet_glob
            ).select(base_columns)
            output_path = Path(params.out_dir) / sub_basin

        output_path.mkdir(parents=True, exist_ok=True)

        dhdt_start, dhdt_end = get_start_end_dates_for_calculation(
            input_df, params.dhdt_start, params.dhdt_end, params.dh_time_varname
        )
        # ----------------------------#
        # 3. Calculate period limits #
        # ----------------------------#
        usable_periods, dhdt_period = get_period_limits_df(
            params.dhdt_period, params.step_length, dhdt_start, dhdt_end
        )
        # --------------------------------#
        # 4. Join periods with input data #
        # --------------------------------#
        dh_input = get_input_df(input_df, usable_periods, params.dh_time_varname)
        # ----------------------------------------------#
        # 5. Calculate dh/dt for each grid cell and period #
        # ---------------------------------------------#
        record_dhdt, status = get_dhdt(dh_input, dhdt_period, params)
        # ------------------------------------------------#
        # 6. Set output directory for dhdt , write data  #
        # ------------------------------------------------#
        pl.DataFrame(record_dhdt).write_parquet(output_path / "dhdt.parquet", compression="zstd")
        # ----------------------------#
        # 7. Write metadata json file #
        # ----------------------------#
        hours, remainder = divmod(int(time.time() - start_time), 3600)
        minutes, seconds = divmod(remainder, 60)

        with open(Path(output_path) / "dhdt_meta.json", "w", encoding="utf-8") as f_meta:
            json.dump(
                {
                    **vars(params),
                    **status,
                    "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
                },
                f_meta,
                indent=2,
            )


if __name__ == "__main__":
    main(sys.argv[1:])

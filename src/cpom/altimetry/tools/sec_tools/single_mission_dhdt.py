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
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy.stats import stats


def get_start_end_dates_for_calculation(
    input_df: pl.LazyFrame, dhdt_start: str, dhdt_end: str, dh_time_var: str
) -> tuple[datetime, datetime]:
    """
    Get time range to use for dh/dt calculation.
    If time range start and/or end are not None, use them, otherwise use dataset time limits.
    Return the start and end dates as datetime objects.

    Args:
        input_df (pl.LazyFrame): Polars LazyFrame of epoch-averaged data.
        dhdt_start (str): Start time of first dh/dt period, format (YYYY/MM/DD | YYYY.MM.DD.)
        dhdt_end (str): End time of first dh/dt period, format (YYYY/MM/DD | YYYY.MM.DD.)
        dh_time_var (str): Name of the time variable in the dataset.

    Returns:
        tuple[datetime]: (dhdt_start, dhdt_end)
    """

    def _get_date(timedt=None):
        if "/" in timedt:
            time_dt = datetime.strptime(timedt, "%Y/%m/%d")
            return time_dt

        if "." in timedt:
            time_dt = datetime.strptime(timedt, "%Y.%m.%d")
            return time_dt

        raise ValueError(f"Unrecognized date format: {timedt}, pass as YYYY/MM/DD or YYYY.MM.DD ")

    if dhdt_start is not None and dhdt_end is not None:
        dhdt_start = _get_date(dhdt_start)
        dhdt_end = _get_date(dhdt_end)
    else:
        result = input_df.select(
            [
                pl.col(dh_time_var).min().alias("min_time"),
                pl.col(dh_time_var).max().alias("max_time"),
            ]
        ).collect()

        dhdt_start = result["min_time"][0]
        dhdt_end = result["max_time"][0]

    return dhdt_start, dhdt_end


def get_period_limits_df(
    dhdt_period: str, step_length: str, dhdt_start: datetime, dhdt_end: datetime
) -> tuple[pl.DataFrame, float]:
    """
    Calculates the period limits and the dhdt_period (in fractional years) for dh/dt calculation.
    If no dh/dt period is given, use the full time range in the dataset.
    Args:
        dhdt_period (str): Length of dh/dt period for calculation as a string (e.g. '2y1m').
        step_length (str): Amount of time to step each succeeding dh/dt period forward by
                            as a string (e.g. '1y')
        dhdt_start (datetime): Start datetime for dh/dt calculation.
        dhdt_end (datetime): End datetime for dh/dt calculation.

    Returns:
        usable_periods (pl.DataFrame): Columns : period_lo, period_hi, period_id
        dhdt_period (float): Length of dh/dt period in fractional years
    """

    def _parse_period_string(period_str, as_string=False):
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
        dhdt_period = (dhdt_end - dhdt_start).days / 365.25
    else:
        con = duckdb.connect()
        usable_periods = con.execute(
            f"""
            SELECT 
                period_lo, 
                period_lo + INTERVAL {_parse_period_string(step_length, True)} AS period_hi,
                ROW_NUMBER() OVER () AS period_id,
            FROM 
            generate_series(DATE '{dhdt_start.strftime("%Y-%m-%d")}',
            DATE '{dhdt_end.strftime("%Y-%m-%d")}', INTERVAL
            {_parse_period_string(dhdt_period, True)} ) AS t(period_lo)
            WHERE --Remove empty periods
            period_lo > DATE '{dhdt_start.strftime("%Y-%m-%d")}'
            OR period_hi < DATE '{dhdt_end.strftime("%Y-%m-%d")}'
        """
        ).pl()

        dhdt_period = _parse_period_string(dhdt_period)
        con.close()
    return usable_periods, dhdt_period


def get_input_df(
    input_df: pl.LazyFrame, dhdt_periods_df: pl.DataFrame, dh_time_var: str
) -> pl.DataFrame:
    """
    Joins input data (e.g. output from : epoch_average, interpolate_grids_of_dh) with
    the calculated periods to be used for calculating dh/dt.
    Joining key: dh_time (the midpoint of the epoch) between period_lo and period_hi

    Args:
        input_df (pl.LazyFrame): Input data.
        dhdt_periods_df (pl.DataFrame): Period limits.
        dh_time_var (str): Name of the time variable in the dataset.

    Returns (pl.DataFrame):  Joined DataFrame with input data and period information.
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
def get_dhdt(dh_input: pl.DataFrame, dhdt_period: float, params) -> tuple[list[dict], dict]:
    """
    Calculates dh/dt and associated uncertainties for each grid cell and period.
    Populates status counters for calculation outcomes.

    Args:
        dh_input: Polars DataFrame containing joined epoch and period data.
        dhdt_period: Length of dh/dt period in fractional years.
        params: Namespace or dataclass with calculation thresholds.

    Returns:
        Tuple of:
            - record_dhdt: List of dictionaries with dh/dt results and uncertainties.
            - status: Dictionary with counts for each calculation status.
    """
    record_dhdt = []
    status = {
        "no_input_data": 0,
        "fewer_datapoints_than_min_pts_in_period": 0,
        "time_coverage_less_than_min_period_coverage": 0,
        "calculated_dhdt_outside_max_allowed_dhdt_range": 0,
        "calculation_successful": 0,
    }

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

        slope, icept, _, _, std_err = stats.linregress(
            group[params.dh_time_fractional_varname], group[params.dh_avg_varname]
        )

        # Check for dhdt outside allowed range
        if abs(slope) > params.max_allowed_dhdt:
            status["calculated_dhdt_outside_max_allowed_dhdt_range"] += 1
            continue

        input_uncertainty = np.sqrt(np.nanmean(group[params.dh_stddev_varname] ** 2)) / dhdt_period

        if np.isnan(input_uncertainty):
            total_uncertainty = std_err
        else:
            total_uncertainty = np.sqrt(input_uncertainty**2 + std_err**2)

        record_dhdt.append(
            {
                "period": period,
                "x_bin": x_bin,
                "y_bin": y_bin,
                "dhdt": slope,
                "dhdt_incept": icept,
                "input_uncertainty": input_uncertainty,
                "model_uncertainty": std_err,
                "total_uncertainty": total_uncertainty,
                "input_dh_start_time": group[params.dh_time_varname].min(),
                "input_dh_end_time": group[params.dh_time_varname].max(),
                "num_pts_in_dhdt": num_pts_in_dhdt,
            }
        )
        status["calculation_successful"] += 1
    return record_dhdt, status


# pylint: disable=R0914
def main(args: list[str]) -> None:
    """
    Main entry point for single mission dh/dt calculation.

    Loads command-line arguments, reads metadata, performs dh/dt calculation,
    writes output parquet and metadata JSON files.

    1. Load command line arguments
    2. Load input fit dataframe and get start and end dates for calculation
    3. Calculate period limits
    4. Join periods with input data
    5. Calculate dh/dt for each grid cell and period
    6. Set output directory for surface fit results

    Args:
        args: List of command-line arguments.

    Returns:
        None
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
        "--indir",
        help=("Path of the epoch average dir containing parquet files"),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        help=("Path of the directory containing dh/dt output files"),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--gridmeta_file",
        help="Path to the grid metadata file",
        required=False,
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

    start_time = time.time()
    # -----------------------------------------#
    # 2. Load input dataframe and get start
    # and end dates for calculation
    # -----------------------------------------#
    params = parser.parse_args(args)

    # Load required variables as a dataframe
    input_df = pl.scan_parquet(Path(params.indir) / "*.parquet").select(
        "x_bin",
        "y_bin",
        params.dh_avg_varname,
        params.dh_stddev_varname,
        params.dh_time_varname,
        params.dh_time_fractional_varname,
    )

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
    with open(params.gridmeta_file, "r", encoding="utf-8") as f:
        grid_dir = json.load(f)

    params.outdir = os.path.join(
        params.outdir,
        grid_dir["mission"],
        f'{grid_dir["gridarea"]}_{int(grid_dir["binsize"]/1000)}km_{grid_dir["mission"]}',
    )

    output_path = Path(params.outdir) / "dhdt.parquet"
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(record_dhdt).write_parquet(output_path, compression="zstd")

    # ----------------------------#
    # 7. Write metadata json file #
    # ----------------------------#
    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    try:
        with open(Path(params.outdir) / "dhdt_meta.json", "w", encoding="utf-8") as f_meta:
            json.dump(
                {
                    **vars(params),
                    **status,
                    "griddir": grid_dir["griddir"],
                    "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
                },
                f_meta,
                indent=2,
            )
    except OSError as e:
        sys.exit(e)


if __name__ == "__main__":
    main(sys.argv[1:])

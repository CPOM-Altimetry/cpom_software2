"""
cpom.altimetry.tools.sec_tools.epoch_average

Purpose:
    Compute epoch-averaged elevation values from gridded altimetry data.

    Divides the time series into epochs (time windows) and calculates mean elevation,
    time, and other statistics for each grid cell within each epoch. Optionally applies
    GIA (Glacial Isostatic Adjustment) corrections.

Output:
    - Epoch data: <out_dir>/epoch_average*.parquet (one or more files)
    - Metadata: <out_dir>/metadata.json
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from cpom.gias.gia import GIA
from cpom.gridding.gridareas import GridArea


def get_min_max_time(
    parquet_glob: str, epoch_time: datetime, time_var: str = "time"
) -> tuple[datetime, datetime]:
    """
    Get the minimum and maximum time in a parquet file.

    Args:
        parquet_glob (str): Glob pattern for the input parquet files.
        epoch_time (datetime): Reference epoch
        time_var (str): Time variable name

    Returns:
        tuple: Minimum and maximum datetime values in the parquet files.
    """

    df = pl.scan_parquet(parquet_glob).select(pl.col(time_var))

    min_max = df.select(
        [pl.col(time_var).min().alias("min_time"), pl.col(time_var).max().alias("max_time")]
    ).collect()

    min_dt = epoch_time + timedelta(seconds=min_max["min_time"][0])
    max_dt = epoch_time + timedelta(seconds=min_max["max_time"][0])
    return min_dt, max_dt


def get_date(epoch_time: datetime, timedt: str) -> tuple[datetime, float]:
    """
    Convert a date string (DD/MM/YYYY or DD.MM.YYYY) to a datetime object
    and seconds since epoch.

    Args:
        epoch_time (datetime): Reference epoch
        timedt (str): Time string to parse.

    Returns:
        tuple[datetime, float]: (datetime object, seconds since epoch_time)

    Raises:
        ValueError: If the time string is not recognized.
    """

    if "/" in timedt:
        time_dt = datetime.strptime(timedt, "%d/%m/%Y")
    elif "." in timedt:
        time_dt = datetime.strptime(timedt, "%d.%m.%Y")
    else:
        raise ValueError(f"Unrecognized date format: {timedt}, pass as YYYY/MM/DD or YYYY.MM.DD ")
    seconds = (time_dt - epoch_time).total_seconds()

    return time_dt, seconds


def get_epoch_df(
    params: argparse.Namespace,
    epoch_time: datetime,
    parquet_glob: str,
    time_var: str = "time",
) -> tuple[pl.LazyFrame, int]:
    """
    Create a polars LazyFrame defining the epoch intervals to use to calculate dh.

    Calculates the epoch start/end datetimes, spaced by the epoch_length.
    Only epochs overlapping the data are included.
    Gets the midpoint from each epoch range and converts the midpoint to a fractional year.

    Args:
        params (argparse.Namespace): Parsed command line arguments.
        epoch_time (datetime): Reference epoch time.
        parquet_glob (str): Glob pattern for the input parquet files.
        time_var (str): Name of the time variable. Defaults to "time".

    Returns:
        tuple[pl.LazyFrame, int]:
            - LazyFrame with columns: epoch_number, epoch_lo_dt, epoch_hi_dt,
              epoch_midpoint_dt, epoch_midpoint_fractional_yr
            - Total number of epochs (int)
    """
    # Get the min and max time from the parquet files.
    data_mintime, data_maxtime = get_min_max_time(parquet_glob, epoch_time, time_var)

    # If epoch_start and epoch_end are provided, convert them to datetime objects.
    # If not provided, use the reference epoch_time as a starting point and and the
    # datasets date end.
    if params.epoch_start is not None and params.epoch_end is not None:
        epoch_start_dt, _ = get_date(epoch_time, params.epoch_start)
        epoch_end_dt, _ = get_date(epoch_time, params.epoch_end)
    else:
        if params.epoch_start is not None:
            epoch_start_dt, _ = get_date(epoch_time, params.epoch_start)
        else:
            epoch_start_dt, _ = epoch_time, 0

        if params.epoch_end is not None:
            epoch_end_dt, _ = get_date(epoch_time, params.epoch_end)
        else:
            epoch_end_dt, _ = (
                data_maxtime,
                (data_maxtime - epoch_time).total_seconds(),
            )  # Use data end time

    total_period_days = (epoch_end_dt - epoch_start_dt).days
    total_epochs = math.ceil(total_period_days / params.epoch_length)

    epochs = (
        # Get the start date for each epoch
        pl.LazyFrame(
            {
                "epoch_lo_dt": pl.datetime_range(
                    start=epoch_start_dt,
                    end=epoch_end_dt,
                    interval=timedelta(days=params.epoch_length),
                    closed="left",  # extends beyond the end date to include the last epoch
                    eager=True,
                )
            }
        )
        # Get the end date and epoch number for each epoch
        .with_columns(
            (pl.col("epoch_lo_dt") + pl.duration(days=params.epoch_length)).alias("epoch_hi_dt"),
            pl.arange(0, pl.len()).alias("epoch_number"),
        )
        # Filter the epochs to only include those that overlap with the data time range
        .filter(pl.col("epoch_hi_dt") > data_mintime, pl.col("epoch_lo_dt") < data_maxtime)
        # Assign epoch numbers after filtering
        .with_columns(
            pl.arange(0, pl.len()).alias("epoch_number"),
        )
        # Calculate the midpoint datetime and fractional year of each epoch
        .with_columns(
            ((pl.col("epoch_lo_dt").cast(pl.Int64) + pl.col("epoch_hi_dt").cast(pl.Int64)) // 2)
            .cast(pl.Datetime("us"))
            .alias("epoch_midpoint_dt"),
        )
        .with_columns(
            (
                (  # Get number of whole years since epoch
                    pl.col("epoch_midpoint_dt").dt.year() - epoch_time.year
                )
                + (  # Add fractional part of current year
                    pl.col("epoch_midpoint_dt").dt.ordinal_day() - 1
                )
                /
                # Day of the year / days in the year (accounting for leap years)
                pl.when(
                    (pl.col("epoch_midpoint_dt").dt.year() % 4 == 0)
                    & (
                        (pl.col("epoch_midpoint_dt").dt.year() % 100 != 0)
                        | (pl.col("epoch_midpoint_dt").dt.year() % 400 == 0)
                    )
                )
                .then(366)
                .otherwise(365)
            ).alias("epoch_midpoint_fractional_yr")
        )
        .select(
            "epoch_number",
            "epoch_lo_dt",
            "epoch_hi_dt",
            "epoch_midpoint_dt",
            "epoch_midpoint_fractional_yr",
        )
    )
    return epochs, total_epochs


def get_gia_correction_df(gia_model: str, json_config: dict[str, Any]) -> pl.LazyFrame:
    """
    Get a Polars LazyFrame with GIA correction values for each grid cell.

    Interpolates GIA uplift values using the specified model and grid configuration.
    Returns a table with x_bin, y_bin, and uplift_value for each cell.

    Args:
        gia_model (str): Name of the GIA model from CPOM/gias/gia.py
        json_config (dict[str, Any]): Grid and metadata configuration.

    Returns:
        pl.LazyFrame: Columns: x_bin, y_bin, uplift_value.
    """
    gia = GIA(gia_model)
    grid = GridArea(json_config["gridarea"], json_config["binsize"])
    grid_x, grid_y = np.meshgrid(grid.cell_x_centres, grid.cell_y_centres)
    uplift_grid = gia.interp_gia(grid.crs_bng, grid.crs_wgs, grid_x, grid_y)
    y_idx, x_idx = np.indices(uplift_grid.shape)

    uplift_grid = pl.LazyFrame(
        {"x_bin": x_idx.ravel(), "y_bin": y_idx.ravel(), "uplift_value": uplift_grid.ravel()}
    )
    return uplift_grid


def get_data(
    params: argparse.Namespace, json_data: dict[str, Any], parquet_glob: str, epoch_time: datetime
) -> tuple[pl.DataFrame, int]:
    """
    Loads gridded surface fit data, assigns each row to an epoch interval,
    and applies an optional GIA correction.

    This function:
        - Gets epoch intervals 'get_epoch_df'
        - Optionally computes GIA uplift corrections for each grid cell 'get_gia_correction_df'.
        - Joins surface fit data with epoch intervals and GIA corrections.

    Args:
        params (argparse.Namespace): Input command line parameters.
        json_data (dict[str, Any]): JSON configuration data from surface fit metadata.
        parquet_glob (str): Path to the parquet files containing surface fit data.
        epoch_time (datetime): The epoch time to use as a reference.

    Returns:
        tuple[pl.DataFrame, int]:
            - Surface fit grid data with epoch and optional GIA correction
            - Total number of epochs (int)
    """

    # Get epoch intervals as a LazyFrame
    epochs_df, total_epochs = get_epoch_df(params, epoch_time, parquet_glob)
    # Get GIA corrections as a LazyFrame
    if params.gia_model is not None:
        uplift_grid = get_gia_correction_df(gia_model=params.gia_model, json_config=json_data)
    else:
        uplift_grid = None

    # Load the surface fit grid as  LazyFrame

    surface_fit_df = pl.scan_parquet(parquet_glob).with_columns(
        (pl.lit(epoch_time) + pl.duration(seconds=pl.col("time"))).alias("time_dt")
    )

    if uplift_grid is not None:
        dh_corrected_expr = "s.dh - time_delta_years * u.uplift_value AS dh_corrected"
        uplift_join = """
        LEFT JOIN uplift_grid u
            ON s.x_bin = u.x_bin AND s.y_bin = u.y_bin
        """
    else:
        dh_corrected_expr = "s.dh AS dh_corrected"
        uplift_join = ""  # No join needed

    query = f"""
    SELECT
        s.x_bin,
        s.y_bin,
        e.*,
        s.time_years,
        s.time_dt,
        MIN(s.time_years) OVER (PARTITION BY s.x_bin, s.y_bin, e.epoch_number) AS min_time,
        s.time_years - min_time AS time_delta_years,
        {dh_corrected_expr}
    FROM surface_fit_df s
    LEFT JOIN epochs_df e
        ON s.time_dt >= e.epoch_lo_dt AND s.time_dt < e.epoch_hi_dt
    {uplift_join}
    WHERE e.epoch_number IS NOT NULL
    """

    conn = duckdb.connect()
    conn.register("surface_fit_df", surface_fit_df)
    conn.register("epochs_df", epochs_df)
    if uplift_grid is not None:
        conn.register("uplift_grid", uplift_grid)
    surface_fit_grid_with_epoch = conn.execute(query).pl()

    return surface_fit_grid_with_epoch, total_epochs


def get_sigma_clipped_stats(
    df: pl.DataFrame,
    sigma: float = 3.0,
    max_iter: int = 5,
    group_by_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Get sigma clipped statistics for the dh_corrected column.
    Replicates the functionality of the sigma_clipped_stats function from astropy.stats
    using the default parameters.

    Args:
        df (pl.DataFrame): Input DataFrame containing elevation and time data.
        sigma (float): Number of standard deviations for clipping threshold. Default is 3.0.
        max_iter (int): Maximum number of clipping iterations. Default is 5.
        group_by_columns (list[str] | None): Columns to group by. Default is
            ["x_bin", "y_bin", "epoch_number", "epoch_lo_dt", "epoch_hi_dt"].

    Returns:
        pl.DataFrame: DataFrame with columns:
            - x_bin, y_bin: Grid cell coordinates
            - epoch_number, epoch_lo_dt, epoch_hi_dt: Epoch identifiers
            - dh_ave: Mean of clipped elevation values
            - dh_stddev: Standard deviation of clipped elevation values
            - dh_count: Number of values after clipping
            - dh_start_time: Earliest time in group
            - dh_end_time: Latest time in group
            - epoch_midpoint_dt: Midpoint of the epoch
            - epoch_midpoint_fractional_yr: Fractional year of the epoch midpoint
    """
    columns = df.columns
    for _ in range(max_iter):
        if group_by_columns is None:
            group_by_columns = ["x_bin", "y_bin", "epoch_number", "epoch_lo_dt", "epoch_hi_dt"]

        stats = df.group_by(group_by_columns).agg(
            [
                pl.col("dh_corrected").mean().alias("dh_mean"),
                pl.col("dh_corrected").median().alias("dh_median"),
                pl.col("dh_corrected").std(ddof=0).alias("dh_std"),
            ]
        )
        df = df.join(stats, on=group_by_columns).filter(
            (pl.col("dh_corrected") >= pl.col("dh_median") - sigma * pl.col("dh_std"))
            & (pl.col("dh_corrected") <= pl.col("dh_median") + sigma * pl.col("dh_std"))
        )

        df = df.select(columns)

    check = df.group_by(group_by_columns).agg(
        pl.col("epoch_midpoint_fractional_yr").n_unique().alias("n_unique")
    )

    if (check["n_unique"] > 1).any():
        raise ValueError("Groups have more than one unique epoch midpoint value.")

    # Aggregate statistics for each group
    result = df.group_by(group_by_columns).agg(
        [
            pl.col("dh_corrected").mean().alias("dh_ave"),
            pl.col("dh_corrected").std(ddof=0).alias("dh_stddev"),
            pl.col("dh_corrected").len().alias("dh_count"),
            pl.col("time_dt").min().alias("dh_start_time"),
            pl.col("time_dt").max().alias("dh_end_time"),
            pl.col("epoch_midpoint_fractional_yr")
            .unique()
            .first()
            .alias("epoch_midpoint_fractional_yr"),
            pl.col("epoch_midpoint_dt").unique().first().alias("epoch_midpoint_dt"),
        ]
    )
    return result


def get_metadata(
    stats: pl.DataFrame,
    args: argparse.Namespace,
    grid_meta: dict[str, Any],
    total_epochs: int,
    start_time: float,
) -> None:
    """
    Create and save epoch_avg_meta.json in the output directory.
    Metadata includes:
        - Command line parameters
        - Processing Status
        - Execution time

    Args:
        stats (pl.DataFrame): DataFrame containing epoch average statistics.
        args (argparse.Namespace): Command line arguments.
        grid_meta (dict[str, Any]): Grid metadata from surface fit.
        total_epochs (int): Total number of epochs processed.
        start_time (float): Start time of the processing.
    """

    epochs_with_data = stats.select("epoch_number").n_unique()
    ncells_with_data = stats.select(["x_bin", "y_bin"]).n_unique()

    # Ensure output directory exists
    output_path = Path(args.out_dir) / "epoch_average.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats.write_parquet(output_path, compression="zstd")

    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    with open(Path(args.out_dir) / "epoch_avg_meta.json", "w", encoding="utf-8") as f_meta:
        json.dump(
            {
                **vars(args),
                "standard_epoch": grid_meta["standard_epoch"],
                "total_epochs": total_epochs,
                "epochs_with_data": epochs_with_data,
                "cells_with_data": ncells_with_data,
                "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
            },
            f_meta,
            indent=2,
        )


def filter_epochs_by_coverage(stats: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """
    Filter epochs based on the coverage of grid cells.
    Only epochs with >= threshold fraction of the maximum number of grid cells are retained.

    Args:
        stats (pl.DataFrame): DataFrame containing epoch average statistics.
        threshold (float): Minimum fraction of grid cells required for an epoch to be retained .

    Returns:
        pl.DataFrame: Filtered DataFrame with epochs meeting the coverage criteria.
    """
    epoch_coverage = stats.group_by("epoch_number").agg(pl.col("x_bin").len().alias("length"))
    max_cells_per_epoch = epoch_coverage["length"].max()
    assert max_cells_per_epoch is not None  # Type narrowing for mypy
    assert isinstance(max_cells_per_epoch, (int, float))
    min_cells_threshold = max(1, int(threshold * max_cells_per_epoch))

    valid_epochs = epoch_coverage.filter(pl.col("length") >= min_cells_threshold).select(
        "epoch_number"
    )
    # Filter stats to only include valid epochs
    stats = stats.join(valid_epochs, on="epoch_number")
    return stats


def main(args: list[str] | None = None) -> None:
    """
    Get epoch averages for gridded surface fit data.

    1. Load command line arguments
    2. Load surface fit metadata
    3. Process data and compute epoch averages
    4. Apply optional GIA correction
    5. Calculate the average elevation over each gridcell
    6. Write results to a parquet file

    Args:
        args (list[str] | None): Command line arguments. If None, uses sys.argv[1:]
    """
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument(
        "--in_dir",
        help="Path to the directory containing surface fit data files.",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        help="Path to the output directory.",
        required=True,
    )
    parser.add_argument(
        "--grid_info_json",
        help="Path to the grid metadata JSON file.",
        required=True,
    )
    parser.add_argument(
        "--epoch_length",
        default=30,
        type=int,
        help="(integer, default=30), length of averaging epoch, in days",
    )
    parser.add_argument(
        "--epoch_start",
        help="(optional) start time of first epoch in DD/MM/YYYY format."
        " If not provided then default epoch_time is used (e.g. 01/01/1991)",
    )
    parser.add_argument(
        "--epoch_end",
        help="(optional) end time of last epoch in DD/MM/YYYY format. "
        "If not provided then latest time in grid is used. ",
    )
    parser.add_argument(
        "--gia_model",
        help="(optional) GIA model to use for correction. "
        "If not provided, no correction will be applied.",
    )

    parser.add_argument(
        "--epoch_filter_threshold",
        help="Filter out epochs with less than % of the number of points in the biggest epoch. ",
        type=float,
    )

    start_time = time.time()
    parsed_args = parser.parse_args(args)

    # Get surface fit metadata
    parquet_glob = f"{parsed_args.in_dir}/x_part=*/**/*.parquet"
    with open(Path(parsed_args.grid_info_json), "r", encoding="utf-8") as f:
        grid_meta = json.load(f)

    data, total_epochs = get_data(
        parsed_args, grid_meta, parquet_glob, datetime.fromisoformat(grid_meta["standard_epoch"])
    )
    stats = get_sigma_clipped_stats(data)

    stats = (
        filter_epochs_by_coverage(stats, threshold=parsed_args.epoch_filter_threshold)
        if parsed_args.epoch_filter_threshold is not None
        else stats
    )

    get_metadata(stats, parsed_args, grid_meta, total_epochs, start_time)


if __name__ == "__main__":
    main(sys.argv[1:])

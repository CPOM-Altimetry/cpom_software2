"""
cpom.altimetry.tools.sec_tools.epoch_average

Purpose:
    Compute epoch-averaged elevation values from the output of surface fitting altimetry data.

Overview:
    - Divides the time series into epochs (time windows) and calculates mean elevation,
        time, and other statistics for each grid cell within each epoch.
    - Optionally applies GIA (Glacial Isostatic Adjustment) corrections.
    - Can operate per grid partition (x_part/y_part) or over the whole dataset.

Run Modes:
    - Default (all-at-once): processes all input parquet files and writes a single
        combined file to <out_dir>/epoch_average.parquet.
    - Per grid partition (x_part/y_part): when '--partitioned' is provided, processes
        each x_part/y_part directory separately and writes outputs under
        <out_dir>/x_part=K/y_part=J/epoch_average.parquet. (Only recommended for is2 data)

Output:
    - Epoch data (combined or per grid partition): epoch_average.parquet
    - Metadata: <out_dir>/epoch_avg_meta.json (for all-at-once processing)
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
    get_basins_to_process,
)
from cpom.altimetry.tools.sec_tools.surface_fit import get_unique_chunks
from cpom.gias.gia import GIA
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers


def parse_arguments(args: list[str] | None) -> argparse.Namespace:
    """Parse command line arguments for epoch averaging.

    Args:
        args: Command line arguments

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Compute epoch-averaged elevation values from plane fitted altimetry data."
    )
    # I/O arguments
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
        "--parquet_glob",
        type=str,
        default="**/dh_time_grid.parquet",
        help="Glob pattern to match surface fit parquet files.",
    )
    parser.add_argument(
        "--grid_params_parquet_glob",
        type=str,
        default="**/*grid_data.parquet",
        help="Glob pattern to match grid parameter parquet files.",
    )
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help=(
            "Process per partition (x_part/y_part) instead of all-at-once. "
            "Use for large datasets that may not fit in memory."
        ),
    )

    # Epoch settings
    parser.add_argument(
        "--epoch_length",
        default=30,
        type=int,
        help="Length of averaging epoch, in days (default: 30).",
    )
    parser.add_argument(
        "--epoch_start",
        help="Start time of first epoch in DD/MM/YYYY format. Defaults to reference epoch.",
    )
    parser.add_argument(
        "--epoch_end",
        help="End time of last epoch in DD/MM/YYYY format. Defaults to data maximum.",
    )
    parser.add_argument(
        "--epoch_filter_threshold",
        help="Fractional coverage threshold (0.0-1.0) for retaining epochs.",
        type=float,
    )
    # Correction settings
    parser.add_argument(
        "--gia_model",
        help="GIA model name for isostatic correction. "
        "If not provided, no correction is applied.",
    )

    # Surface-fit filters
    parser.add_argument(
        "--abs_dhdt_limit",
        type=float,
        default=None,
        help="Maximum absolute dhdt. When None, no filtering is applied.",
    )
    parser.add_argument(
        "--rms_limit",
        type=float,
        default=None,
        help="Maximum RMS. When None, no filtering is applied.",
    )
    parser.add_argument(
        "--std_dev_limit",
        type=float,
        default=None,
        help="Maximum standard deviation. When None, no filtering is applied.",
    )
    parser.add_argument(
        "--slope_limit",
        type=float,
        default=None,
        help="Maximum slope. When None, no filtering is applied.",
    )

    # Basin/region selection
    add_basin_selection_arguments(parser)

    return parser.parse_args(args)


# ----------------------- #
# Date and time functions #
# ----------------------- #


def get_min_max_time(
    parquet_glob: str, epoch_time: datetime, logger: logging.Logger, time_var: str = "time"
) -> tuple[datetime, datetime]:
    """
    Get the minimum and maximum time in a parquet file.

    Args:
        parquet_glob (str): Glob pattern for the input parquet files.
        epoch_time (datetime): Reference epoch
        time_var (str): Time variable name
        logger (logging.Logger): Logger object

    Returns:
        tuple: Minimum and maximum datetime values in the parquet files.
    """

    df = pl.scan_parquet(parquet_glob).select(pl.col(time_var))

    min_max = df.select(
        [pl.col(time_var).min().alias("min_time"), pl.col(time_var).max().alias("max_time")]
    ).collect()

    min_dt = epoch_time + timedelta(seconds=min_max["min_time"][0])
    max_dt = epoch_time + timedelta(seconds=min_max["max_time"][0])
    logger.info(f"Data time range: {min_dt} to {max_dt}")
    return min_dt, max_dt


def get_date(epoch_time: datetime, timedt: str) -> tuple[datetime, float]:
    """
    Parse a date string and calculate seconds elapsed since the epoch.

    Accepts date strings in DD/MM/YYYY or DD.MM.YYYY format and converts them
    to a datetime object. Calculates the time difference (in seconds) between
    the parsed date and the reference epoch_time.

    Args:
        epoch_time (datetime): Reference epoch for computing time deltas.
        timedt (str): Date string in DD/MM/YYYY or DD.MM.YYYY format.

    Returns:
        tuple[datetime, float]:
            - datetime object representing the parsed date.
            - Seconds elapsed from epoch_time to the parsed date.

    Raises:
        ValueError: If timedt format is not DD/MM/YYYY or DD.MM.YYYY.
    """

    if "/" in timedt:
        time_dt = datetime.strptime(timedt, "%d/%m/%Y")
    elif "." in timedt:
        time_dt = datetime.strptime(timedt, "%d.%m.%Y")
    else:
        raise ValueError(f"Unrecognized date format: {timedt}, pass as YYYY/MM/DD or YYYY.MM.DD ")
    seconds = (time_dt - epoch_time).total_seconds()

    return time_dt, seconds


# ----------------------- #
# Data load functions   #
# ----------------------- #
def get_epoch_lf(
    params: argparse.Namespace,
    parquet_glob: str,
    logger: logging.Logger,
    time_var: str = "time",
) -> pl.LazyFrame:
    """
    Create a polars LazyFrame defining the epoch intervals to use to calculate dh.

    Calculates the epoch start/end datetimes, spaced by the epoch_length.
    Only epochs overlapping the data are included.
    Gets the midpoint from each epoch range and converts the midpoint to a fractional year.

    Args:
        params (argparse.Namespace): Command line arguments.
        parquet_glob (str): Glob pattern for the input parquet files.
        time_var (str): Name of the time variable. Defaults to "time".
        logger (logging.Logger): Logger object.

    Returns:
        tuple[pl.LazyFrame, int]:
            - LazyFrame with columns: epoch_number, epoch_lo_dt, epoch_hi_dt,
              epoch_midpoint_dt, epoch_midpoint_fractional_yr
    """
    # Get the min and max time from the parquet files.
    logger.info("Creating epoch dataframe")
    data_mintime, data_maxtime = get_min_max_time(parquet_glob, params.epoch_time, logger, time_var)

    # If epoch_start and epoch_end are provided, convert them to datetime objects.
    # If not provided, use the reference epoch_time as a starting point and and the
    # datasets date end.
    if params.epoch_start is not None and params.epoch_end is not None:
        epoch_start_dt, _ = get_date(params.epoch_time, params.epoch_start)
        epoch_end_dt, _ = get_date(params.epoch_time, params.epoch_end)
    else:
        if params.epoch_start is not None:
            epoch_start_dt, _ = get_date(params.epoch_time, params.epoch_start)
        else:
            epoch_start_dt, _ = params.epoch_time, 0

        if params.epoch_end is not None:
            epoch_end_dt, _ = get_date(params.epoch_time, params.epoch_end)
        else:
            epoch_end_dt, _ = (
                data_maxtime,
                (data_maxtime - params.epoch_time).total_seconds(),
            )  # Use data end time

    total_period_days = (epoch_end_dt - epoch_start_dt).days
    total_epochs = math.ceil(total_period_days / params.epoch_length)
    logger.info(
        f"Epoch params: {params.epoch_length} intervals ({epoch_start_dt} to {epoch_end_dt})"
    )

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
            [
                # Fractional year calculation
                (pl.col("epoch_midpoint_dt").dt.year() - params.epoch_time.year).alias(
                    "year_offset"
                ),
                (pl.col("epoch_midpoint_dt").dt.ordinal_day() - 1).alias("day_in_year"),
                pl.when(
                    (pl.col("epoch_midpoint_dt").dt.year() % 4 == 0)
                    & (
                        (pl.col("epoch_midpoint_dt").dt.year() % 100 != 0)
                        | (pl.col("epoch_midpoint_dt").dt.year() % 400 == 0)
                    )
                )
                .then(366)
                .otherwise(365)
                .alias("days_in_year"),
            ]
        )
        .with_columns(
            (pl.col("year_offset") + pl.col("day_in_year") / pl.col("days_in_year")).alias(
                "epoch_midpoint_fractional_yr"
            )
        )
        .drop(["year_offset", "day_in_year", "days_in_year"])
        .select(
            "epoch_number",
            "epoch_lo_dt",
            "epoch_hi_dt",
            "epoch_midpoint_dt",
            "epoch_midpoint_fractional_yr",
        )
    )
    logger.info(f"Created {total_epochs} epochs")
    return epochs


def get_gia_correction_lf(
    gia_model: str, json_config: dict[str, Any], logger: logging.Logger
) -> pl.LazyFrame:
    """
    Get a Polars LazyFrame with GIA correction values for each grid cell.

    Interpolates GIA uplift values using the specified model and grid configuration.
    Returns a table with x_bin, y_bin, and uplift_value for each cell.

    Args:
        gia_model (str): Name of the GIA model from CPOM/gias/gia.py
        json_config (dict[str, Any]): Grid and metadata configuration.
        logger (logging.Logger | None): Logger object

    Returns:
        pl.LazyFrame: Columns: x_bin, y_bin, uplift_value.
    """
    logger.info(f"Loading GIA correction data using model: {gia_model}")
    gia = GIA(gia_model)
    grid = GridArea(json_config["gridarea"], json_config["binsize"])
    grid_x, grid_y = np.meshgrid(grid.cell_x_centres, grid.cell_y_centres)
    uplift_grid = gia.interp_gia(
        grid_x, grid_y, {"crs_bng": grid.crs_bng, "crs_wgs": grid.crs_wgs}, method="linear"
    )
    y_idx, x_idx = np.indices(uplift_grid.shape)

    uplift_grid = pl.LazyFrame(
        {"x_bin": x_idx.ravel(), "y_bin": y_idx.ravel(), "uplift_value": uplift_grid.ravel()}
    )
    return uplift_grid


# ----------------------- #
# Data processing functions #
# ----------------------- #
def assign_epochs_lf(
    surface_fit_lf: pl.LazyFrame,
    epoch_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Assign each surface-fit observation to an epoch interval.

    Uses a range join to match time_dt against epoch boundaries, then computes
    time deltas within each (cell, epoch) group.

    Args:
        surface_fit_lf: Surface fit data with time_dt.
        epoch_lf: Epoch interval definitions.

    Returns:
        Surface fit data with epoch assignments and time deltas.
    """
    sf_with_epochs_lf = surface_fit_lf.join_where(
        epoch_lf,
        pl.col("time_dt") >= pl.col("epoch_lo_dt"),
        pl.col("time_dt") < pl.col("epoch_hi_dt"),
    ).filter(pl.col("epoch_number").is_not_null())

    # Compute time delta from start of epoch within each cell
    sf_with_epochs_lf = sf_with_epochs_lf.with_columns(
        pl.col("time_years").min().over(["x_bin", "y_bin", "epoch_number"]).alias("min_time"),
    ).with_columns(
        (pl.col("time_years") - pl.col("min_time")).alias("time_delta_years"),
    )

    return sf_with_epochs_lf


def apply_gia_lf(
    surface_fit_lf: pl.LazyFrame,
    uplift_grid_lf: pl.LazyFrame | None,
    logger: logging.Logger,
) -> pl.LazyFrame:
    """Apply GIA correction to surface-fit data if a GIA uplift grid is provided.

    Args:
        surface_fit_lf: Surface fit data with time_delta_years.
        uplift_grid_lf: GIA uplift values per grid cell.
        logger: Logger for progress messages.

    Returns:
        Surface fit data with dh_corrected column.
    """
    if uplift_grid_lf is not None:
        logger.info("Applying GIA correction")
        surface_fit_lf = surface_fit_lf.join(
            uplift_grid_lf, on=["x_bin", "y_bin"], how="left"
        ).with_columns(
            (pl.col("dh") - pl.col("time_delta_years") * pl.col("uplift_value")).alias(
                "dh_corrected"
            )
        )
    else:
        surface_fit_lf = surface_fit_lf.with_columns(pl.col("dh").alias("dh_corrected"))

    return surface_fit_lf.select(
        [
            "x_bin",
            "y_bin",
            "epoch_number",
            "epoch_lo_dt",
            "epoch_hi_dt",
            "epoch_midpoint_dt",
            "epoch_midpoint_fractional_yr",
            "time_years",
            "time_dt",
            "time_delta_years",
            "dh_corrected",
        ]
    )


def filter_surface_fit_lf(
    params: argparse.Namespace,
    surface_fit_lf: pl.LazyFrame,
    grid_params_glob: str | None,
) -> pl.LazyFrame:
    """
    Apply threshold filters to surface-fit data by excluding bad grid cells.
    Based on plane fit model residuals saves to grid parameter parquet files.
    Args:
        params: Command line arguments with filter thresholds.
        surface_fit_lf: Input surface fit data.
        grid_params_glob: Glob pattern for grid parameter parquet files.


    Returns:
        Filtered surface fit data with bad cells removed.
    """

    if all(
        v is None
        for v in (params.std_dev_limit, params.abs_dhdt_limit, params.rms_limit, params.slope_limit)
    ):
        return surface_fit_lf

    if not grid_params_glob:
        return surface_fit_lf

    parameters_grid_lf = pl.scan_parquet(grid_params_glob)
    filtered_frames: list[pl.LazyFrame] = []

    if params.std_dev_limit is not None:
        filtered_frames.append(
            parameters_grid_lf.filter(pl.col("sigma") > params.std_dev_limit).select(
                "x_bin", "y_bin"
            )
        )
    if params.abs_dhdt_limit is not None:
        filtered_frames.append(
            parameters_grid_lf.filter(pl.col("dhdt").abs() > params.abs_dhdt_limit).select(
                "x_bin", "y_bin"
            )
        )
    if params.rms_limit is not None:
        filtered_frames.append(
            parameters_grid_lf.filter(pl.col("rms") > params.rms_limit).select("x_bin", "y_bin")
        )
    if params.slope_limit is not None:
        filtered_frames.append(
            parameters_grid_lf.filter(pl.col("slope").abs() > params.slope_limit).select(
                "x_bin", "y_bin"
            )
        )

    if filtered_frames:
        bad_cells_lf = pl.concat(filtered_frames).unique()
        return surface_fit_lf.join(bad_cells_lf, on=("x_bin", "y_bin"), how="anti")

    return surface_fit_lf


def get_sigma_clipped_stats(
    surface_fit_lf: pl.LazyFrame,
    sigma: float = 3.0,
    max_iter: int = 5,
    group_by_columns: list[str] | None = None,
) -> pl.LazyFrame:
    """
    Get sigma clipped statistics for the dh_corrected column.
    Replicates the functionality of the sigma_clipped_stats function from astropy.stats
    using the default parameters.

    Args:
        lf (pl.LazyFrame): Input LazyFrame containing elevation and time data.
        sigma (float): Number of standard deviations for clipping threshold. Default is 3.0.
        max_iter (int): Maximum number of clipping iterations. Default is 5.
        group_by_columns (list[str] | None): Columns to group by. Default is
            ["x_bin", "y_bin", "epoch_number"].

    Returns:
        pl.LazyFrame: LazyFrame with columns:
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
    if group_by_columns is None:
        group_by_columns = ["x_bin", "y_bin", "epoch_number"]

    for _ in range(max_iter):
        surface_fit_lf = (
            surface_fit_lf.with_columns(
                pl.col("dh_corrected").median().over(group_by_columns).alias("dh_median"),
                pl.col("dh_corrected").std(ddof=0).over(group_by_columns).alias("dh_std"),
            )
            .filter(
                (pl.col("dh_corrected") >= pl.col("dh_median") - sigma * pl.col("dh_std"))
                & (pl.col("dh_corrected") <= pl.col("dh_median") + sigma * pl.col("dh_std"))
            )
            .drop(["dh_median", "dh_std"])
        )

    epoch_stats_lf = surface_fit_lf.group_by(group_by_columns).agg(
        [
            pl.col("dh_corrected").mean().alias("dh_ave"),
            pl.col("dh_corrected").std(ddof=0).alias("dh_stddev"),
            pl.col("dh_corrected").len().alias("dh_count"),
            pl.col("time_dt").min().alias("dh_start_time"),
            pl.col("time_dt").max().alias("dh_end_time"),
            pl.col("epoch_midpoint_fractional_yr").first().alias("epoch_midpoint_fractional_yr"),
            pl.col("epoch_midpoint_dt").first().alias("epoch_midpoint_dt"),
            pl.col("epoch_lo_dt").first().alias("epoch_lo_dt"),
            pl.col("epoch_hi_dt").first().alias("epoch_hi_dt"),
        ]
    )
    return epoch_stats_lf


def filter_epochs_by_coverage_lf(epoch_stats_lf: pl.LazyFrame, threshold: float) -> pl.LazyFrame:
    """Filter out epochs with insufficient spatial coverage.

    Only epochs with >= threshold fraction of the maximum number of grid cells are retained.

    Args:
        epoch_stats_lf: Epoch statistics LazyFrame.
        threshold: Minimum fraction of grid cells required (0.0 to 1.0).

    Returns:
        Filtered epoch statistics.
    """
    epoch_coverage_lf = epoch_stats_lf.group_by("epoch_number").agg(
        pl.col("x_bin").len().alias("n_cells")
    )
    max_cells_per_epoch = epoch_coverage_lf.select(pl.col("n_cells").max()).collect().item()
    min_cells_threshold = max(1, int(threshold * max_cells_per_epoch))

    valid_epochs_lf = epoch_coverage_lf.filter(pl.col("n_cells") >= min_cells_threshold).select(
        "epoch_number"
    )

    return epoch_stats_lf.join(valid_epochs_lf, on="epoch_number")


# ----------------------- #
# Data output functions  #
# ----------------------- #
def write_epoch_stats_lf(epoch_stats_lf: pl.LazyFrame, output_path: Path) -> None:
    """Write epoch statistics to parquet file.

    Args:
        epoch_stats_lf: Epoch statistics LazyFrame.
        output_path: Output parquet file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_stats_lf.sink_parquet(output_path, compression="zstd")


def get_metadata_json(
    stats: pl.LazyFrame,
    args: argparse.Namespace,
    grid_meta: dict[str, Any],
    start_time: float,
) -> None:
    """
    Create and save epoch_avg_meta.json in the output directory.
    Metadata includes:
        - Command line parameters
        - Processing Status
        - Execution time

    Args:
        stats (pl.LazyFrame): LazyFrame containing epoch average statistics.
        args (argparse.Namespace): Command line arguments.
        grid_meta (dict[str, Any]): Grid metadata from surface fit.
        start_time (float): Start time of the processing.
    """
    print("Writing metadata file")
    epochs_with_data = stats.select("epoch_number").collect().n_unique()
    ncells_with_data = stats.select(["x_bin", "y_bin"]).collect().n_unique()

    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    args_dict: dict[str, Any] = {}
    for k, v in vars(args).items():
        if isinstance(v, datetime):
            args_dict[k] = v.isoformat()
        elif isinstance(v, Path):
            args_dict[k] = str(v)
        else:
            args_dict[k] = v

    with open(Path(args.out_dir) / "epoch_avg_meta.json", "w", encoding="utf-8") as f_meta:
        json.dump(
            {
                **args_dict,
                "standard_epoch": grid_meta["standard_epoch"],
                "epochs_with_data": epochs_with_data,
                "cells_with_data": ncells_with_data,
                "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
            },
            f_meta,
            indent=2,
        )


# ----------------------- #
# Main processing flow   #
# ----------------------- #
def process_partition(
    params: argparse.Namespace,
    epoch_lf: pl.LazyFrame,
    uplift_lf: pl.LazyFrame | None,
    logger: logging.Logger,
) -> pl.LazyFrame:
    """
    Processing pipeline for one partition:
      1. Load surface-fit data
      2. Assign epochs via range join
      3. Apply optional GIA correction
      4. Filter bad surface fits by grid-parameter thresholds
      5. Compute sigma-clipped epoch statistics
      6. Filter epochs by spatial coverage

    Args:
        params: Command line parameters.
        epoch_lf: Pre-computed epoch intervals.
        uplift_lf: Pre-computed GIA uplift values or None.
        grid_meta: Grid metadata for GIA.
        logger: Logger for progress messages.

    Returns:
        Epoch statistics LazyFrame.
    """
    # 1. Load surface fit
    logger.info("Loading surface fit data")
    surface_fit_lf = pl.scan_parquet(params.parquet_glob).with_columns(
        (pl.lit(params.epoch_time) + pl.duration(seconds=pl.col("time"))).alias("time_dt")
    )

    # 2. Assign epochs
    logger.info("Assigning epochs")
    surface_fit_lf = assign_epochs_lf(surface_fit_lf, epoch_lf)

    # 3. Apply GIA correction
    logger.info("Applying GIA correction (if provided)")
    surface_fit_lf = apply_gia_lf(surface_fit_lf, uplift_lf, logger)

    # 4. Filter bad surface fits
    logger.info("Applying surface-fit quality filters")
    surface_fit_lf = filter_surface_fit_lf(
        params,
        surface_fit_lf,
        params.grid_params_glob,
    )

    # 5. Compute epoch statistics
    logger.info("Computing sigma-clipped epoch statistics")
    epoch_stats_lf = get_sigma_clipped_stats(surface_fit_lf)

    # 6. Filter epochs by coverage
    if params.epoch_filter_threshold is not None:
        logger.info(
            "Filtering epochs by spatial coverage (threshold=%.2f)", params.epoch_filter_threshold
        )
        epoch_stats_lf = filter_epochs_by_coverage_lf(
            epoch_stats_lf, threshold=params.epoch_filter_threshold
        )

    return epoch_stats_lf


# pylint: disable=R0914
def main(args: list[str] | None = None) -> None:
    """Main entry point for epoch averaging.

    Steps:
    1. Parse arguments and load grid metadata
    2. Set up logging
    3. Determine input/output paths (per-basin or root)
    4. Build epoch intervals once per dataset/basin
    5. Process each partition (or all data if non-partitioned)
    6. Write epoch statistics
    7. Write metadata
    """
    start_time = time.time()
    params = parse_arguments(args)

    # Load grid metadata
    with open(Path(params.grid_info_json), "r", encoding="utf-8") as f:
        grid_meta = json.load(f)

    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_file_info=str(Path(params.out_dir) / "info.log"),
        log_file_error=str(Path(params.out_dir) / "errors.log"),
        log_file_debug=str(Path(params.out_dir) / "debug.log"),
        log_file_warning=str(Path(params.out_dir) / "warnings.log"),
    )

    params.epoch_time = datetime.fromisoformat(grid_meta["standard_epoch"])
    logger.info("Standard epoch time: %s", params.epoch_time)

    # Get input/output paths
    in_dir_root = Path(params.in_dir)
    out_dir_root = Path(params.out_dir)

    if params.basin_structure is True:
        paths = [
            (in_dir_root / basin, out_dir_root / basin)
            for basin in get_basins_to_process(params, in_dir_root, logger)
        ]
    else:
        paths = [(in_dir_root, out_dir_root)]

    epoch_stats_lf = None
    uplift_lf = (
        get_gia_correction_lf(params.gia_model, grid_meta, logger) if params.gia_model else None
    )

    for in_dir, out_dir in paths:
        # Build epoch intervals once per basin/dataset
        logger.info("Get epoch intervals for %s", in_dir)
        epoch_lf = get_epoch_lf(params, f"{in_dir}/x_part=*/y_part=*/{params.parquet_glob}", logger)

        params.grid_params_glob = f"{in_dir}/{params.grid_params_parquet_glob}"

        if params.partitioned:
            for row in get_unique_chunks(params).iter_rows(named=True):
                logger.info(
                    "Processing partition x_part=%s, y_part=%s", row["x_part"], row["y_part"]
                )
                params.parquet_glob = (
                    f"{in_dir}/x_part={row['x_part']}/y_part={row['y_part']}/{params.parquet_glob}"
                )

                epoch_stats_lf = process_partition(
                    params,
                    epoch_lf,
                    uplift_lf,
                    logger,
                )

                output_path = out_dir / "epoch_average.parquet"
                write_epoch_stats_lf(epoch_stats_lf, output_path)
                logger.info("Wrote partition results to %s", output_path)

            # Write metadata once after all partitions - scan all written files for accurate stats
            logger.info("Computing metadata from all partitions")
            combined_stats_lf = pl.scan_parquet(str(out_dir / "**" / "epoch_average.parquet"))
            get_metadata_json(
                stats=combined_stats_lf,
                args=params,
                grid_meta=grid_meta,
                start_time=start_time,
            )
            logger.info("Metadata written successfully")

        else:
            logger.info("Processing all data.")
            params.parquet_glob = f"{in_dir}/x_part=*/y_part=*/{params.parquet_glob}"

            epoch_stats_lf = process_partition(
                params,
                epoch_lf,
                uplift_lf,
                logger,
            )

            output_path = out_dir / "epoch_average.parquet"
            write_epoch_stats_lf(epoch_stats_lf, output_path)
            logger.info("Wrote combined results to %s", output_path)

            get_metadata_json(
                stats=epoch_stats_lf,
                args=params,
                grid_meta=grid_meta,
                start_time=start_time,
            )
            logger.info("Metadata written successfully")


if __name__ == "__main__":
    main(sys.argv[1:])

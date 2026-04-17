"""
cpom.altimetry.tools.sec_tools.epoch_average

Purpose:
    Compute epoch-averaged elevation values from the output of surface fitting altimetry data.
    Divides the time series into epochs (time windows) and calculates mean elevation,
    time, and other statistics for each grid cell within each epoch.

    Optionally applies GIA (Glacial Isostatic Adjustment) corrections.

Run Modes:
    - Default: processes all input parquet files, writes <out_dir>/epoch_average.parquet.
    - Partitioned (--partitioned): processes each x_part/y_part directory separately,
      writes <out_dir>/x_part=K/y_part=J/epoch_average.parquet. (Recommended for IS2 data.)
Output:
    - Epoch data (combined or per grid partition): epoch_average.parquet
    - Metadata: <out_dir>/epoch_avg_meta.json (for all-at-once processing)
"""

import argparse
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
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    elapsed,
    get_algo_name,
    get_metadata_params,
    write_metadata,
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
    parser.add_argument("--in_dir", required=True, help="Directory containing surface fit data")
    parser.add_argument(
        "--in_step",
        type=str,
        required=False,
        help="Input algorithm step to source metadata from",
    )
    parser.add_argument("--out_dir", required=True, help="Path to the output directory.")
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
    parser.add_argument("--epoch_length", default=30, type=int, help="Epoch length in days.")
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
        type=float,
        help="Fractional coverage threshold (0.0-1.0) for retaining epochs.",
    )
    # Correction settings
    parser.add_argument("--gia_model", type=str, help="GIA model name. Omit to skip correction.")

    # Fall back if grid parameters are not provided in metadata
    parser.add_argument(
        "--standard_epoch",
        type=str,
        help="Reference epoch in ISO format (e.g., 1991-01-01T00:00:00).",
    )
    parser.add_argument("--gridarea", type=str, help="Grid area name for GIA interpolation.")
    parser.add_argument("--binsize", type=float, help="Grid bin size for GIA interpolation.")

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
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging")

    # Basin/region selection
    add_basin_selection_arguments(parser)

    return parser.parse_args(args)


# ----------------------- #
# Date and time functions #
# ----------------------- #


def get_date(epoch_time: datetime, timedt: str) -> tuple[datetime, float]:
    """Parse DD/MM/YYYY or DD.MM.YYYY date string; return datetime and seconds since epoch.

    Args:
        epoch_time (datetime): Reference epoch for computing time deltas.
        timedt (str): Date string in DD/MM/YYYY or DD.MM.YYYY format.

    Returns:
        tuple[datetime, float]:
            - datetime object representing the parsed date.
            - Seconds elapsed from epoch_time to the parsed date.
    """
    sep = "/" if "/" in timedt else "." if "." in timedt else None
    if sep is None:
        raise ValueError(f"Unrecognized date format: {timedt}. Use DD/MM/YYYY or DD.MM.YYYY.")
    fmt = "%d/%m/%Y" if sep == "/" else "%d.%m.%Y"
    dt = datetime.strptime(timedt, fmt)
    return dt, (dt - epoch_time).total_seconds()


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

    min_max = (
        pl.scan_parquet(parquet_glob)
        .select(
            [pl.col(time_var).min().alias("min_time"), pl.col(time_var).max().alias("max_time")]
        )
        .collect()
    )

    min_dt = epoch_time + timedelta(seconds=min_max["min_time"][0])
    max_dt = epoch_time + timedelta(seconds=min_max["max_time"][0])
    logger.info(f"Data time range: {min_dt} to {max_dt}")
    return min_dt, max_dt


# ----------------------- #
# Data load functions   #
# ----------------------- #
def get_epoch_lf(
    params: argparse.Namespace,
    parquet_glob: str,
    standard_epoch: datetime,
    logger: logging.Logger,
    time_var: str = "time",
) -> pl.LazyFrame:
    """
    Build a LazyFrame of epoch intervals covering the data time range

    Calculates the epoch start/end datetimes, spaced by the epoch_length.
    Only epochs overlapping the data are included.
    Gets the midpoint from each epoch range and converts the midpoint to a fractional year.

    Args:
        params (argparse.Namespace): Command line arguments.
        parquet_glob (str): Glob pattern for the input parquet files.
        standard_epoch (datetime): Reference epoch for computing time deltas.
        time_var (str): Name of the time variable. Defaults to "time".
        logger (logging.Logger): Logger object.

    Returns:
        tuple[pl.LazyFrame, int]:
            - LazyFrame with columns: epoch_number, epoch_lo_dt, epoch_hi_dt,
              epoch_midpoint_dt, epoch_midpoint_fractional_yr
    """
    # Get the min and max time from the parquet files.
    logger.info("Creating epoch dataframe")
    data_mintime, data_maxtime = get_min_max_time(parquet_glob, standard_epoch, logger, time_var)

    # If epoch_start and epoch_end are provided, convert them to datetime objects.
    # If not provided, use the reference epoch_time as a starting point and and the
    # datasets date end.
    epoch_start_dt = (
        get_date(standard_epoch, params.epoch_start)[0] if params.epoch_start else standard_epoch
    )
    epoch_end_dt = (
        get_date(standard_epoch, params.epoch_end)[0] if params.epoch_end else data_maxtime
    )
    total_epochs = math.ceil((epoch_end_dt - epoch_start_dt).days / params.epoch_length)
    logger.info(
        "Epoch params: %d-day intervals (%s to %s)",
        params.epoch_length,
        epoch_start_dt,
        epoch_end_dt,
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
        # Get the end date and absolute epoch number for each epoch
        .with_columns(
            (pl.col("epoch_lo_dt") + pl.duration(days=params.epoch_length)).alias("epoch_hi_dt"),
            pl.arange(0, pl.len()).alias("epoch_number"),
        )
        # Filter the epochs to only include those that overlap with the data time range
        .filter(pl.col("epoch_hi_dt") > data_mintime, pl.col("epoch_lo_dt") < data_maxtime)
        # Calculate the midpoint datetime and fractional year of each epoch
        .with_columns(
            ((pl.col("epoch_lo_dt").cast(pl.Int64) + pl.col("epoch_hi_dt").cast(pl.Int64)) // 2)
            .cast(pl.Datetime("us"))
            .alias("epoch_midpoint_dt"),
        )
        .with_columns(
            [
                # Fractional year calculation
                (pl.col("epoch_midpoint_dt").dt.year() - standard_epoch.year).alias("year_offset"),
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
    logger.info("Created %d epochs", total_epochs)
    return epochs


def get_gia_correction_lf(
    gia_model: str, grid_area: str, binsize: int, logger: logging.Logger
) -> pl.LazyFrame:
    """
    Get a Polars LazyFrame with GIA correction values for each grid cell.

    Interpolates GIA uplift values using the specified model and grid configuration.
    Returns a table with x_bin, y_bin, and uplift_value for each cell.

    Args:
        gia_model (str): Name of the GIA model from CPOM/gias/gia.py
        grid_area (str): Grid area name.
        binsize (int): Grid cell size in meters.
        logger (logging.Logger | None): Logger object

    Returns:
        pl.LazyFrame: Columns: x_bin, y_bin, uplift_value.
    """
    logger.info("Loading GIA correction data using model: %s", gia_model)
    gia = GIA(gia_model)
    grid = GridArea(grid_area, binsize)
    grid_x, grid_y = np.meshgrid(grid.cell_x_centres, grid.cell_y_centres)
    uplift_grid = gia.interp_gia(
        grid_x, grid_y, {"crs_bng": grid.crs_bng, "crs_wgs": grid.crs_wgs}, method="linear"
    )
    y_idx, x_idx = np.indices(uplift_grid.shape)
    return pl.LazyFrame(
        {"x_bin": x_idx.ravel(), "y_bin": y_idx.ravel(), "uplift_value": uplift_grid.ravel()}
    )


# ----------------------- #
# Data processing functions #
# ----------------------- #
def assign_epochs_lf(
    surface_fit_lf: pl.LazyFrame,
    epoch_lf: pl.LazyFrame,
    epoch_length_days: int,
) -> pl.LazyFrame:
    """Assign each observation to an epoch and compute its time delta within that epoch.

    Args:
        surface_fit_lf: Surface fit data with time_dt.
        epoch_lf: Epoch interval definitions.
        epoch_length_days: Length of each epoch in days.

    Returns:
        Surface fit data with epoch assignments and time deltas.
    """

    # Get the datetime of the first epoch
    epoch_bounds = epoch_lf.select(
        pl.col("epoch_lo_dt").min().alias("first_epoch_lo_dt"),
        pl.col("epoch_number").min().alias("first_epoch_number"),
    ).collect()

    epoch_origin_dt = epoch_bounds["first_epoch_lo_dt"][0] - timedelta(
        days=int(epoch_bounds["first_epoch_number"][0]) * epoch_length_days
    )

    epoch_length_seconds = epoch_length_days * 86400

    # Assign each observation an epoch number
    sf_with_epochs_lf = surface_fit_lf.with_columns(
        ((pl.col("time_dt") - epoch_origin_dt).dt.total_seconds() / epoch_length_seconds)
        .floor()
        .cast(pl.Int64)
        .alias("epoch_number")
    )
    # Join to get the epoch interval for each observation
    sf_with_epochs_lf = sf_with_epochs_lf.join(epoch_lf, on="epoch_number", how="inner")

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
    Remove grid cells exceeding quality thresholds derived from plane-fit residuals.
    Args:
        params: Command line arguments with filter thresholds.
        surface_fit_lf: Input surface fit data.
        grid_params_glob: Glob pattern for grid parameter parquet files.

    Returns:
        Filtered surface fit data with bad cells removed.
    """

    limits = {
        "sigma": params.std_dev_limit,
        "dhdt": params.abs_dhdt_limit,
        "rms": params.rms_limit,
        "slope": params.slope_limit,
    }

    if all(v is None for v in limits.values()):
        return surface_fit_lf

    if not grid_params_glob:
        return surface_fit_lf

    parameters_grid_lf = pl.scan_parquet(grid_params_glob)
    bad_cells = pl.concat(
        [
            parameters_grid_lf.filter(pl.col(col).abs() > limit).select("x_bin", "y_bin")
            for col, limit in limits.items()
            if limit is not None
        ]
    ).unique()

    return surface_fit_lf.join(bad_cells, on=("x_bin", "y_bin"), how="anti")


def get_sigma_clipped_stats(
    surface_fit_lf: pl.LazyFrame,
    logger: logging.Logger,
    sigma: float = 3.0,
    max_iter: int = 5,
    group_by_columns: list[str] | None = None,
) -> pl.LazyFrame:
    """
    Compute sigma-clipped mean elevation statistics per grid cell and epoch.
    Replicates the functionality of the sigma_clipped_stats function from astropy.stats
    using the default parameters.

    Args:
        surface_fit_lf (pl.LazyFrame): Input LazyFrame containing elevation and time data.
        logger (logging.Logger): Logger Object.
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

    for i in range(max_iter):
        logger.debug("Sigma-clipping iteration %d of %d", i + 1, max_iter)
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

    logger.debug("Grouping and aggregating clipped statistics")
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
    coverage = epoch_stats_lf.group_by("epoch_number").agg(pl.col("x_bin").len().alias("n_cells"))
    max_cells_per_epoch = coverage.select(pl.col("n_cells").max()).collect().item()
    min_cells = max(1, int(threshold * max_cells_per_epoch))
    valid_lf = coverage.filter(pl.col("n_cells") >= min_cells).select("epoch_number")
    return epoch_stats_lf.join(valid_lf, on="epoch_number")


# ----------------------- #
# Main processing flow   #
# ----------------------- #
def process_partition(
    params: argparse.Namespace,
    parquet_glob: str,
    standard_epoch: datetime,
    epoch_lf: pl.LazyFrame,
    uplift_lf: pl.LazyFrame | None,
    logger: logging.Logger,
) -> pl.LazyFrame:
    """
    Run the full epoch-averaging pipeline for one partition.

      1. Load surface-fit data
      2. Assign epochs via range join
      3. Apply optional GIA correction
      4. Filter bad surface fits by grid-parameter thresholds
      5. Compute sigma-clipped epoch statistics
      6. Filter epochs by spatial coverage

    Args:
        params: Command line parameters.
        parquet_glob: Glob pattern for input surface fit parquet files.
        standard_epoch: Reference epoch for computing time deltas.
        epoch_lf: Pre-computed epoch intervals.
        uplift_lf: Pre-computed GIA uplift values or None.
        logger: Logger for progress messages.

    Returns:
        Epoch statistics LazyFrame.
    """
    # 1. Load surface fit
    logger.info("Loading surface fit data from %s", parquet_glob)
    surface_fit_lf = pl.scan_parquet(parquet_glob).with_columns(
        (pl.lit(standard_epoch) + pl.duration(seconds=pl.col("time"))).alias("time_dt")
    )

    # 2. Assign epochs
    logger.info("Assigning epochs")
    surface_fit_lf = assign_epochs_lf(surface_fit_lf, epoch_lf, params.epoch_length)

    # 3. Apply GIA correction
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
    epoch_stats_lf = get_sigma_clipped_stats(surface_fit_lf, logger)

    # 6. Filter epochs by coverage
    if params.epoch_filter_threshold is not None:
        logger.info("Filtering epochs by coverage (threshold=%.2f)", params.epoch_filter_threshold)
        epoch_stats_lf = filter_epochs_by_coverage_lf(
            epoch_stats_lf, threshold=params.epoch_filter_threshold
        )

    return epoch_stats_lf


# ----------------------- #
# Data output functions  #
# ----------------------- #
def write_output(
    epoch_stats_lf: pl.LazyFrame, output_path: Path, logger: logging.Logger | None = None
) -> None:
    """Write epoch statistics to parquet file.

    Args:
        epoch_stats_lf: Epoch statistics LazyFrame.
        output_path: Output parquet file path.
        logger: Optional logger for write progress messages.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if logger is not None:
        logger.info("Writing epoch statistics to %s", output_path)
    epoch_stats_lf.sink_parquet(output_path, compression="zstd")
    if logger is not None:
        logger.info("Wrote in %.2fs", time.time() - t0)


def write_metadata_json(
    params: argparse.Namespace,
    stats: pl.LazyFrame,
    start_time: float,
    logger: logging.Logger,
) -> None:
    """
    Create and save epoch_avg_meta.json in the output directory.
    Metadata includes:
        - Command line parameters
        - Processing Status
        - Execution time

    Args:
        params (argparse.Namespace): Command line arguments.
        stats (pl.LazyFrame): LazyFrame containing epoch average statistics.
        start_time (float): Start time of the processing.
    """
    args_dict: dict[str, Any] = {
        k: v.isoformat() if isinstance(v, datetime) else str(v) if isinstance(v, Path) else v
        for k, v in vars(params).items()
    }

    meta_json_path = Path(params.out_dir)
    try:
        write_metadata(
            params,
            get_algo_name(__file__),
            meta_json_path,
            {
                **args_dict,
                "epochs_with_data": stats.select("epoch_number").collect().n_unique(),
                "cells_with_data": stats.select(["x_bin", "y_bin"]).collect().n_unique(),
                "execution_time": elapsed(start_time),
            },
        )
        logger.info("Metadata written to %s", params.out_dir)

    except OSError as e:
        logger.error("Failed to write surface_fit_meta.json with %s", e)


# ----------------
# Main Function #
# ----------------
# pylint: disable=R0914
def epoch_average(args: list[str] | None = None) -> None:
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

    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )

    try:
        grid_params = get_metadata_params(
            params,
            ["standard_epoch", "gridarea", "binsize"],
        )
        standard_epoch = datetime.fromisoformat(str(grid_params["standard_epoch"]))
        logger.info("Standard epoch time: %s", standard_epoch)
    except ValueError as exc:
        sys.exit(str(exc))

    # Get input/output paths
    in_dir, out_dir = Path(params.in_dir), Path(params.out_dir)
    paths = (
        [(in_dir / b, out_dir / b) for b in get_basins_to_process(params, in_dir, logger)]
        if params.basin_structure
        else [(in_dir, out_dir)]
    )

    uplift_lf = (
        get_gia_correction_lf(
            params.gia_model, str(grid_params["gridarea"]), int(grid_params["binsize"]), logger
        )
        if params.gia_model
        else None
    )

    for in_dir, out_dir in paths:
        # Build epoch intervals once per basin/dataset
        logger.info("Get epoch intervals for %s", in_dir)
        epoch_lf = get_epoch_lf(
            params, f"{in_dir}/x_part=*/y_part=*/{params.parquet_glob}", standard_epoch, logger
        )

        params.grid_params_glob = f"{in_dir}/{params.grid_params_parquet_glob}"

        if params.partitioned:
            for row in get_unique_chunks(params).iter_rows(named=True):
                x = row["x_part"]
                y = row["y_part"]

                logger.info("Processing partition x_part=%s, y_part=%s", x, y)
                glob = f"{in_dir}/x_part={x}/y_part={y}/{params.parquet_glob}"
                output_path = out_dir / f"x_part={x}/y_part={y}/epoch_average.parquet"

                write_output(
                    process_partition(
                        params,
                        glob,
                        standard_epoch,
                        epoch_lf,
                        uplift_lf,
                        logger,
                    ),
                    output_path,
                    logger,
                )

            write_metadata_json(
                params=params,
                stats=pl.scan_parquet(str(out_dir / "**" / "epoch_average.parquet")),
                start_time=start_time,
                logger=logger,
            )
            logger.info("Metadata written successfully")

        else:
            logger.info("Processing all data.")
            glob = f"{in_dir}/x_part=*/y_part=*/{params.parquet_glob}"
            output_path = out_dir / "epoch_average.parquet"
            write_output(
                process_partition(
                    params,
                    glob,
                    standard_epoch,
                    epoch_lf,
                    uplift_lf,
                    logger,
                ),
                output_path,
                logger,
            )

            write_metadata_json(
                params=params,
                stats=pl.scan_parquet(str(output_path)),
                start_time=start_time,
                logger=logger,
            )
            logger.info("Metadata written successfully")


if __name__ == "__main__":
    epoch_average(sys.argv[1:])

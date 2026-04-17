"""
cpom.altimetry.tools.sec_tools.clip_to_basins

Purpose:

    Clips data to basin, glacier, or region defined by a CPOM Mask class grid mask,
    by clipping points to mask label based on a 2km grid.

    Supports non-partitioned datasets after epoch_average.
    To process large partitioned datasets saved before epoch_average, use
    clip_to_basins_from_shapefile.py.

Supported Masks:
    Any grid mask from cpom.masks.mask_list that has grid_value_names defined.

Output:
    - Clipped Parquet data written per basin:
            <out_dir>/<basin_name>/data.parquet
    - Per-basin metadata written alongside each parquet:
        <out_dir>/<basin_name>/clip_to_basins_meta.json
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import polars as pl

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
)
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    elapsed,
    get_algo_name,
    get_metadata_params,
    write_metadata,
)
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers
from cpom.masks.masks import Mask


def parse_arguments(args: list[str]) -> argparse.Namespace:
    """
    Parse command line arguments for clipping altimetry data
    using the CPOM Mask class.

    Args:
        args: List of command-line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Clip altimetry data to regions defined by CPOM grid mask"
    )
    parser.add_argument(
        "--in_step",
        help="Input algorithm step to source metadata from",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Directory containing input altimetry data",
    )
    parser.add_argument(
        "--out_dir",
        help="Directory for output results",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--parquet_glob",
        type=str,
        default="**/*.parquet",
        help="File glob pattern for selecting input files.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="CPOM Mask Class to be used for clipping.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging",
    )
    # Fall back if grid parameters are not provided in metadata
    parser.add_argument(
        "--gridarea",
        type=str,
        required=False,
        help="Grid area name used for GIA interpolation when metadata is unavailable.",
    )
    parser.add_argument(
        "--binsize",
        type=float,
        required=False,
        help="Grid bin size used for GIA interpolation when metadata is unavailable.",
    )

    # Standardize basin selection arguments across tools
    add_basin_selection_arguments(parser)
    return parser.parse_args(args)


def get_data(grid_area: GridArea, infile: str, logger: logging.Logger) -> pl.LazyFrame:
    """
    Load data and add geographic coordinates.

    Add latitude/longitude coordinates for grid-cell centres to a dataset.

    Coordinates are derived from x_bin and y_bin indices using the GridArea
    definition and appended as new columns.

    Args:
        grid_area (GridArea): CPOM GridArea object for coordinate transformations.
        infile (str): Path to the input parquet file.
        logger (logging.Logger): Logger for progress messages.

    Returns:
        pl.LazyFrame: Epoch-averaged data with added 'lat' and 'lon' columns.
    """
    logger.info(f"Loading data from: {Path(infile)}")
    epoch_data = pl.scan_parquet(Path(infile))

    # Get unique cells
    unique_cells = epoch_data.select(["x_bin", "y_bin"]).unique().collect()

    # Compute lat/lon
    lats, lons = grid_area.get_cellcentre_lat_lon_from_col_row(
        unique_cells.get_column("x_bin").to_numpy(),
        unique_cells.get_column("y_bin").to_numpy(),
    )
    coords_df = unique_cells.with_columns(
        [
            pl.Series("lat", lats),
            pl.Series("lon", lons),
        ]
    ).lazy()

    # Join coordinates to data
    return epoch_data.join(
        coords_df,
        on=["x_bin", "y_bin"],
        how="left",
    )


def clip_data_to_shape(
    mask: Mask,
    basin_name: str,
    data: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Clip point data to a specific region defined by a CPOM Mask.

    Each row in the input data is assigned a mask label based on its latitude
    and longitude. Only rows corresponding to the specified basin_name
    (from mask.grid_value_names) are kept.

    Args:
        mask (Mask): CPOM Mask object for mask value lookup.
        basin_name (str): Target mask label (from mask.grid_value_names) to keep.
        data (pl.LazyFrame): Data with 'lat' and 'lon' columns.

    Returns:
        pl.LazyFrame: Clipped data containing only rows where mask_label matches
        basin_name.
    """
    if isinstance(data, pl.LazyFrame):
        data_df: pl.DataFrame = data.collect()
    else:
        data_df = data

    # Evaluate mask values
    mask_values = mask.grid_mask_values(
        data_df.get_column("lat").to_numpy(), data_df.get_column("lon").to_numpy()
    )

    def _label_for_value(mask: Mask, value: int, unknown_label: str) -> str:
        if 0 <= value < len(mask.grid_value_names):
            return mask.grid_value_names[value]
        return unknown_label

    labels = [_label_for_value(mask, int(v), "unknown") for v in mask_values]

    # Filter to target basin
    data_df = (
        data_df.with_columns(
            [
                pl.Series("mask_label", labels),
            ]
        )
        .filter(pl.col("mask_label") == basin_name)
        .drop("mask_label")
    )

    return data_df.lazy()


def process_single_basin(
    params: argparse.Namespace,
    mask: Mask,
    grid_area: GridArea,
    basin_name: str,
    logger: logging.Logger,
) -> dict[str, int | str]:
    """
    Clip altimetry data to a single basin and write results to disk.

    Args:
        params: Command line parameters.
            Includes:
            - in_dir: Input directory with gridded altimetry data.
            - out_dir: Output directory for clipped data.
            - parquet_glob: Glob pattern to match parquet files.
        mask: CPOM Mask object for mask value lookup.
        grid_area: CPOM GridArea object for coordinate transformations.
        basin_name: Name of the basin, used for output directory naming.
        logger: Logger object.

    Output:
        - Non-partitioned: <out_dir>/<basin_name>/data.parquet
        - Partitioned: <out_dir>/<basin_name>/x_part=<x>/y_part=<y>/data.parquet
                    (one file per non-empty partition)
    """
    logger.info(f"Clipping to: {basin_name}")
    output_dir = Path(params.out_dir) / basin_name
    output_dir.mkdir(parents=True, exist_ok=True)

    in_file = Path(params.in_dir) / params.parquet_glob
    data = get_data(grid_area, in_file, logger)
    clipped_data = clip_data_to_shape(mask, basin_name, data)
    stats_df = clipped_data.select(
        [
            pl.len().alias("n_rows"),
            pl.struct(["x_bin", "y_bin"]).n_unique().alias("n_unique_cells"),
        ]
    ).collect()

    output_file = output_dir / "data.parquet"
    clipped_data.drop(["lat", "lon"]).sink_parquet(output_file)
    logger.info(f"Wrote: {output_file}")

    return {
        "output_file": str(output_file),
        "n_rows": int(stats_df["n_rows"][0]),
        "n_unique_cells": int(stats_df["n_unique_cells"][0]),
    }


def get_metadata_json(
    params: argparse.Namespace,
    start_time: float,
    logger: logging.Logger,
    basin_name: str,
    basin_output_dir: Path,
    basin_stats: dict[str, int | str],
):
    """
    Generate per-basin metadata JSON for clipped data.

    Args:
        params (argparse.Namespace): Command line parameters.
        start_time (float): Start time.
        logger (logging.Logger): Logger object.
    """
    try:
        write_metadata(
            params,
            get_algo_name(__file__),
            basin_output_dir,
            {
                **dict(vars(params)),
                "basin_name": basin_name,
                **basin_stats,
                "execution_time": elapsed(start_time),
            },
            basin_name=basin_name,
            logger=logger,
        )
        logger.info("Wrote basin metadata to folder %s", basin_output_dir)

    except OSError as e:
        logger.error("Failed to write basin metadata with %s", e)


# ----------------#
# Main Function #
# ----------------#
def clip_to_basins(args):
    """
    Entry point for mask-based clipping of data to basins from CPOM
    Mask class grid.

    Steps:
        1. Parse command line arguments
        2. Set up output directory and logging
        3. Load grid metadata from JSON file
        4. Initialize GridArea and Mask objects
        5. Process and clip data by mask regions
        6. Write metadata JSON

    Args:
        args (list[str]): Command line arguments.
    """

    start_time = time.time()
    params = parse_arguments(args)
    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )

    try:
        grid_params = get_metadata_params(params=params, fields=["gridarea", "binsize"])
    except ValueError as exc:
        logger.error("Couldn't resolve required grid parameters: %s", exc)
        sys.exit(str(exc))

    mask = Mask(params.mask)
    # Clip to basin shape by bin centres
    for region in (
        mask.grid_value_names if params.region_selector == ["all"] else params.region_selector
    ):
        logger.info("Processing region: %s", region)
        output_dir = Path(params.out_dir) / region
        output_dir.mkdir(parents=True, exist_ok=True)

        basin_stats = process_single_basin(
            params,
            mask,
            GridArea(str(grid_params["gridarea"]), float(grid_params["binsize"])),
            region,
            logger,
        )
        get_metadata_json(params, start_time, logger, region, output_dir, basin_stats)


if __name__ == "__main__":
    clip_to_basins(sys.argv[1:])

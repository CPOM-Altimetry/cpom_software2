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
    - metadata.json summarising processing parameters and execution time
"""

import argparse
import json
import os
import sys
import time
from logging import Logger
from pathlib import Path

import polars as pl

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
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
        "--grid_info_json",
        help="Path to the grid metadata JSON file, if not passed is inferred",
        required=True,
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="CPOM Mask Class to be used for clipping.",
    )

    # Standardize basin selection arguments across tools
    add_basin_selection_arguments(parser)
    return parser.parse_args(args)


def get_data(grid_area: GridArea, infile: str, logger: Logger) -> pl.LazyFrame:
    """
    Load data and add geographic coordinates.

    Add latitude/longitude coordinates for grid-cell centres to a dataset.
    
    Coordinates are derived from x_bin and y_bin indices using the GridArea
    definition and appended as new columns.

    Args:
        grid_area (GridArea): CPOM GridArea object for coordinate transformations.
        infile (str): Path to the input parquet file.
        logger (Logger): Logger for progress messages.

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
    logger: Logger,
):
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
    clipped_data.drop(["lat", "lon"]).sink_parquet(output_dir / "data.parquet")
    logger.info(f"Wrote: {output_dir / 'data.parquet'}")


def get_metadata_json(params: argparse.Namespace, start_time, logger: Logger):
    """
    Generate metadata JSON for clipped data.

    Args:
        params (argparse.Namespace): Command line parameters.
        start_time (float): Start time.
        logger (Logger): Logger object.
    """
    meta_json_path = Path(params.out_dir) / "metadata.json"
    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    try:
        with open(meta_json_path, "w", encoding="utf-8") as f_meta:
            json.dump(
                {
                    **vars(params),
                    "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
                },
                f_meta,
                indent=2,
            )
        logger.info("Wrote data_set metadata to %s", meta_json_path)

    except OSError as e:
        logger.error("Failed to write surface_fit_meta.json with %s", e)


def main(args):
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
        log_file_info=Path(params.out_dir) / "info.log",
        log_file_error=Path(params.out_dir) / "errors.log",
        log_file_warning=Path(params.out_dir) / "warnings.log",
        log_file_debug=Path(params.out_dir) / "debug.log",
    )

    with open(Path(params.grid_info_json), "r", encoding="utf-8") as f:
        grid_meta = json.load(f)

    grid_area = GridArea(grid_meta["gridarea"], grid_meta["binsize"])
    mask = Mask(params.mask)

    # Clip to basin shape by bin centres
    for region in (
        mask.grid_value_names if params.region_selector == ["all"] else params.region_selector
    ):
        logger.info("Processing region: %s", region)
        output_dir = Path(params.out_dir) / region
        output_dir.mkdir(parents=True, exist_ok=True)

        process_single_basin(
            params,
            mask,
            grid_area,
            region,
            logger,
        )

    get_metadata_json(params, start_time, logger)


if __name__ == "__main__":
    main(sys.argv[1:])

"""
cpom.altimetry.tools.sec_tools.clip_to_basins_from_shapefile

Purpose:
    Clips altimetry data to glacier or basin boundaries defined in a shapefile.

    Spatially subset altimetry data to user-defined regions
    (e.g., glaciers, ice shelves, drainage basins) by clipping points to polygon geometries.
    Supports both non-partitioned datasets
        and large partitioned datasets processed incrementally.

Shapefile Requirements:
    - Polygon geometries defining basin/region boundaries
    - A column containing basin/region identifiers (name, ID, etc.)

Configuration Options:
    - Use a CPOM Mask class (if it includes shapefile metadata), or
    - Provide custom shapefile path and column name directly

Processing Modes:
    - Non-partitioned: Load full dataset into memory and clip to each basin
    - Partitioned: Process data one partition at a time (for large datasets saved
      before the epoch_average step), with bounding box prefiltering.

Output:
    - Clipped data: <out_dir>/<basin_name>/data.parquet
                   (Non-partitioned: single file per basin)
                   (Partitioned: multiple files per basin with x_part/y_part structure)
    - Metadata: <out_dir>/metadata.json (processing parameters and timing)
"""

import argparse
import json
import os
import sys
import time
from logging import Logger
from pathlib import Path

import geopandas as gpd
import polars as pl

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
)
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers
from cpom.masks.masks import Mask


def parse_arguments(args: list[str]) -> argparse.Namespace:
    """
    Parse command line arguments for clipping altimetry data to basins
    using a shapefile.

    Args:
        args: List of command-line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Clip altimetry data to glacier outlines from shapefile"
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Input gridded altimetry data" "e.g. from epoch_average.parquet",
    )
    parser.add_argument(
        "--out_dir",
        help="Path of output directory for clipped_epochs results",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--parquet_glob",
        type=str,
        default="**/*.parquet",
        help="Glob pattern to match parquet files in input directory.",
    )
    parser.add_argument(
        "--grid_info_json",
        help="Path to the grid metadata JSON file, if not passed is infered",
        required=True,
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="CPOM mask class. If the mask class has shapefile info, "
        "this will be used for clipping.",
    )
    parser.add_argument(
        "--shapefile", type=str, help="Path to the shapefile to use for clipping", required=False
    )
    parser.add_argument(
        "--shp_file_column",
        type=str,
        help="Column name in the shapefile to use for basin selection",
        required=False,
    )
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help="Optional flag for large partitioned datasets. "
        "If set, data will be loaded one partition and a bounding box filter applied "
        "before clipping to each basin shape.",
    )

    # Standardize basin selection arguments across tools
    add_basin_selection_arguments(parser)

    return parser.parse_args(args)


def clip_data_to_shape(
    grid_area: GridArea,
    subregion_shape: gpd.GeoDataFrame,
    data: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame:
    """
    Clip data to basin shape by filtering to bins whose centres lie
    within the basin shapefile.

    Args:
        grid_area (GridArea): CPOM GridArea object for coordinate conversion.
        subregion_shape (gpd.GeoDataFrame): GeoDataFrame of the basin or subregion shape.
        data (pl.LazyFrame | pl.DataFrame): Data with x_bin, y_bin columns.

    Returns:
        pl.LazyFrame: Clipped data containing only bins whose centres are within the basin.
    """
    # Get unique bins in the data
    unique_bins = data.select(["x_bin", "y_bin", "x", "y"]).unique(subset=["x_bin", "y_bin"])

    # Always collect to DataFrame for pandas operations
    unique_bins_df: pl.DataFrame = (
        unique_bins.collect() if isinstance(unique_bins, pl.LazyFrame) else unique_bins
    )

    # Create GeoDataFrame of bin centres
    bin_centres_gdf = gpd.GeoDataFrame(
        unique_bins_df.to_pandas(),
        geometry=gpd.points_from_xy(
            unique_bins_df["x"].to_numpy(),
            unique_bins_df["y"].to_numpy(),
        ),
        crs=grid_area.crs_bng,
    )

    # Spatial join to find bins whose centres are within the basin
    bins_in_basin = gpd.sjoin(bin_centres_gdf, subregion_shape, how="inner", predicate="within")[
        ["x_bin", "y_bin"]
    ].drop_duplicates()

    # Convert to Polars and filter original data
    filter_df = pl.from_pandas(bins_in_basin)
    if isinstance(data, pl.LazyFrame):
        return data.join(filter_df.lazy(), on=["x_bin", "y_bin"], how="inner").drop(["x", "y"])
    return data.lazy().join(filter_df.lazy(), on=["x_bin", "y_bin"], how="inner").drop(["x", "y"])


def add_coordinates_to_data(
    data: pl.LazyFrame,
    grid_area: GridArea,
) -> pl.LazyFrame:
    """
    Add cell centre x/y coordinates to data based on x_bin and y_bin columns.

    Args:
        data (pl.LazyFrame): Data with x_bin and y_bin columns.
        grid_area (GridArea): CPOM GridArea object for coordinate conversion.

    Returns:
        pl.LazyFrame: Data with added 'x' and 'y' columns.
    """
    x, y = grid_area.get_cellcentre_x_y_from_col_row(
        col=data.select(pl.col("x_bin")).collect().to_numpy().flatten(),
        row=data.select(pl.col("y_bin")).collect().to_numpy().flatten(),
    )
    return data.with_columns(
        [
            pl.Series("x", x),
            pl.Series("y", y),
        ]
    )


def load_partitioned_data_for_basin(
    grid_area: GridArea,
    params: argparse.Namespace,
    basin_shape: gpd.GeoDataFrame,
    logger: Logger,
):
    """
    Load partitioned Parquet data and filter by a basin bounding box.

    Iterates over partitioned Parquet files and yields one partition at a time,
    filtering by the basin extent so only a single partition is held in memory.
    If projected coordinates (`x`, `y`) are not present, they are derived from
    grid indices (`x_bin`, `y_bin`) using `GridArea`.

    Args:
        grid_area (GridArea): CPOM GridArea object.
        params (argparse.Namespace): Command line parameters.
            Includes:
            - in_dir (str): Directory containing partitioned parquet files
            - parquet_glob (str): Glob pattern for parquet files
        basin_shape (gpd.GeoDataFrame): Basin geometry to get bounds from.
        logger (Logger): Logger for progress messages.

    Yields:
        pl.DataFrame: Each partition with x/y coordinates added and filtered by basin bounds.
    """
    # Get basin bounds
    minx, miny, maxx, maxy = basin_shape.total_bounds
    logger.info(f"Basin bounds: x=[{minx}, {maxx}], y=[{miny}, {maxy}]")

    # Get all parquet files matching the glob pattern
    in_dir = Path(params.in_dir)
    parquet_files = sorted(in_dir.glob(params.parquet_glob))
    logger.info(f"Found {len(parquet_files)} parquet files to process")

    for file_idx, parquet_file in enumerate(parquet_files):
        logger.info(
            f"Processing partition {file_idx + 1}/{len(parquet_files)}: {parquet_file.name}"
        )
        schema = pl.scan_parquet(parquet_file).collect_schema()
        # Read partition with bounds filter
        if {"x", "y"}.issubset(schema):
            partition = (
                pl.scan_parquet(parquet_file)
                .filter(
                    (pl.col("x") >= minx)
                    & (pl.col("x") <= maxx)
                    & (pl.col("y") >= miny)
                    & (pl.col("y") <= maxy)
                )
                .collect()
            )
        else:
            partition_lf = add_coordinates_to_data(pl.scan_parquet(parquet_file), grid_area)
            partition = partition_lf.filter(
                (pl.col("x") >= minx)
                & (pl.col("x") <= maxx)
                & (pl.col("y") >= miny)
                & (pl.col("y") <= maxy)
            ).collect()

        # Skip empty partitions
        if partition.height == 0:
            logger.info(f"Partition {file_idx + 1} is empty after bounds filtering, skipping")
            continue

        yield partition


def process_single_basin(
    params: argparse.Namespace,
    grid_area: GridArea,
    basin_shape: gpd.GeoDataFrame,
    basin_name: str,
    logger: Logger,
):
    """
    Clip altimetry data to a single basin and save results to parquet files.

    Supports two processing modes:
    - Non-partitioned: If params.partitioned is False, clips the entire dataset to the basin
      and saves as a single parquet file.
    - Partitioned: If params.partitioned is True, loads and processes data one partition at a time,
      applying a bounding box filter before clipping, and saves each non-empty partition
      to its own parquet file with x_part and y_part directory structure.

    Args:
        params: Command line parameters.
        grid_area: CPOM GridArea
        basin_shape: GeoDataFrame containing the basin polygon geometry.
        basin_name: Name of the basin to clip to.
        logger: Logger object.

    Output:
        - Non-partitioned: <out_dir>/<basin_name>/data.parquet
        - Partitioned: <out_dir>/<basin_name>/x_part=<x>/y_part=<y>/data.parquet
                      (one file per non-empty partition)
    """
    logger.info(f"Clipping to: {basin_name}")
    output_dir = Path(params.out_dir) / basin_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not params.partitioned:
        in_file = Path(params.in_dir) / params.parquet_glob
        input_data = add_coordinates_to_data(pl.scan_parquet(in_file), grid_area)
        logger.info(f"Loading data from: {in_file}")
        # Clip to basin shape by bin centres
        clipped_data = clip_data_to_shape(grid_area, basin_shape, input_data)
        clipped_data.collect().write_parquet(output_dir / "data.parquet")
        logger.info(f"Wrote: {output_dir / 'data.parquet'}")
    else:
        for partition in load_partitioned_data_for_basin(grid_area, params, basin_shape, logger):
            # Clip this partition to the basin shape by bin centres
            clipped_chunk = clip_data_to_shape(grid_area, basin_shape, partition)

            # Count points after clipping
            clipped_height = clipped_chunk.select(pl.len()).collect().item()
            if clipped_height == 0:
                logger.info("Partition empty after clipping, skipping")
                continue

            logger.info(f"Clipped partition: {clipped_height} rows")
            # Save clipped partition
            chunked_out = (
                Path(output_dir)
                / f"x_part={partition['x_part'][0]}"
                / f"y_part={partition['y_part'][0]}"
            )
            chunked_out.mkdir(parents=True, exist_ok=True)
            clipped_chunk.sink_parquet(chunked_out / "data.parquet")


def get_metadata_json(params: argparse.Namespace, start_time, logger: Logger):
    """
    Generate metadata JSON for clipped data.
    Args:
        params (argparse.Namespace): Command line parameters.
            Includes:
                Command line parameters
                Processing time
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
    Entry point for clipping altimetry data to shapefile basins.
    Steps:
        1. Parse command line arguments
        2. Set up output directory and logging
        3. Load grid metadata from JSON file
        4. Initialize GridArea object
        5. Load shapefile (from CPOM mask or explicit path/column)
        6. (Non-partitioned) Load full dataset into memory
        7. Process and clip data by shapefile basins
        8. Write metadata JSON

    Args:
        args (list[str]): Command line arguments (typically sys.argv[1:]).
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

    if params.mask is None:
        shp = gpd.read_file(params.shapefile)
        selector = params.shp_file_column
        logger.info("Loaded shapefile with %d features from %s", len(shp), params.shapefile)
    else:
        # Load shapefile from CPOM Mask class
        mask = Mask(params.mask)

        if mask.shapefile_path is not None and mask.shapefile_column_name is not None:
            shp = gpd.read_file(mask.shapefile_path)
            selector = mask.shapefile_column_name
            logger.info("Loaded shapefile with %d features from %s", len(shp), mask.shapefile_path)
        else:
            logger.error("Mask class does not contain shapefile information.")
            sys.exit(1)

    for region in (
        set(shp[selector]) if params.region_selector == ["all"] else params.region_selector
    ):
        logger.info("Processing region: %s", region)
        region_shape = shp[shp[selector] == region]

        if not region_shape.empty:
            process_single_basin(params, grid_area, region_shape, region, logger)

    get_metadata_json(params, start_time, logger)


if __name__ == "__main__":
    main(sys.argv[1:])

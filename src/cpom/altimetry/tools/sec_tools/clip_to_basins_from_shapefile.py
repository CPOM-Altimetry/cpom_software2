"""
cpom.altimetry.tools.sec_tools.clip_to_basins_from_shapefile

Purpose:

    Clips data to basin, glacier, or region boundaries defined by polygon geometries
    in a shapefile, based on whether the cell centre lies within a polygon boundary.

    It supports both:
        - Non-partitioned datasets, loaded and processed in memory.
        - Large partitioned datasets, processed incrementally using bounding-box
        pre-filtering to reduce memory usage.

    It is reccommended to use this tool after epoch_average.

Shapefile requirements:
    - Polygon geometries defining basin/region boundaries
    - A column containing basin/region identifiers (name, ID, etc.)

Shapefile configuration options:
    - Use a CPOM Mask class that defines a shapefile and selector column.
    - Provide a shapefile path and selector column explicitly via CLI arguments.

Output:
    - Clipped Parquet data written per basin:
        * Non-partitioned:
            <out_dir>/<basin_name>/data.parquet
        * Partitioned:
            <out_dir>/<basin_name>/x_part=<x>/y_part=<y>/data.parquet
    - metadata.json summarising processing parameters and execution time
"""

import argparse
import json
import logging
import os
import sys
import time
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
    Parse command-line arguments for clipping altimetry data to basin polygons.

    Args:
        args: List of command-line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Clip altimetry data to basin polygons from shapefile"
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Directory containing input altimetry data",
    )
    parser.add_argument(
        "--out_dir",
        help="Directory for output clipped results",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging",
    )
    # Standardize basin selection arguments across tools
    add_basin_selection_arguments(parser)

    return parser.parse_args(args)


def add_coordinates_to_data(
    data: pl.LazyFrame,
    grid_area: GridArea,
) -> pl.LazyFrame:
    """
    Add projected x/y coordinates for grid-cell centres to a dataset.

    Coordinates are derived from x_bin and y_bin indices using the GridArea
    definition and appended as new columns.

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
    logger: logging.Logger,
):
    """
    Incrementally load partitioned Parquet data filtered to a basin bounding box.

    Each partition is read independently, spatially filtered using the basin
    bounding box, and yielded as a DataFrame. This avoids loading the full
    dataset into memory.

    If projected coordinates are not present, they are derived from grid indices.

    Args:
        grid_area (GridArea): CPOM GridArea object.
        params (argparse.Namespace): Command line parameters.
            Includes:
            - in_dir (str): Directory containing partitioned parquet files
            - parquet_glob (str): Glob pattern for parquet files
        basin_shape (geopandas.GeoDataFrame): Basin geometry used to determine bounding-box limits.
        logger (logging.Logger): Logger for progress and status messages.

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


def clip_data_to_shape(
    grid_area: GridArea,
    subregion_shape: gpd.GeoDataFrame,
    data: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame:
    """
    Spatially clip data to a basin polygon using grid-cell centre locations.

    Args:
        grid_area (GridArea): CPOM GridArea object for coordinate conversion.
        subregion_shape ( geopandas.GeoDataFrame ):Basin or region polygon geometry.
        data (pl.LazyFrame | pl.DataFrame): Input data containing [x_bin, y_bin, x , y] columns.

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


def process_single_basin(
    params: argparse.Namespace,
    grid_area: GridArea,
    basin_shape: gpd.GeoDataFrame,
    basin_name: str,
    logger: logging.Logger,
):
    """
    Clip altimetry data to a single basin and write results to disk.

    Supports two processing modes:
    - Non-partitioned:
        The full dataset is loaded, clipped, and written as a single Parquet file.
    - Partitioned:
        Data are processed one partition at a time, clipped to the basin,
        and written to partitioned output directories. --partitioned flag must be set.

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


def get_metadata_json(params: argparse.Namespace, start_time, logger: logging.Logger):
    """
    Generate metadata JSON for clipped data.
    Args:
        params (argparse.Namespace): Command line parameters.
        start_time (float): Start time.
        logger (logging.Logger): Logger object.
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
        args (list[str]): Command line arguments.
    """

    start_time = time.time()
    params = parse_arguments(args)
    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
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

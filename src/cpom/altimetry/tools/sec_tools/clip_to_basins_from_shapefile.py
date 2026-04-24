"""
cpom.altimetry.tools.sec_tools.clip_to_basins_from_shapefile

Purpose:
    Clip gridded altimetry data to basin, glacier, or region boundaries defined
    by polygon geometries in a shapefile, using grid cell centres.

    - Non-partitioned: full dataset loaded into memory and clipped in one pass.
    - Partitioned: large datasets processed partition-by-partition using
                       bounding-box pre-filtering to limit memory usage.

    Recommended for use after epoch_average.

Shapefile configuration options:
    - Pass a CPOM Mask class name (--mask); its shapefile and selector column are used.
    - Supply a shapefile path (--shapefile) and column name (--shp_file_column) directly.

Output:
    - Clipped Parquet data written per basin:
        - Non-partitioned: <out_dir>/<basin_name>/data.parquet
        - Partitioned:     <out_dir>/<basin_name>/x_part=<x>/y_part=<y>/data.parquet
    - metadata.json summarising processing parameters and execution time.
"""

import argparse
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
    """Parse command-line arguments for clipping altimetry data to basin polygons."""
    parser = argparse.ArgumentParser(
        description="Clip altimetry data to basin polygons from shapefile"
    )
    parser.add_argument(
        "--in_step",
        type=str,
        help="Input algorithm step to source metadata from",
    )
    parser.add_argument("--in_dir", type=str, required=True, help="Input data directory")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory Path",
    )
    parser.add_argument(
        "--parquet_glob",
        type=str,
        default="**/*.parquet",
        help="Glob pattern for selecting input files relative to --in_dir.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="CPOM Mask class name. Its shapefile path and selector column are used for clipping.",
    )
    parser.add_argument(
        "--shapefile",
        type=str,
        help="Path to the shapefile for clipping (alternative to --mask).",
    )
    parser.add_argument(
        "--shp_file_column",
        type=str,
        help="Shapefile column containing basin/region identifiers.",
    )
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help="Enable partition-by-partition processing for large datasets. "
        "Each partition is bounding-box filtered before clipping.",
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
        help="Grid area name. Grid metadata fallback",
    )
    parser.add_argument(
        "--binsize",
        type=float,
        required=False,
        help="Grid bin size. Grid metadata fallback",
    )

    # Standardize basin selection arguments across tools
    add_basin_selection_arguments(parser)

    return parser.parse_args(args)


def add_coordinates_to_data(
    data: pl.LazyFrame,
    grid_area: GridArea,
) -> pl.LazyFrame:
    """
    Add projected x/y coordinates for grid-cell centres to the dataset.

    Coordinates are derived from x_bin and y_bin indices using the GridArea
    definition and added as columns 'x' and 'y'.

    Args:
        data (pl.LazyFrame): LazyFrame with x_bin and y_bin columns.
        grid_area (GridArea): CPOM GridArea object for coordinate conversion.

    Returns:
        pl.LazyFrame: Input data with 'x' and 'y' columns appended.
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
    params: argparse.Namespace,
    grid_area: GridArea,
    basin_shape: gpd.GeoDataFrame,
    logger: logging.Logger,
):
    """
    Yeild partitioned parquet data filtered to a basin bounding box.

    Each partition is read independently, rows outside the bounding box are disguarded.

    Args:
        params (argparse.Namespace): Command line parameters (uses in_dir, parquet_glob).
        grid_area (GridArea): CPOM GridArea object for coordinate conversion.
        basin_shape (geopandas.GeoDataFrame): Basin geometry to determine bounding-box limits.
        logger (logging.Logger): Logger Object.

    Yields:
        pl.DataFrame: One partition per file, with x/y coordinates present
                      and rows filtered to the basin bounding box.
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
    Keep grid cells, where the cell centre falls within basin polygon.

    Finds unique (x_bin, y_bin) pairs in the data, performs a spatial join
    against the basin polygon, then filters the full dataset to matching bins.

    Args:
        grid_area (GridArea): CPOM GridArea object for coordinate conversion.
        subregion_shape ( geopandas.GeoDataFrame ): Basin or region polygon geometry.
        data (pl.LazyFrame | pl.DataFrame): Input data containing (x_bin, y_bin, x , y).

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

    Non-partitioned: Loads the full dataset, clips it, and writes a single file.
    Partitioned: Iterates over partitions, clips each one, and writes per-partition
    output directories (requires --partitioned flag).

    Args:
        params (argparse.Namespace): Command line parameters.
        grid_area (GridArea): CPOM GridArea Object.
        basin_shape (gpd.GeoDataFrame): GeoDataFrame containing the basin polygon geometry.
        basin_name (str): Basin identifier
        logger (logging.Logger): Logger object.

    Returns:
        dict with keys:
            - output_file (str): Path to the output file or directory.
            - n_rows (int): Total rows written.
            - n_unique_cells (int): Number of unique grid cells written.

    Output:
        - Non-partitioned: <out_dir>/<basin_name>/data.parquet
        - Partitioned: <out_dir>/<basin_name>/x_part=<x>/y_part=<y>/data.parquet
    """
    logger.info(f"Clipping to: {basin_name}")
    output_dir = Path(params.out_dir) / basin_name
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    unique_cells: set[tuple[int, int]] = set()

    if not params.partitioned:
        in_file = Path(params.in_dir) / params.parquet_glob
        input_data = add_coordinates_to_data(pl.scan_parquet(in_file), grid_area)
        logger.info(f"Loading data from: {in_file}")
        # Clip to basin shape by bin centres
        clipped_data = clip_data_to_shape(grid_area, basin_shape, input_data)
        clipped_df = clipped_data.collect()
        total_rows = clipped_df.height
        unique_cells = {
            (int(x_bin), int(y_bin))
            for x_bin, y_bin in clipped_df.select(["x_bin", "y_bin"]).unique().iter_rows()
        }

        output_file = output_dir / "data.parquet"
        clipped_df.write_parquet(output_file)
        logger.info(f"Wrote: {output_file}")
    else:
        for partition in load_partitioned_data_for_basin(params, grid_area, basin_shape, logger):
            # Clip this partition to the basin shape by bin centres
            clipped_chunk = clip_data_to_shape(grid_area, basin_shape, partition)
            clipped_df = clipped_chunk.collect()

            # Count points after clipping
            clipped_height = clipped_df.height
            if clipped_height == 0:
                logger.info("Partition empty after clipping, skipping")
                continue

            logger.info(f"Clipped partition: {clipped_height} rows")
            total_rows += clipped_height
            unique_cells.update(
                (int(x_bin), int(y_bin))
                for x_bin, y_bin in clipped_df.select(["x_bin", "y_bin"]).unique().iter_rows()
            )

            # Save clipped partition
            chunked_out = (
                Path(output_dir)
                / f"x_part={partition['x_part'][0]}"
                / f"y_part={partition['y_part'][0]}"
            )
            chunked_out.mkdir(parents=True, exist_ok=True)
            clipped_df.write_parquet(chunked_out / "data.parquet")
        output_file = output_dir

    return {
        "output_file": str(output_file),
        "n_rows": int(total_rows),
        "n_unique_cells": int(len(unique_cells)),
    }


def get_metadata_json(
    params: argparse.Namespace,
    start_time,
    logger: logging.Logger,
    basin_name: str,
    basin_output_dir: Path,
    basin_stats: dict[str, int | str],
):
    """
    Generate metadata JSON for processed basin.

    Args:
        params (argparse.Namespace): Command line parameters.
        start_time (float): Start time.
        logger (logging.Logger): Logger object.
        basin_name (str): Basin identifier.
        basin_output_dir (Path): Output basin directory
        basin_stats (dict): Output statistics from process_single_basin()
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
        )
        logger.info("Wrote data_set metadata to folder %s", basin_output_dir)

    except OSError as e:
        logger.error("Failed to write surface_fit_meta.json with %s", e)


# ----------------#
# Main Function  #
# ----------------#
def clip_to_basins_from_shapefile(args):
    """
    Main entry point for clipping altimetry data to shapefile basins.
    For each selected region, clip the input dataset to polygon and write results to disk.

    Steps:
        1. Parse command line arguments and initialise logging.
        2. Resolve grid parameters and build GridArea object.
        3. Load shapefile (from CPOM mask or explicit path/column)
        4. For each region: Clip data and write output and metadata.

    Args:
        args (list[str]): Arguments.
    """

    start_time = time.time()
    params = parse_arguments(args)
    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )
    try:
        grid_params = get_metadata_params(params, fields=["gridarea", "binsize"])
        grid_area = GridArea(grid_params["gridarea"], grid_params["binsize"])
    except ValueError as exc:
        logger.error("Couldn't resolve required grid parameters: %s", exc)
        sys.exit(str(exc))

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
        output_dir = Path(params.out_dir) / region
        output_dir.mkdir(parents=True, exist_ok=True)
        region_shape = shp[shp[selector] == region]

        if not region_shape.empty:
            basin_stats = process_single_basin(params, grid_area, region_shape, region, logger)
            get_metadata_json(params, start_time, logger, region, output_dir, basin_stats)
        else:
            logger.warning("Region %s not found in shapefile selector column %s", region, selector)


if __name__ == "__main__":
    clip_to_basins_from_shapefile(sys.argv[1:])

"""
cpom.altimetry.tools.sec_tools.clip_to_basins

Clip gridded SEC outputs to selected regions and write one Parquet per region.
To be run after epoch_average step.

This module supports two clipping routes:
1. Grid-mask clipping via CPOM Mask classes (--mask)
2. Polygon clipping via shapefile boundaries (--shapefile or mask-provided shapefile)

When --mask is used and the mask provides a shapefile, --clip_method selects which route
to take ('grid_mask', 'precise', or 'auto' to prefer the shapefile when available).

For each selected region, the tool:
- Loads input parquet data
- Resolves grid-cell centre x/y coordinates from x_bin/y_bin
- Applies region clipping (mask-based or polygon-within test)
- Writes clipped parquet and metadata to a per-region output folder

Outputs:
- <out_dir>/<region_label>/data.parquet
- <out_dir>/<region_label>/clip_to_basins_meta.json
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import cast

import geopandas as gpd
import polars as pl

from cpom.altimetry.datasets.parquet_tools.spatio_temporal import ParquetFilter
from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
)
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    elapsed,
    get_metadata_params,
    write_metadata,
)
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers
from cpom.masks.masks import Mask


def parse_arguments(args: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for mask-based clipping of altimetry data."""

    parser = argparse.ArgumentParser(
        description="Clip altimetry data to regions defined by CPOM grid mask"
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
        help="CPOM Mask class name used for clipping.",
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
        "--clip_method",
        type=str,
        choices=["auto", "grid_mask", "precise"],
        default="auto",
        help=(
            "Clipping route to use with --mask: 'grid_mask' clips using the mask's grid "
            "values (fast), 'precise' clips using the mask's shapefile boundary (exact but "
            "slower, errors if the mask provides no shapefile). 'auto' (default) uses the "
            "shapefile when the mask provides one, else falls back to grid_mask. Ignored "
            "when --shapefile is given directly (always precise)."
        ),
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
    parser.add_argument(
        "--keep_output_as_numbers",
        action="store_true",
        help=(
            "Keep output directory and metadata basin labels as numeric mask values "
            "instead of mask.grid_value_names labels."
        ),
    )
    # Standardize basin selection arguments across tools
    add_basin_selection_arguments(parser)
    return parser.parse_args(args)


def get_basin_values_and_numbers(
    params: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[list[tuple[str, int]], Mask | None, gpd.GeoDataFrame | None]:
    """
    Get basin names and corresponding mask values for clipping.

    Returns:
        tuple[list[tuple[str, int]], Mask | None, gpd.GeoDataFrame | None]:
            - (basin_name, basin_number) pairs for each region to process
            - Mask object when clipping from a CPOM mask, else None
            - GeoDataFrame when clipping with polygon boundaries, else None
    """

    mask, shp = None, None

    if params.mask:
        mask = Mask(
            params.mask,
            basin_numbers=params.region_selector if params.region_selector != ["all"] else None,
        )

        numbers = mask.resolve_basin_numbers()
        if numbers:
            names = [str(name) for name in mask.get_grid_value_names_from_grid_value(numbers)]
        else:
            names = [str(name) for name in mask.grid_value_names]
            numbers = [int(number) for number in mask.mask_grid_possible_values]

        mask_has_shapefile = hasattr(mask, "shapefile_path")
        if params.clip_method == "precise" and not mask_has_shapefile:
            raise ValueError(
                f"--clip_method=precise requested but mask '{params.mask}' provides no shapefile"
            )

        if mask_has_shapefile and params.clip_method != "grid_mask":
            logger.info(
                "Clipping using mask: %s with shapefile: %s", params.mask, mask.shapefile_path
            )
            shp = gpd.read_file(mask.shapefile_path)
            params.shp_file_column = mask.shapefile_column_name
        else:
            logger.info("Clipping using mask: %s with grid_mask", params.mask)

    elif params.shapefile:
        logger.info("Clipping using user provided shapefile: %s", params.shapefile)
        shp = gpd.read_file(params.shapefile)
        if not params.shp_file_column:
            raise ValueError("--shp_file_column must be provided when using --shapefile")
        unique_regions = shp[params.shp_file_column].unique()
        unique_regions_list = [str(name) for name in unique_regions]
        names = unique_regions_list if params.region_selector == ["all"] else params.region_selector
        numbers = [unique_regions_list.index(str(name)) for name in names]
    else:
        logger.error(
            "No valid clipping method provided. Please specify either --mask or --shapefile."
        )
        sys.exit("No valid clipping method provided. Please specify either --mask or --shapefile.")

    return list(zip([str(name) for name in names], [int(number) for number in numbers])), mask, shp


def get_data(
    infile: str | Path,
    grid_area: GridArea,
    logger: logging.Logger,
) -> tuple[pl.LazyFrame, pl.DataFrame]:
    """
    Load data and add projected x/y coordinates for each grid-cell centre.

    Unique (x_bin, y_bin) pairs are converted to lat/lon using the GridArea
    definition and appended as new columns.

    Args:
        grid_area (GridArea): CPOM GridArea object for coordinate conversion.
        infile (str): Path to the input Parquet file.
        logger (logging.Logger): Logger Object.

    Returns:
        tuple[pl.LazyFrame, pl.DataFrame]:
            - Input data with 'x' and 'y' columns joined to each row
            - DataFrame of unique bin centres (x_bin, y_bin, x, y)
    """

    logger.info("Loading data from: %s", Path(infile))
    epoch_data = pl.scan_parquet(Path(infile))

    # Get unique cells
    unique_cells = epoch_data.select(["x_bin", "y_bin"]).unique().collect()

    x, y = grid_area.get_cellcentre_x_y_from_col_row(
        unique_cells.get_column("x_bin").to_numpy(),
        unique_cells.get_column("y_bin").to_numpy(),
    )

    unique_cells = unique_cells.with_columns(
        [
            pl.Series("x", x),
            pl.Series("y", y),
        ]
    )

    # Join coordinates to data
    epoch_data = epoch_data.join(unique_cells.lazy(), on=["x_bin", "y_bin"], how="left")
    return epoch_data, unique_cells


def process_single_basin(
    params: argparse.Namespace,
    data: pl.LazyFrame,
    mask: Mask | None,
    shapefile: gpd.GeoDataFrame | None,
    unique_cells: pl.DataFrame | None,
    data_crs: int | None,
    basin_name: str,
    basin_number: int,
) -> pl.LazyFrame:
    """
    Clip input data to a single basin.

    Args:
        params (argparse.Namespace): Runtime parameters.
        data (pl.LazyFrame): Input data with 'x' and 'y' columns for masking.
        mask (Mask | None): CPOM Mask object for grid-based clipping.
        shapefile (gpd.GeoDataFrame | None): Region polygons for shp clipping.
        unique_cells (pl.DataFrame | None): Unique grid-cell centres (x_bin, y_bin, x, y)
            for shp clipping.
        data_crs (int | None): EPSG code of the 'x'/'y' columns, for shp clipping.
        basin_name (str): Basin/region label.
        basin_number (int): Numeric basin identifier.

    Returns:
        pl.LazyFrame: Clipped basin data.
    """
    if shapefile is not None:
        if unique_cells is None:
            raise ValueError("unique_cells is required when clipping by shapefile")

        # Find bins with centres inside basin.
        bins_in_basin = (
            ParquetFilter(
                unique_cells, lon_col="x", lat_col="y", data_crs=data_crs, engine="duckdb"
            )
            .get_polygon(
                shapefile[shapefile[params.shp_file_column] == basin_name],
                precise=True,
            )
            .select(["x_bin", "y_bin"])
            .run()
        )
        bins_in_basin = cast(pl.DataFrame, bins_in_basin)

        basin_data = data.join(bins_in_basin.lazy(), on=["x_bin", "y_bin"], how="inner")

    else:
        if mask is None:
            raise ValueError("mask is required when clipping by grid mask")
        basin_data = (
            ParquetFilter(data, engine="polars")
            .get_cpom_grid_mask(mask, basin_numbers=[basin_number])
            .run()
        )

    return basin_data


def write_clipped_data(
    output_data: pl.LazyFrame, output_dir: Path, logger: logging.Logger
) -> dict[str, int | str]:
    """Write clipped data and return simple row/cell summary statistics."""

    stats_df = output_data.select(
        [
            pl.len().alias("n_rows"),
            pl.struct(["x_bin", "y_bin"]).n_unique().alias("n_unique_cells"),
        ]
    ).collect()

    output_file = output_dir / "data.parquet"
    output_data.sink_parquet(output_file)
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
    basin_number: int,
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
            "clip_to_basins",
            basin_output_dir,
            {
                **dict(vars(params)),
                "basin_name": basin_name,
                "basin_number": basin_number,
                **basin_stats,
                "execution_time": elapsed(start_time),
            },
        )
        logger.info("Wrote data_set metadata to folder %s", basin_output_dir)

    except OSError as e:
        logger.error("Failed to write surface_fit_meta.json with %s", e)


# ----------------#
# Main Function #
# ----------------#
def clip_to_basins(args: list[str]) -> None:
    """
    Main entry point for clipping altimetry data to shapefile basins.
    from CPOM Mask class grid.

    Steps:
        1. Parse command line arguments and initialise logging.
        2. Resolve grid parameters and build GridArea object.
        3. Process and clip data by mask regions
        4. For each region: Clip data and write output and metadata.

    Args:
        args (list[str]): Arguments.
    """
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

    selector_name_num, this_mask, shapefile = get_basin_values_and_numbers(params, logger)

    grid_area = GridArea(str(grid_params["gridarea"]), int(grid_params["binsize"]))
    data_crs = grid_area.crs_bng.to_epsg()

    input_data, unique_cells = get_data(
        Path(params.in_dir) / params.parquet_glob,
        grid_area,
        logger,
    )

    for basin_name, basin_number in selector_name_num:
        start_time = time.time()

        logger.info("Processing region: %s (mask value %s)", basin_name, basin_number)
        output_dir = (
            Path(params.out_dir) / basin_name
            if not params.keep_output_as_numbers
            else Path(params.out_dir) / str(basin_number)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        stats_dict = write_clipped_data(
            process_single_basin(
                params,
                input_data,
                this_mask,
                shapefile,
                unique_cells,
                data_crs,
                basin_name,
                basin_number,
            ),
            output_dir,
            logger,
        )

        get_metadata_json(
            params, start_time, logger, basin_name, basin_number, output_dir, stats_dict
        )


if __name__ == "__main__":
    clip_to_basins(sys.argv[1:])

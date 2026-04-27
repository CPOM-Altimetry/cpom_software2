"""
cpom.altimetry.tools.sec_tools.clip_to_basins

Purpose:

    Clips data to basin, glacier, or region defined by a CPOM Mask class grid mask,
    by clipping points to mask label based on a 2km grid.

    - Non-partitioned: full dataset loaded into memory and clipped in one pass.
    - Partitioned: Use clip_to_basins_from_shapefile.py.

Supported masks:
    Any CPOM grid mask from cpom.masks.mask_list with grid_value_names defined.

Output:
    - Clipped Parquet data written per basin:
        - Non-partitioned: <out_dir>/<basin_name>/data.parquet
    metadata.json summarising processing parameters and execution time.
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
from cpom.altimetry.tools.sec_tools.clip_to_basins_from_shapefile import (
    get_metadata_json,
)
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    get_metadata_params,
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
        required=True,
        help="CPOM Mask class name used for clipping.",
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
        "--mask_values",
        nargs="+",
        default=None,
        help=(
            "Mask values to clip to. Accepts integer pixel values (e.g. 1 2 3) or "
            "string names from mask.grid_value_names (e.g. basin_A basin_B). "
            "Overrides --region_selector when supplied."
        ),
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


def resolve_basin_targets(
    mask: Mask, values: list[str] | None, keep_output_as_numbers: bool = False
) -> list[tuple[str, int]]:
    """
    Convert basin selectors into validated output labels and mask pixel values.

    Args:
        mask (Mask): CPOM Mask object with grid_value_names populated.
        values (list[str] | None): Raw CLI values. Each item is either a decimal integer
            string (e.g. '3') or a name that already appears in mask.grid_value_names.
            If None, all mask values are returned.
        keep_output_as_numbers (bool): If True, use the numeric mask value as the output label.

    Returns:
        list[tuple[str, int]]: Pairs of (output_label, basin_number).

    Raises:
        ValueError: If an integer index is out of range or a name is not found.
    """
    if values is None:
        return [
            (
                str(index) if keep_output_as_numbers else name,
                index,
            )
            for index, name in enumerate(mask.grid_value_names)
        ]

    resolved: list[tuple[str, int]] = []
    for val in values:
        # Try integer index first
        try:
            idx = int(val)
        except ValueError:
            idx = None

        if idx is not None:
            if idx < 0 or idx >= len(mask.grid_value_names):
                raise ValueError(
                    f"Mask value {idx} is out of range (mask has "
                    f"{len(mask.grid_value_names)} values)."
                )
            resolved.append(
                (str(idx) if keep_output_as_numbers else mask.grid_value_names[idx], idx)
            )
        elif val in mask.grid_value_names:
            idx = mask.grid_value_names.index(val)
            resolved.append((str(idx) if keep_output_as_numbers else val, idx))
        else:
            raise ValueError(
                f"'{val}' is not a valid mask pixel index or name. "
                f"Valid names: {mask.grid_value_names}"
            )
    return resolved


def get_data(
    grid_area: GridArea,
    logger: logging.Logger,
    infile: str | None = None,
    lazyframe: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """
    Load data and add latitude/longitude coordinates for each grid-cell centre.

    Unique (x_bin, y_bin) pairs are converted to lat/lon using the GridArea
    definition and appended as new columns.

    Args:
        grid_area (GridArea): CPOM GridArea object for coordinate conversion.
        infile (str): Path to the input Parquet file.
        logger (logging.Logger): Logger Object.

    Returns:
        pl.LazyFrame: Input data with 'lat' and 'lon' columns appended.
    """
    if lazyframe is not None and not lazyframe.is_empty():
        epoch_data = lazyframe
    else:
        if infile is None:
            raise ValueError("Either 'infile' or 'lazyframe' must be provided")
        logger.info(f"Loading data from: {Path(infile)}")
        epoch_data = pl.scan_parquet(Path(infile))

    # Get unique cells
    unique_cells = epoch_data.select(["x_bin", "y_bin"]).unique().collect()

    x, y = grid_area.get_cellcentre_x_y_from_col_row(
        unique_cells.get_column("x_bin").to_numpy(),
        unique_cells.get_column("y_bin").to_numpy(),
    )
    coords_df = unique_cells.with_columns(
        [
            pl.Series("x", x),
            pl.Series("y", y),
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
    basin_number: int,
    data: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Clip data rows where the grid-cell centre falls within the basin mask region.

    Uses Mask.points_inside_polars with x, y columns. x and y are dropped from
    the returned frame.

    Args:
        mask (Mask): CPOM Mask object. Must have x and y columns present in data.
        basin_number (int): Integer pixel value in mask.grid_value_names to keep.
        data (pl.LazyFrame): Input data containing 'x' and 'y' columns.

    Returns:
        pl.LazyFrame: Rows inside the basin with x and y columns dropped.
    """
    # points_inside_polars filters by self.basin_numbers; set it for this call then restore.
    prev_basin_numbers = mask.basin_numbers
    mask.basin_numbers = [basin_number]
    try:
        result = mask.points_inside_polars(data).drop(["x", "y"])
    finally:
        mask.basin_numbers = prev_basin_numbers
    return result


def process_single_basin(
    params: argparse.Namespace,
    mask: Mask,
    grid_area: GridArea,
    basin_name: str,
    basin_number: int,
    output_label: str,
    logger: logging.Logger,
) -> dict[str, int | str]:
    """
    Clip altimetry data to a single basin and write results to disk.

    Loads the full dataset, assigns mask labels, keep rows that match basin_name,
    and writes a single file.

    Args:
        params (argparse.Namespace): Command line parameters.
        mask (Mask): CPOM Mask object for mask value lookup.
        grid_area (GridArea): CPOM GridArea Object.
        basin_name (str): Basin identifier
        logger (logging.Logger): Logger object.

    Returns:
        dict with keys:
            - output_file (str): Path to the output file or directory.
            - n_rows (int): Total rows written.
            - n_unique_cells (int): Number of unique grid cells written.

    Output:
        - Non-partitioned: <out_dir>/<basin_name>/data.parquet
    """
    logger.info("Clipping to: %s (mask value %s)", basin_name, basin_number)
    output_dir = Path(params.out_dir) / output_label
    output_dir.mkdir(parents=True, exist_ok=True)

    in_file = Path(params.in_dir) / params.parquet_glob
    data = get_data(grid_area, logger, in_file)
    clipped_data = clip_data_to_shape(mask, basin_number, data)
    stats_df = clipped_data.select(
        [
            pl.len().alias("n_rows"),
            pl.struct(["x_bin", "y_bin"]).n_unique().alias("n_unique_cells"),
        ]
    ).collect()

    output_file = output_dir / "data.parquet"
    clipped_data.sink_parquet(output_file)
    logger.info(f"Wrote: {output_file}")

    return {
        "output_file": str(output_file),
        "n_rows": int(stats_df["n_rows"][0]),
        "n_unique_cells": int(stats_df["n_unique_cells"][0]),
    }


# ----------------#
# Main Function #
# ----------------#
def clip_to_basins(args):
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

    # Get basins to process.
    if params.mask_values:
        raw_values = params.mask_values
    elif params.region_selector == ["all"]:
        raw_values = None  # processed below
    else:
        raw_values = params.region_selector

    if raw_values is not None:
        try:
            regions_to_process = resolve_basin_targets(
                mask,
                raw_values,
                keep_output_as_numbers=params.keep_output_as_numbers,
            )
        except ValueError as exc:
            logger.error("Invalid basin selection: %s", exc)
            sys.exit(str(exc))
    else:
        regions_to_process = resolve_basin_targets(
            mask,
            None,
            keep_output_as_numbers=params.keep_output_as_numbers,
        )

    # Clip to basin shape by bin centres
    for output_label, basin_number in regions_to_process:
        basin_name = mask.grid_value_names[basin_number]
        logger.info("Processing region: %s (mask value %s)", basin_name, basin_number)
        output_dir = Path(params.out_dir) / output_label
        output_dir.mkdir(parents=True, exist_ok=True)

        basin_stats = process_single_basin(
            params,
            mask,
            GridArea(str(grid_params["gridarea"]), float(grid_params["binsize"])),
            basin_name,
            basin_number,
            output_label,
            logger,
        )
        get_metadata_json(params, start_time, logger, output_label, output_dir, basin_stats)


if __name__ == "__main__":
    clip_to_basins(sys.argv[1:])

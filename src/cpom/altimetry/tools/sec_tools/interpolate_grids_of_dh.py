"""
cpom.altimetry.tools.sec_tools.interpolate_grids_of_dh

Purpose:
    Interpolate missing values in gridded elevation change data.

    Uses Delaunay triangulation (matplotlib LinearTriInterpolator) to fill gaps
    in gridded data produced by epoch_average or calculate_dhdt. Original input
    values are preserved; only missing cells are interpolated. A variable flag column distinguishes
    between original and interpolated values (0 = no data, 1 = original input, 2 = interpolated).

Output:
    <out_dir>/interpolated.parquet
    Parquet file containing x_bin, y_bin, interpolated variable values, per-variable
    flags, and any static metadata columns present in the input.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from matplotlib.tri import LinearTriInterpolator, Triangulation

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
    get_basins_to_process,
)
from cpom.altimetry.tools.sec_tools.clip_to_basins import get_data
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    elapsed,
    get_algo_name,
    get_metadata_params,
    write_metadata,
)
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers
from cpom.masks.masks import Mask


def _parse_optional_float(value: Any) -> float | None:
    """Parse a numeric filter value, allowing null-like values to mean None."""
    if value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "none", "null", "None"}
    ):
        return None
    return float(value)


def parse_arguments(args) -> argparse.Namespace:
    """
    Parse command line arguments for grid interpolation.
    """
    parser = argparse.ArgumentParser(
        description="Interpolate missing values in gridded elevation"
        " change data using Delaunay triangulation."
    )
    # I/O Arguments
    parser.add_argument(
        "--in_step",
        type=str,
        help="Input algorithm step to source metadata from",
    )
    parser.add_argument("--in_dir", required=True, help="Input data directory (epoch_average)")
    parser.add_argument("--out_dir", required=True, help="Output directory Path")

    parser.add_argument(
        "--parquet_glob",
        type=str,
        default="*.parquet",
        help="File glob pattern for input Parquet files, relative to --in_dir.",
    )
    # Variable arguments
    parser.add_argument(
        "--timestamp_column",
        type=str,
        default="epoch_number",
        help="Column name for grouping timestamps (e.g. epoch_number or period_number)."
        " Each group will be interpolated separately.",
    )
    parser.add_argument(
        "--variables_in",
        required=True,
        nargs="+",
        help="Column names to interpolate. Comma-separated list",
    )
    parser.add_argument(
        "--lo_filter",
        nargs="+",
        help="Lower bound per variable for filtering before and after interpolation."
        " Space -seperated list of length -- variables in.",
    )
    parser.add_argument(
        "--hi_filter",
        nargs="+",
        help="Upper bound per variable for filtering before and after interpolation."
        " Space -seperated list of length -- variables in.",
    )
    parser.add_argument(
        "--tri_lim", default=20.0, help="Maximum triangle edge length for Delaunay triangulation"
    )

    parser.add_argument("--mask", help="CPOM Mask name to apply to output data")
    parser.add_argument(
        "--mask_values",
        nargs="+",
        help="Mask pixel values to accept as valid, e.g. --mask_values 1, 2, 3.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging")
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
    add_basin_selection_arguments(parser)

    return parser.parse_args(args)


def get_grid_and_flags(
    group: pl.DataFrame, nrows: int, ncols: int
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """
    Get a 2D grid and flag array for a single group. Mark cells with 1 where input data is present.

    Args:
        group (pl.DataFrame): Rows for a single group. (Epoch or period)
        nrows (int): Grid row count from GridArea metadata.
        ncols (int): Grid column count from GridArea metadata.

    Returns:
        grid (np.ndarray): 2D float array of shape (nrows, ncols) filled with NaN.
        grid_flags (np.ndarray): 2D int array of shape (nrows, ncols). 0 = No data, 1 = input data.
        populated_idx (tuple[np.ndarray, np.ndarray]): Row and column idx of populated cells.
        x_bins (np.ndarray): x_bin values for populated cells
        y_bins (np.ndarray): y_bin values for populated cells
    """

    # Get array of populated grid cell indices
    x_bins = group["x_bin"].to_numpy()
    y_bins = group["y_bin"].to_numpy()

    # Reconstruct the grid
    grid = np.full((nrows, ncols), np.nan)

    # Populate gridcells with data with flag 1
    grid_flags = np.zeros_like(
        grid, dtype=np.byte
    )  # Flag array, 0 = no data, 1 = input, 2 = interpolated (added later)

    grid_flags[y_bins, x_bins] = 1

    # Find points containing values
    row_idx, col_idx = np.where(grid_flags != 0)
    populated_idx = (row_idx, col_idx)

    return grid, grid_flags, populated_idx, x_bins, y_bins


def triangulate_points(
    params: argparse.Namespace,
    group_name: Any,
    populated_idx: tuple[np.ndarray, np.ndarray],
    logger: logging.Logger,
) -> Triangulation | None:
    """
    Build Delaunay triangulation object from populated grid cell indicies, masking long edges.

    Triangles whose longest edge exceeds params.tri_lim are masked to prevent
    interpolation across large data gaps.

    Args:
        params (argparse.Namespace): Command line arguments (uses tri_lim).
        group_name (str): Name of the params.timestamp_column value being processed.
        populated_idx (tuple[np.ndarray, np.ndarray]): Row and column indices of populated cells.
        logger (logging.Logger): Logger object.

    Returns:
        Triangulation | None: Triangulation with long-edge mask applied, or None if there are too
        few points or triangulation fails.
    """
    # --------------------------------------------------------#
    # If no or too few points (3 is minimum, in the points
    # array that's 3x2 elements, so size 6)
    # --------------------------------------------------------#
    group_label = str(group_name[0]) if isinstance(group_name, tuple) else str(group_name)

    if len(populated_idx[0]) < 6:
        logger.debug("Not enough points for triangulation for group %s", group_label)
        return None
    try:
        tri_obj = Triangulation(populated_idx[0], populated_idx[1])
    # pylint: disable=W0703
    except Exception as e:
        logger.debug("Triangulation error for group %s: %s", group_label, e)
        return None

    # Mask out triangles whose longest edge exceeds tri_lim
    xtri = tri_obj.x[tri_obj.triangles] - np.roll(tri_obj.x[tri_obj.triangles], 1, axis=1)
    ytri = tri_obj.y[tri_obj.triangles] - np.roll(tri_obj.y[tri_obj.triangles], 1, axis=1)
    tri_obj.set_mask(np.max(np.sqrt(xtri**2 + ytri**2), axis=1) > params.tri_lim)
    return tri_obj


# pylint: disable=R0914
def get_trilinear_interpolation(
    params: argparse.Namespace,
    group_name: Any,
    group: pl.DataFrame,
    nrows: int,
    ncols: int,
    logger: logging.Logger,
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """
    Fill missing grid cells for all variables in one group, using LinearTriInterpolator.

    Fits a Delaunay triangulation to the populated cells and estimates values at every
    empty cell within the triangulated region.

    Flag values :
        0 : No data (empty in input, not reached by interpolation).
        1 : Original input data
        2 : Interpolated by LinearTriInterpolator.

    Args:
        params (argparse.Namespace): Command line arguments (uses tri_lim , variables_in).
        group_name (str): Name of the params.timestamp_column value being processed.
        group (pl.DataFrame): Rows for a single group. (Epoch or period)
        nrows (int): Grid row count from GridArea metadata.
        ncols (int): Grid column count from GridArea metadata.

    Returns:
        tuple:
            - dict[str, np.ndarray]: Dictionary of interpolated variable arrays and flag arrays
                if successful, None otherwise.
            - str | None: Failure reason string if interpolation fails, None otherwise.
    """

    group_label = str(group_name[0]) if isinstance(group_name, tuple) else str(group_name)

    grids = {}
    grid, grid_flags, populated_idx_raw, x_bins, y_bins = get_grid_and_flags(group, nrows, ncols)
    populated_idx = cast(tuple[np.ndarray, np.ndarray], populated_idx_raw)
    if len(populated_idx[0]) < 6:
        return None, "not_enough_points"

    tri_obj = triangulate_points(params, group_label, populated_idx, logger)
    if tri_obj is None:
        return None, "triangulation_error"

    # Construct full grid coordinates
    grid_y, grid_x = np.meshgrid(range(grid.shape[0]), range(grid.shape[1]), indexing="ij")

    # -------------------------------------------------------------------------------#
    # Make predictions for all points on grid, again catch failure of interpolation
    # The interpolator returns a masked array, with NaNs where the mask is true.
    # The input array was not masked, so only return the data
    # -------------------------------------------------------------------------------#

    for var in params.variables_in:
        grid[y_bins, x_bins] = group[var].to_numpy()
        values = grid[populated_idx[0], populated_idx[1]]
        try:
            f_pred = LinearTriInterpolator(tri_obj, values)
            pred = f_pred(grid_y, grid_x)
            pred_data = pred.data
        # pylint: disable=W0703
        except Exception as e:
            logger.warning(
                "LinearTriInterpolator error for group %s var %s: %s", group_label, var, e
            )
            return None, "linear_interpolator_error"

        # ---------------------------------------------------------------------------------#
        # The interpolator returns values at the input data points that vary very slightly
        #  from the original input. Doesn't return any input datapoints that lay only on
        # triangles that were removed. However, those datapoints are still good.
        # To fix both issues, copy existing input data to the output data before returning.
        # ----------------------------------------------------------------------------------#

        pred_data[y_bins, x_bins] = group[var].to_numpy()
        grids[var] = pred_data

        # -----------------------------------------#
        # Add to flag grid: 2 = interpolated data
        # -----------------------------------------#
        r = np.where((np.isfinite(pred)) & (grid_flags == 0))
        if len(r[0]) > 0:
            grid_flags[r] = 2
        grids[f"{var}_flags"] = grid_flags

    return grids, None


def apply_range_filters(params: argparse.Namespace, group_df: pl.DataFrame) -> pl.DataFrame:
    """
    Null out per-variable measurements outside their configured lo/hi bounds.

    Args:
        params (argparse.Namespace): Command line arguments.
            (uses lo_filter, hi_filter, variables_in).
        group_df (pl.DataFrame): Rows for a single group. (Epoch or period)

    Returns:
        pl.DataFrame: DataFrame with out-of-range values set to None.
    """
    # Set measurements outside lo/hi filters to None
    if params.lo_filter or params.hi_filter:
        for idx, var in enumerate(params.variables_in):
            lo = _parse_optional_float(params.lo_filter[idx]) if params.lo_filter else None
            hi = _parse_optional_float(params.hi_filter[idx]) if params.hi_filter else None

            if lo is not None:
                group_df = group_df.with_columns(
                    pl.when(pl.col(var) < lo).then(None).otherwise(pl.col(var)).alias(var)
                )
            if hi is not None:
                group_df = group_df.with_columns(
                    pl.when(pl.col(var) > hi).then(None).otherwise(pl.col(var)).alias(var)
                )
    return group_df


def get_epoch_stats(
    group_label: str,
    rows_before: int,
    populated_input_cells: int,
    df: pl.DataFrame | None,
    grids: dict[str, np.ndarray] | None,
    variables_in: list[str],
    failure_reason: str | None,
) -> dict[str, Any]:
    """
    Calculate statistics for a single group after interpolation,
    including counts of populated input cells, cells in output and interpolated cells.

    Args:
        group_label (str): String label for epoch or period.
        rows_before (int): Row count before interpolation.
        populated_input_cells (int): Unique grid cells present in the input.
        df (pl.DataFrame | None): Output DataFrame after interpolation,None if interpolation failed.
        grids (dict[str, np.ndarray] | None): Raw grids arras from get_trilinear_interpolation,
            or None on failure.
        failure_reason (str | None): String reason for interpolation failure, or None if successful.

    Returns:
        dict[str, Any]: Statistics dictionary with keys: timestamp, rows_before,
            rows_after, populated_input_cells, cells_in_output, interpolated_cells,
            failure_reason.
    """
    if grids is None or df is None:
        return {
            "timestamp": group_label,
            "rows_before": rows_before,
            "rows_after": 0,
            "populated_input_cells": populated_input_cells,
            "cells_in_output": 0,
            "interpolated_cells": 0,
            "failure_reason": failure_reason,
        }

    flag_stack = np.stack([grids[f"{var}_flags"] for var in variables_in], axis=-1)
    return {
        "timestamp": group_label,
        "rows_before": rows_before,
        "rows_after": len(df),
        "populated_input_cells": populated_input_cells,
        "cells_in_output": len(df),
        "interpolated_cells": int(np.sum(np.all(flag_stack == 2, axis=-1))),
        "failure_reason": None,
    }


def aggregate_epoch_stats(epoch_stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-epoch statistics into a summary for metadata output.

    Args:
        epoch_stats_list (list[dict[str, Any]]): Per-epoch statistics from get_epoch_stats.
    Returns:
        dict[str, Any]: Dictionary with keys:
            aggregated (dict): Total groups, succeeded, failed, input/output row counts,
                populated input cells, cells in output, and interpolated cells.
            failure_counts (dict[str, int]): Count of each failure reason string.
    """
    failure_counts: dict[str, int] = {}
    failed = 0

    for epoch_stats in epoch_stats_list:
        reason = epoch_stats["failure_reason"]
        if reason is not None:
            failed += 1
            failure_counts[str(reason)] = failure_counts.get(str(reason), 0) + 1

    return {
        "aggregated": {
            "groups": len(epoch_stats_list),
            "succeeded": len(epoch_stats_list) - failed,
            "failed": failed,
            "input_rows": sum(int(stats["rows_before"]) for stats in epoch_stats_list),
            "output_rows": sum(int(stats["rows_after"]) for stats in epoch_stats_list),
            "populated_input_cells": sum(
                int(stats["populated_input_cells"]) for stats in epoch_stats_list
            ),
            "cells_in_output": sum(int(stats["cells_in_output"]) for stats in epoch_stats_list),
            "interpolated_cells": sum(
                int(stats["interpolated_cells"]) for stats in epoch_stats_list
            ),
        },
        "failure_counts": failure_counts,
    }


# pylint: disable=R0914
def process_timesteps(
    params: argparse.Namespace,
    input_df: pl.DataFrame,
    nrows: int,
    ncols: int,
    logger: logging.Logger,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Interpolate missing grid cells for every group in the input DataFrame.

    For each group :
        1. Extract static (single-value) metadata columns to re-attach after interpolation.
        2. Null out per-variable measurements outside lo/hi range filters.
        3. Interpolate missing cells via Delaunay triangulation.
        4. Retain cells where every variable has a valid value (flag > 0).
        5. Assemble an output DataFrame with interpolated values, flags, and static columns.

    Args:
        params (argparse.Namespace): Command line arguments
        input_df (pl.DataFrame): All input rows to process.
        nrows (int): Grid row count from GridArea metadata.
        ncols (int): Grid column count from GridArea metadata.
        logger (logging.Logger): Logger object.

    Returns:
        pl.DataFrame: Concatenated output with columns: y_bin, x_bin,
            params.timestamp_column, interpolated variables, per-variable flags,
            and static metadata columns. Empty DataFrame if all groups fail.
        dict[str, Any]: Aggregated statistics from aggregate_epoch_stats.
    """
    # Group by the specified column (e.g., epoch if input is epoch averaged grids)
    logger.info("Loaded input rows: %d", input_df.height)
    dataframes: list[pl.DataFrame] = []
    epoch_stats_list: list[dict[str, Any]] = []

    groups = input_df.sort(params.timestamp_column).group_by(
        params.timestamp_column, maintain_order=True
    )

    for group_name, group_df in groups:
        group_label = str(group_name[0]) if isinstance(group_name, tuple) else str(group_name)
        rows_before = group_df.height
        logger.debug(
            "Processing group %s:  %s with %d rows",
            params.timestamp_column,
            group_label,
            rows_before,
        )
        populated_input_cells = len(
            np.unique(
                np.column_stack((group_df["y_bin"].to_numpy(), group_df["x_bin"].to_numpy())),
                axis=0,
            )
        )
        # Get columns that are unique per group
        static_cols = {
            col: group_df[col][0]
            for col in group_df.columns
            if col not in params.variables_in
            and col not in ("x_bin", "y_bin", params.timestamp_column)
            and group_df[col].n_unique() == 1
        }

        group_df = apply_range_filters(params, group_df)

        grids, failure_reason = get_trilinear_interpolation(
            params,
            group_name,
            group_df,
            nrows,
            ncols,
            logger,
        )
        if grids is None:
            group_stats = get_epoch_stats(
                group_label,
                rows_before,
                populated_input_cells,
                None,
                None,
                params.variables_in,
                failure_reason,
            )
            epoch_stats_list.append(group_stats)
            logger.debug("Group stats: %s", group_stats)
            continue

        # Stack flag grids into a 3D stack
        # Find all grid cells indicies that are populated (Flag = 1 or 2)
        flag_grids = np.stack([grids[f"{var}_flags"] for var in params.variables_in], axis=-1)
        y_idx, x_idx = np.where(np.all(flag_grids > 0, axis=-1))

        # Build output DataFrame for valid grid cells
        df = pl.DataFrame(
            {
                "y_bin": y_idx,
                "x_bin": x_idx,
                **{var: grids[var][y_idx, x_idx] for var in params.variables_in},
                **{
                    f"{var}_flag": grids[f"{var}_flags"][y_idx, x_idx]
                    for var in params.variables_in
                },
                **{col: pl.Series(col, [val] * len(y_idx)) for col, val in static_cols.items()},
            }
        )

        dataframes.append(df)
        epoch_stats_list.append(
            get_epoch_stats(
                group_label,
                rows_before,
                populated_input_cells,
                df,
                grids,
                params.variables_in,
                failure_reason=None,
            )
        )

    status = aggregate_epoch_stats(epoch_stats_list)
    logger.info("Aggregated interpolation stats: %s", status["aggregated"])

    if not dataframes:
        empty = {
            "y_bin": pl.Series("y_bin", [], dtype=pl.Int64),
            "x_bin": pl.Series("x_bin", [], dtype=pl.Int64),
            params.timestamp_column: pl.Series(params.timestamp_column, [], dtype=pl.Int64),
            **{var: pl.Series(var, [], dtype=pl.Float64) for var in params.variables_in},
            **{
                f"{var}_flag": pl.Series(f"{var}_flag", [], dtype=pl.Int8)
                for var in params.variables_in
            },
        }
        return pl.DataFrame(empty), status

    return pl.concat(dataframes), status


def process_target(
    params: argparse.Namespace,
    start_time: float,
    logger: logging.Logger,
    sub_basin: str | None = None,
) -> None:
    """
    Interpolate and write output for a single root directory or named basin.
    Resolves grid parameters, runs process_timesteps, applies the configured mask,
        writes interpolated.parquet, and records pipeline metadata.

    Args:
    params (argparse.Namespace): Command line arguments.
    start_time (float): Pipeline start time for metadata.
    logger (logging.Logger): Logger object.
    sub_basin (str | None): Sub-basin name if processing with basin structure.
    """

    grid_params = get_metadata_params(
        params=params,
        fields=["gridarea", "binsize"],
        basin_name=sub_basin,
    )

    ga = GridArea(grid_params["gridarea"], grid_params["binsize"])

    ncols, nrows = ga.get_ncols_nrows()

    in_dir = Path(params.in_dir) / sub_basin if sub_basin else Path(params.in_dir)
    out_dir = Path(params.out_dir) / sub_basin if sub_basin else Path(params.out_dir)
    output_path = out_dir / "interpolated.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    interpolated_df, status = process_timesteps(
        params, pl.read_parquet(in_dir / params.parquet_glob), nrows, ncols, logger
    )
    # Mask out-of-range values in the output DataFrame as well, to be safe
    interpolated_lf = get_data(grid_area=ga, logger=logger, lazyframe=interpolated_df.lazy())
    masked_lf = (
        Mask(params.mask)
        .points_inside_polars(interpolated_lf, basin_numbers=params.mask_values)
        .drop(["x", "y"])
    )

    masked_lf.sink_parquet(output_path, compression="zstd")

    try:
        write_metadata(
            params,
            get_algo_name(__file__),
            out_dir,
            {
                **vars(params),
                "status": status,
                "execution_time": elapsed(start_time),
            },
            basin_name=sub_basin,
            logger=logger,
        )
    except OSError as e:
        sys.exit(str(e))


# -------------------
# Main function
# -------------------
def interpolate_grids_of_dh(args):
    """
    Main entry point for grid interpolation.

    Parses arguments, resolves basins, and calls process_target for each target.

    Args:
        args (list[str]): Command line arguments, e.g. sys.argv[1:].
    """

    start_time = time.time()
    params = parse_arguments(args)

    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )

    if params.basin_structure is False:
        logger.info("Processing root-level data")
        try:
            process_target(
                params,
                start_time,
                logger,
            )
        except ValueError as exc:
            logger.error("Couldn't resolve required grid parameters: %s", exc)
            sys.exit(str(exc))
    else:
        logger.info("Processing basin/sub-basin level data")
        # Determine which sub-basins to process
        for sub_basin in get_basins_to_process(params, params.in_dir, logger=logger):
            logger.info("Processing sub-basin: %s", sub_basin)
            try:
                process_target(
                    params,
                    start_time,
                    logger,
                    sub_basin=sub_basin,
                )
            except ValueError as exc:
                logger.error(
                    "Couldn't resolve required grid parameters for basin %s: %s",
                    sub_basin,
                    exc,
                )
                continue


if __name__ == "__main__":
    interpolate_grids_of_dh(sys.argv[1:])

"""
cpom.altimetry.tools.sec_tools.interpolate_grids_of_dh

Purpose:
    Interpolate missing values in gridded elevation change data.

    Uses Delaunay triangulation (matplotlib LinearTriInterpolator) to fill gaps
    in gridded data from epoch_average or calculate_dhdt outputs. Original input
    values are preserved; only missing cells are interpolated.

Output:
    - Interpolated data: <out_dir>/interpolated_<variable
    Set up enmpty grid and flag arraymatricies es to be populated in at interpoolation.
    >- grud    Builds 2D arrats ys of shape (nrows, ncols) for the grid.  and flgladlagdgflags.
    Populate the firlag frgrid with 1'ss where dinput data is present.
    .parquet
    - Includes 'interpolation_flag' column to distinguish original vs interpolated values
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
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    _resolve_basin_in_meta,
    _resolve_grid_params,
    write_metadata,
)
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers


def _parse_optional_float(value: Any) -> float | None:
    """Parse a numeric filter value, allowing null-like values to mean None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "None"}:
        return None
    return float(value)


def parse_arguments(args) -> argparse.Namespace:
    """
    Parse command line arguments for grid interpolation.

    Args:
        args (list): Command line arguments
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        default="interpolate_grids_of_dh",
    )
    # Required arguments
    parser.add_argument(
        "--in_dir",
        help="Input directory, with path to the directory containing variables of interest. "
        "E.g. epoch averaged grids",
        required=True,
    )
    parser.add_argument(
        "--parquet_glob",
        type=str,
        default="*.parquet",
        help="Glob pattern to match surface fit parquet files.",
    )
    parser.add_argument("--out_dir", help="Output directory", required=True)
    parser.add_argument(
        "--in_meta",
        type=str,
        required=False,
        help="Path to upstream metadata JSON for resolving grid parameters.",
    )

    parser.add_argument(
        "--timestamp_column", help="Timestamp/ grouping column to loop through", required=True
    )
    parser.add_argument(
        "--variables_in",
        help="Comma-separated list of variable(s) names to process",
        required=True,
        nargs="+",
    )
    parser.add_argument("--tri_lim", default=20.0)
    parser.add_argument(
        "--lo_filter",
        nargs="+",
        help="Lowest allowable data value, used to clip before and after interpolation. "
        "Lower values are removed, not set to the lo_filter value."
        "Space separated list of values for the number of variables",
    )
    parser.add_argument(
        "--hi_filter",
        nargs="+",
        help="Highest allowable data value, used to clip before and after interpolation. "
        "Higher values are removed, not set to the hi_filter value."
        "Space separated list of values for the number of variables",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    add_basin_selection_arguments(parser)

    return parser.parse_args(args)


def get_grid_and_flags(
    group: pl.DataFrame, nrows: int, ncols: int
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """
    Builds empty 2D (nrows, ncols) arrays for a single group (e.g. epoch).
    Marks cells that contain input data with flag 1, and
    returns the grid, flags, and populated indices for interpolation.

    Args:
        group (pl.DataFrame): Grouped DataFrame for a single epoch/period.
        nrows (int): Number of grid rows, from GridArea metadata.
        ncols (int): Number of grid columns, from GridArea metadata.

    Returns:
        grid (np.ndarray): 2D (nrows, ncols) variable array. Initialised with NaNs.
        grid_flags (np.ndarray): 2D (nrows, ncols) array. Initialised with 0.
        Cells with observations are set to 1.
        populated_idx (tuple[np.ndarray, np.ndarray]): Row and col indicies of populated cells.
        x_bins (np.ndarray): Populated x_bins
        y_bins (np.ndarray): Populated y_bins
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
    populated_idx = np.where(grid_flags != 0)

    return grid, grid_flags, populated_idx, x_bins, y_bins


def triangulate_points(
    params: argparse.Namespace,
    group_name: Any,
    populated_idx: tuple[np.ndarray, np.ndarray],
    logger: logging.Logger,
) -> Triangulation | None:
    """
    Build Delaunay triangulation object from populated grid cell indicies.
    Mask triangles whose longest edge exceeds params.tri_lim.

    Args:
        params (argparse.Namespace): Parsed command line arguments.
        group_name (str): Name of the epoch group or period being processed.
        populated_idx (tuple[np.ndarray, np.ndarray]): Row and column indices of populated cells.

    Returns:
        Triangulation or None: Triangulation object if successful, None otherwise.
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
    Interpolate missing grid cells for selected variables using Delaunay triangulation.

    A LinearTriInterpolator is fitted to the Triangulation and estimates values at for each
    missing grid cell.

    Flag values :
        0 : No data (missing input, and not interpolated)
        1 : Original input data
        2 : Interpolated data (added by LinearTriInterpolator)

    Args:
        params (argparse.Namespace): Parsed command line arguments.
        group_name (str): Name of the params.timestamp_column value being processed.
        group (pl.DataFrame): DataFrame for one grouped timestamp.
        nrows (int): Number of grid rows.
        ncols (int): Number of grid columns.

    Returns:
        tuple[dict[str, np.ndarray] | None, str | None]:
            - dict of interpolated variable arrays and flag arrays if successful, None otherwise.
            - String reason for failure if interpolation was not successful, None otherwise.

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


def apply_range_filters(params, group_df: pl.DataFrame) -> pl.DataFrame:
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
    Calculate statistics for a single group after interpolation, including counts of populated input cells, cells in output,
    and interpolated cells.

    Args:
        group_label (str): String label for the group being processed (e.g. epoch or period).
        rows_before (int): Number of rows in the group before interpolation.
        populated_input_cells (int): Number of populated input cells before interpolation.
        df (pl.DataFrame | None): DataFrame for the group after interpolation.
        grids (dict[str, np.ndarray]): Raw grids dict returned by get_trilinear_interpolation,
            or None if interpolation failed.
        failure_reason (str | None): Reason for interpolation failure, or None if interpolation succeeded.

    Returns:
        dict[str, Any]: Dictionary containing statistics for the group.
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
    """Aggregate per-epoch statistics for metadata output."""
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
    Process each params.timestamp_column group (e.g. Epoch or Period) in the input DataFrame:

    For each group :
        1. Extract static (single-value) metadata columns to re-attach after interpolation.
        2. Null out per-variable measurements outside lo/hi range filters.
        3. Interpolate missing grid cells via Delaunay triangulation calls : get_trilinear_interpolation
        4. Build a boolean mask of cells where every variable has a valid value (flag > 0).
        5. Assemble an output DataFrame for all valid cells, including interpolated
            values, flags, and static metadata columns.

    Concatinate all group DataFrames into a single output DataFrame.

    Args:
        params (argparse.Namespace): Parsed command line arguments.
        input_df (pl.DataFrame): DataFrame containing all epochs.
        nrows (int): Number of grid rows.
        ncols (int): Number of grid columns.

    Returns:
        pl.DataFrame: Concatenated DataFrame containing :
            x_bin,
            y_bin,
            params.timestamp_column,
            interpolated variables,
            variable flags,
            and static metadata columns for all valid grid cells across all groups.
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


def write_output(
    params: argparse.Namespace,
    interpolated_df: pl.DataFrame,
    start_time: float,
    status: dict[str, Any],
    output_dir: str,
) -> Path:
    """
    Write interpolated results and metadata to output files.

    Args:
        params (argparse.Namespace): Parsed command line arguments.
        interpolated_df (pl.DataFrame): Interpolated results for all epochs.
        start_time (float): Script start time (seconds since epoch).

    Returns:
        Path: Path to the output parquet file.
    """
    output_path = Path(output_dir) / f"{params.algo}_meta.parquet"
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    interpolated_df.write_parquet(output_path, compression="zstd")

    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    args_dict = dict(vars(params))
    if args_dict.get("gridarea") is None:
        args_dict.pop("gridarea", None)
    if args_dict.get("binsize") is None:
        args_dict.pop("binsize", None)

    meta_path = Path(output_dir) / f"{params.algo}_meta.json"
    try:
        write_metadata(
            params,
            meta_path,
            {
                **args_dict,
                "status": status,
                "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
            },
        )
    except OSError as e:
        sys.exit(str(e))
    return output_path


# -------------------
# Main function
# -------------------
def interpolate_grids_of_dh(args):
    """
    Main entry point for grid interpolation.

    1. Parse command line arguments
    2. Load grid metadata
    3. Read epoch data
    4. Interpolate missing grid cells
    5. Write results and metadata to output files

    Args:
        args (list): Command line arguments (typically sys.argv[1:])
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
            resolved = _resolve_grid_params(
                params,
                ["gridarea", "binsize"],
            )
        except ValueError as exc:
            logger.error("Couldn't resolve required grid parameters: %s", exc)
            sys.exit(str(exc))

        grid_area = str(resolved[0])
        binsize = float(resolved[1])

        ga = GridArea(grid_area, binsize)
        ncols, nrows = ga.get_ncols_nrows()
        input_df = pl.read_parquet(Path(params.in_dir) / params.parquet_glob)
        interpolated_df, status = process_timesteps(params, input_df, nrows, ncols, logger)
        write_output(params, interpolated_df, start_time, status, params.out_dir)
    else:
        logger.info("Processing basin/sub-basin level data")
        # Determine which sub-basins to process
        for sub_basin in get_basins_to_process(params, params.in_dir, logger=logger):
            logger.info("Processing sub-basin: %s", sub_basin)
            basin_in_dir = Path(params.in_dir) / sub_basin
            basin_out_dir = Path(params.out_dir) / sub_basin
            basin_out_dir.mkdir(parents=True, exist_ok=True)

            basin_params = argparse.Namespace(**vars(params))
            basin_params.in_dir = str(basin_in_dir)
            basin_params.out_dir = str(basin_out_dir)
            basin_params.region_selector = [sub_basin]
            basin_params.in_meta = _resolve_basin_in_meta(
                getattr(params, "in_meta", None),
                sub_basin,
                basin_in_dir,
                logger,
            )

            try:
                resolved = _resolve_grid_params(
                    basin_params,
                    ["gridarea", "binsize"],
                )
            except ValueError as exc:
                logger.error(
                    "Couldn't resolve required grid parameters for basin %s: %s",
                    sub_basin,
                    exc,
                )
                continue

            grid_area = str(resolved[0])
            binsize = float(resolved[1])

            ga = GridArea(grid_area, binsize)
            ncols, nrows = ga.get_ncols_nrows()
            input_df = pl.read_parquet(basin_in_dir / params.parquet_glob)
            interpolated_df, status = process_timesteps(
                basin_params, input_df, nrows, ncols, logger
            )
            write_output(basin_params, interpolated_df, start_time, status, str(basin_out_dir))


if __name__ == "__main__":
    interpolate_grids_of_dh(sys.argv[1:])

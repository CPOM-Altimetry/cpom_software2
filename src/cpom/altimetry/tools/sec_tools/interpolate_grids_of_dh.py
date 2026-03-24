"""
cpom.altimetry.tools.sec_tools.interpolate_grids_of_dh

Purpose:
    Interpolate missing values in gridded elevation change data.

    Uses Delaunay triangulation (matplotlib LinearTriInterpolator) to fill gaps
    in gridded data from epoch_average or calculate_dhdt outputs. Original input
    values are preserved; only missing cells are interpolated.

Output:
    - Interpolated data: <out_dir>/interpolated_<variable>.parquet
    - Includes 'interpolation_flag' column to distinguish original vs interpolated values
    - Metadata JSON: <out_dir>/interp_meta.json with processing parameters and summary statistics
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from matplotlib.tri import LinearTriInterpolator, Triangulation

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
    get_basins_to_process,
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


def _group_label(group_name: Any) -> Any:
    """Return a stable scalar key for group name"""
    if isinstance(group_name, tuple) and len(group_name) == 1:
        return group_name[0]
    return group_name


def parse_arguments(args) -> argparse.Namespace:
    """
    Parse command line arguments for grid interpolation.

    Args:
        args (list): Command line arguments
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()
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
        "--grid_info_json",
        help="Path to the grid metadata json file",
        required=False,
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
) -> tuple[np.ndarray, np.ndarray, tuple, np.ndarray, np.ndarray]:
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
        populated_idx (tuple): Row and col indicies of populated cells.
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

    # Populate grid flags with 1 where there is input data.
    grid_flags[y_bins, x_bins] = 1

    # Find points containing values
    populated_idx = np.where(grid_flags != 0)

    return grid, grid_flags, populated_idx, x_bins, y_bins


def triangulate_points(
    params: argparse.Namespace,
    group_name: Any,
    populated_idx: tuple,
    logger: logging.Logger,
) -> Triangulation | None:
    """
    Build Delaunay triangulation object from populated grid cell indicies.
    Mask triangles whose longest edge exceeds params.tri_lim.

    Args:
        params (argparse.Namespace): Parsed command line arguments.
        group_name (str): Name of the epoch group or period being processed.
        populated_idx: Row and column indices of populated cells.

    Returns:
        Triangulation or None: Triangulation object if successful, None otherwise.
    """
    # --------------------------------------------------------#
    # If no or too few points (3 is minimum, in the points
    # array that's 3x2 elements, so size 6)
    # --------------------------------------------------------#
    group_label = _group_label(group_name)

    if len(populated_idx[0]) < 6:
        logger.info("Not enough points for triangulation for group %s", group_label)
        return None
    try:
        tri_obj = Triangulation(populated_idx[0], populated_idx[1])
    # pylint: disable=W0703
    except Exception as e:
        logger.info("Triangulation error for group %s: %s", group_label, e)
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
    group_label = _group_label(group_name)

    grids = {}
    grid, grid_flags, populated_idx, x_bins, y_bins = get_grid_and_flags(group, nrows, ncols)
    if len(populated_idx[0]) < 6:
        logger.info("Not enough points for interpolation for group %s", group_label)
        return None, "not_enough_points"

    tri_obj = triangulate_points(params, group_label, populated_idx, logger)
    if tri_obj is None:
        logger.info("Triangulation failed for group %s, skipping interpolation", group_label)
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
            logger.error("LinearTriInterpolator error for group %s var %s: %s", group_label, var, e)
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


def apply_hi_lo_filters(params: argparse.Namespace, group_df: pl.DataFrame):
    """Filter out rows outside lo/hi bounds and count removals per variable
    Args:
        params (argparse.Namespace): Parsed command line arguments.
        group_df (pl.DataFrame): DataFrame for one grouped timestamp.
    Returns:
    tuple[pl.DataFrame, dict[str, int]]: Filtered DataFrame and
    dict of counts of removed rows per variable.
    """
    removed = {var: 0 for var in params.variables_in}
    filter_mask = pl.lit(True)

    for idx, var in enumerate(params.variables_in):
        # Remove rows where and variable to be interpolated is outside the lo/hi range
        lo = _parse_optional_float(params.lo_filter[idx]) if params.lo_filter else None
        hi = _parse_optional_float(params.hi_filter[idx]) if params.hi_filter else None
        if lo is not None:
            removed[var] += int(
                group_df.filter(pl.col(var).is_not_null() & (pl.col(var) < lo)).height
            )
            filter_mask = filter_mask & (pl.col(var) >= lo)
        if hi is not None:
            removed[var] += int(
                group_df.filter(pl.col(var).is_not_null() & (pl.col(var) > hi)).height
            )
            filter_mask = filter_mask & (pl.col(var) <= hi)

    return group_df.filter(filter_mask), removed


# pylint: disable=R0914
def process_timesteps(
    params: argparse.Namespace,
    input_df: pl.DataFrame,
    nrows: int,
    ncols: int,
    logger: logging.Logger,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Process each params.timestamp_column group (e.g. Epoch or Period) in the input DataFrame:

    For each group :
        1. Extract static (single-value) metadata columns to re-attach after interpolation.
        2. Null out per-variable measurements outside lo/hi range filters.
        3. Interpolate missing grid cells via Delaunay triangulation.
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
    groups = input_df.sort(params.timestamp_column).group_by(
        params.timestamp_column, maintain_order=True
    )

    # Detect columns with exactly one unique value per group (static metadata)
    non_variable_cols = [
        col
        for col in input_df.columns
        if col not in params.variables_in and col != params.timestamp_column
    ]
    unique_counts = (
        groups.agg(pl.col(c).n_unique() for c in non_variable_cols).select(non_variable_cols).max()
    )
    static_col_names = [c for c in non_variable_cols if unique_counts[c][0] == 1]

    logger.info("Processing groups based on: %s", params.timestamp_column)
    status_rows: list[dict[str, Any]] = []
    dataframes = []
    for group_name, group_df in groups:
        group_label = _group_label(group_name)
        populated_input_cells = group_df.select(pl.struct("y_bin", "x_bin").n_unique()).item()

        group_df, removed = apply_hi_lo_filters(params, group_df)
        grids, failure_reason = get_trilinear_interpolation(
            params,
            group_name,
            group_df,
            nrows,
            ncols,
            logger,
        )
        if grids is None:
            status_rows.append(
                {
                    "timestamp": group_label,
                    "populated_input_cells": int(populated_input_cells),
                    "cells_in_output": 0,
                    "interpolated_cells": 0,
                    "filter_removed_by_variable": removed,
                    "failure_reason": failure_reason,
                }
            )
            continue

        # Stack flag grids into a 3D stack
        # Find all grid cells indicies that are populated (Flag = 1 or 2)
        flag_grids = np.stack([grids[f"{var}_flags"] for var in params.variables_in], axis=-1)
        valid_mask = np.all(flag_grids > 0, axis=-1)
        interpolated_cells = int(np.sum(np.all(flag_grids == 2, axis=-1)))
        y_idx, x_idx = np.where(valid_mask)

        # Build output DataFrame for valid grid cells
        static_cols = {col: group_df[col][0] for col in static_col_names}
        df = pl.DataFrame(
            {
                "y_bin": y_idx,
                "x_bin": x_idx,
                params.timestamp_column: pl.Series(
                    params.timestamp_column, [group_label] * len(y_idx)
                ),
                **{var: grids[var][y_idx, x_idx] for var in params.variables_in if var in grids},
                **{
                    f"{var}_flag": grids[f"{var}_flags"][y_idx, x_idx]
                    for var in params.variables_in
                },
                **{col: pl.Series(col, [val] * len(y_idx)) for col, val in static_cols.items()},
            }
        )

        dataframes.append(df)
        status_rows.append(
            {
                "timestamp": group_label,
                "populated_input_cells": int(populated_input_cells),
                "cells_in_output": len(df),
                "interpolated_cells": interpolated_cells,
                "filter_removed_by_variable": removed,
                "failure_reason": None,
            }
        )
        if params.debug:
            logger.debug(
                "Group %s: populated_input_cells=%d, "
                "cells_in_output=%d, "
                "interpolated_cells=%d, "
                "filter_removed_by_variable=%s",
                group_label,
                populated_input_cells,
                len(df),
                interpolated_cells,
                removed,
            )

    if not dataframes:
        return pl.DataFrame(
            {
                "y_bin": pl.Series("y_bin", [], dtype=pl.Int64),
                "x_bin": pl.Series("x_bin", [], dtype=pl.Int64),
                params.timestamp_column: pl.Series(params.timestamp_column, []),
                **{var: pl.Series(var, [], dtype=pl.Float64) for var in params.variables_in},
                **{
                    f"{var}_flag": pl.Series(f"{var}_flag", [], dtype=pl.Int8)
                    for var in params.variables_in
                },
            }
        ), pl.DataFrame(status_rows)

    return pl.concat(dataframes), pl.DataFrame(status_rows)


def get_count_summary(status_df: pl.DataFrame) -> dict[str, Any]:
    """Build aggregate counts across all processed groups."""
    failed_groups = status_df.filter(pl.col("failure_reason").is_not_null()).height

    # Aggregate lo/hi filter removals per variable across all groups.
    filter_removed_totals: dict[str, int] = {}
    for item in status_df["filter_removed_by_variable"].to_list():
        if not isinstance(item, dict):
            continue
        for var, count in item.items():
            filter_removed_totals[var] = filter_removed_totals.get(var, 0) + int(count)

    summary: dict[str, Any] = {
        "groups_total": status_df.height,
        "groups_succeeded": status_df.height - failed_groups,
        "groups_failed": failed_groups,
        "rows_removed_by_variable_lo_hi": filter_removed_totals,
        "lo_hi_filter_removed_total": int(sum(filter_removed_totals.values())),
        "populated_input_cells_total": int(status_df["populated_input_cells"].sum()),
        "interpolated_cells_total": int(status_df["interpolated_cells"].sum()),
        "cells_in_output_total": int(status_df["cells_in_output"].sum()),
    }

    return summary


def write_output(
    params: argparse.Namespace,
    interpolated_df: pl.DataFrame,
    start_time: float,
    status_df: pl.DataFrame,
    sub_basin: str | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """
    Write interpolated results and metadata to output files.

    Args:
        params (argparse.Namespace): Parsed command line arguments.
        interpolated_df (pl.DataFrame): Interpolated results for all epochs.
        start_time (float): Script start time (seconds since epoch).
        status_df (pl.DataFrame): DataFrame containing counts and status for each processed group.
        sub_basin (str | None): Optional sub-basin name for output directory.
        logger (logging.Logger): Logger object for logging messages.
    Returns:
        Path: Path to the output parquet file.
    """
    if sub_basin is not None:
        if logger is not None:
            logger.info("Writing output for sub-basin %s to disk", sub_basin)
        out_dir = Path(params.out_dir) / sub_basin
        output_path = out_dir / "interpolated.parquet"
    else:
        if logger is not None:
            logger.info("Writing root-level output to disk")
        out_dir = Path(params.out_dir)
        output_path = out_dir / "interpolated.parquet"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interpolated_df.write_parquet(output_path, compression="zstd")

    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    try:
        metadata_payload: dict[str, Any] = {
            **vars(params),
            "counts": get_count_summary(status_df),
            "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
        }
        if params.debug:
            metadata_payload["counts_per_group"] = status_df.to_dicts()

        metadata_path = out_dir / "interp_meta.json"
        with open(metadata_path, "w", encoding="utf-8") as f_meta:
            json.dump(
                metadata_payload,
                f_meta,
                indent=2,
            )
    except OSError as e:
        sys.exit(str(e))
    return output_path


# --------------------
# Main Function
# --------------------


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
    params = parse_arguments(args)
    start_time = time.time()
    params = parse_arguments(args)

    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )

    with open(Path(params.grid_info_json), "r", encoding="utf-8") as f:
        grid_metadata = json.load(f)
    ga = GridArea(grid_metadata["gridarea"], grid_metadata["binsize"])
    ncols, nrows = ga.get_ncols_nrows()

    if params.basin_structure is False:
        logger.info("Processing root-level data")
        input_df = pl.read_parquet(Path(params.in_dir) / params.parquet_glob)
        interpolated_df, status_df = process_timesteps(params, input_df, nrows, ncols, logger)
        write_output(
            params,
            interpolated_df,
            start_time,
            status_df,
            sub_basin=None,
            logger=logger,
        )
    else:
        logger.info("Processing basin/sub-basin level data")
        # Determine which sub-basins to process
        for sub_basin in get_basins_to_process(params, params.in_dir, logger=logger):
            logger.info("Processing sub-basin: %s", sub_basin)
            input_df = pl.read_parquet(Path(params.in_dir) / sub_basin / params.parquet_glob)
            interpolated_df, status_df = process_timesteps(params, input_df, nrows, ncols, logger)
            write_output(
                params,
                interpolated_df,
                start_time,
                status_df,
                sub_basin=sub_basin,
                logger=logger,
            )


if __name__ == "__main__":
    interpolate_grids_of_dh(sys.argv[1:])

"""
cpom.altimetry.tools.sec_tools.interpolate_grids_of_dh.py

Purpose:
Take epoch-averaged gridded surface fit data and interpolate missing grid cells as far as possible,
for chosen variables using Delaunay triangulation (matplotlib linear tri-interpolator).
Input datapoints are not changed.

The code was written to take files output by epoch_average or
calculate_dhdt, but could take any other named variable.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.tri import LinearTriInterpolator, Triangulation

from cpom.gridding.gridareas import GridArea


def _initialize_grid_and_flags(group, nrows, ncols):
    # Populate grid with values
    grid = np.full((nrows, ncols), np.nan)
    grid[group["y_bin"].to_numpy(), group["x_bin"].to_numpy()] = 1
    points = np.where(np.isfinite(grid))
    points_arr = np.array(points)

    # Set up Flag array
    grid_flags = np.zeros_like(
        grid, dtype=np.byte
    )  # Flag array, 0 = no data, 1 = input, 2 = interpolated (added later)
    grid_flags[points] = 1

    return grid, grid_flags, points, points_arr


def _triangulate_points(points_arr, args):
    # --------------------------------------------------------#
    # If no or too few points (3 is minimum, in the points
    # array that's 3x2 elements, so size 6)
    # This will fail if eg triangles are co-linear,
    # so catch that event
    # --------------------------------------------------------#
    if points_arr.size < 6:
        print("Not enough points for triangulation.")
        return None
    try:
        tri_obj = Triangulation(points_arr[0, :], points_arr[1, :])
    # pylint: disable=W0703
    except Exception as e:
        print(f"trilinear_single: matplotlib.Triangulation error, returning with error: {e}")
        return None
    # --------------------------------------------------------#
    # Mask out triangles that are too long along any side
    # --------------------------------------------------------#

    xtri = tri_obj.x[tri_obj.triangles] - np.roll(tri_obj.x[tri_obj.triangles], 1, axis=1)
    ytri = tri_obj.y[tri_obj.triangles] - np.roll(tri_obj.y[tri_obj.triangles], 1, axis=1)
    maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)
    tri_obj.set_mask(maxi > args.tri_lim)
    return tri_obj


# pylint: disable=R0914
def get_trilinear_interpolation(group, nrows, ncols, args):
    """
    Interpolate missing grid cells for selected variables using Delaunay triangulation.

    Args:
        group (pl.DataFrame): Data for a single epoch group.
        nrows (int): Number of grid rows.
        ncols (int): Number of grid columns.
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        dict: Dictionary of interpolated grids for each variable and their flags.
    """

    grids = {}
    grid, grid_flags, points, points_arr = _initialize_grid_and_flags(group, nrows, ncols)
    tri_obj = _triangulate_points(points_arr, args)
    if tri_obj is None:
        return None

    grid_y, grid_x = np.meshgrid(range(grid.shape[0]), range(grid.shape[1]), indexing="ij")

    # -------------------------------------------------------------------------------#
    # Make predictions for all points on grid, again catch failure of interpolation
    # The interpolator returns a masked array, with NaNs where the mask is true. T
    # The input array was not masked, so only return the data
    # -------------------------------------------------------------------------------#

    for var in args.variables_in:
        values = group[var].to_numpy()
        grid[group["y_bin"].to_numpy(), group["x_bin"].to_numpy()] = values
        try:
            f_pred = LinearTriInterpolator(tri_obj, values)
            pred = f_pred(grid_y, grid_x)
            pred_data = pred.data
        # pylint: disable=W0703
        except Exception as e:
            print(
                f"trilinear_single: matplotlib.LinearTriInterpolator error,\
                                        returning with error: {e}"
            )
            return None

        # ---------------------------------------------------------------------------------#
        # The interpolator returns values at the input data points that vary very slightly
        #  from the original input. Doesn't return any input datapoints that lay only on
        # triangles that were removed. However, those datapoints are still good.
        # To fix both issues, copy existing input data to the output data before returning.
        # ----------------------------------------------------------------------------------#

        pred_data[points] = values
        grids[var] = pred_data

        # -----------------------------------------#
        # Add to flag grid: 2 = interpolated data
        # -----------------------------------------#
        r = np.where((np.isfinite(pred)) & (grid_flags == 0))
        if len(r[0]) > 0:
            grid_flags[r] = 2
        grids[f"{var}_flags"] = grid_flags

    return grids


def load_metadata(args):
    """
    Load grid metadata from file or from epoch averaged metadata.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        dict: Grid metadata dictionary.
    """

    if args.gridmeta_file:
        with open(Path(args.gridmeta_file), "r", encoding="utf-8") as f:
            grid_metadata = json.load(f)
    else:
        with open(Path(args.input_dir) / "epoch_avg_meta.json", "r", encoding="utf-8") as f:
            epoch_meta = json.load(f)
        with open(Path(epoch_meta["griddir"]), "r", encoding="utf-8") as f:
            grid_metadata = json.load(f)
    return grid_metadata


# pylint: disable=R0914
def process_group(input_df, args, nrows, ncols):
    """
    Process each group (e.g. Epoch or Period) in the input DataFrame:
    - Filter data
    - Interpolate missing grid cells
    - Build output DataFrame for valid grid cells and variables

    Steps:
    1. Group input DataFrame by the specified column (e.g., epoch).
    2. Filter data by lo/hi filters for each variable.
    3. Interpolate missing grid cells using Delaunay triangulation.
    4. Build flag grids and mask for valid interpolated cells.
    5. Collect unique columns for the group and add to output.
    6. Assemble output DataFrame for valid grid cells and variables.

    Args:
        input_df (pl.DataFrame): DataFrame containing all epochs.
        args (argparse.Namespace): Parsed command line arguments.
        nrows (int): Number of grid rows.
        ncols (int): Number of grid columns.

    Returns:
        pl.DataFrame: Concatenated DataFrame of interpolated results for all epochs.
    """
    # Group by the specified column (e.g., epoch if input is epoch averaged grids)
    groups = input_df.group_by(args.timestamp_column)
    dataframes = []
    for group_name, group_df in groups:
        # Filter group to only include data in range for each variable
        if args.lo_filter and args.hi_filter:
            for idx, var in enumerate(args.variables_in):
                if args.lo_filter[idx] is None or args.hi_filter[idx] is None:
                    tbl = group_df
                else:
                    tbl = group_df.filter(
                        (pl.col(var) > args.lo_filter[idx]) & (pl.col(var) < args.hi_filter[idx])
                    )
        else:
            tbl = group_df

        # Interpolate missing grid cells for this group
        grids = get_trilinear_interpolation(tbl, nrows, ncols, args)
        # Stack flag grids for all variables to create a mask of valid interpolated cells
        flag_grids = np.stack([grids[f"{var}_flags"] for var in args.variables_in], axis=-1)
        valid_mask = np.all(flag_grids > 0, axis=-1)
        y_idx, x_idx = np.where(valid_mask)

        exclude_cols = {args.timestamp_column, "x_bin", "y_bin"}

        # Add back columns that are unique to the group (single value per group)
        single_value_cols = {
            col: group_df[col].unique()[0]
            for col in group_df.columns
            if col not in exclude_cols and group_df[col].n_unique() == 1
        }

        # Build output DataFrame for valid grid cells
        df = pl.DataFrame(
            {
                "y_bin": y_idx,
                "x_bin": x_idx,
                args.timestamp_column: int(group_name[0]),
            }
        )

        # Add single-value columns to output DataFrame
        for col, value in single_value_cols.items():
            df = df.with_columns(pl.lit(value).alias(col))

        # Add interpolated variable values and flags to output DataFrame
        for var in args.variables_in:
            df = df.with_columns(
                pl.Series(var, grids[var][y_idx, x_idx]),
                pl.Series(f"{var}_flag", grids[f"{var}_flags"][y_idx, x_idx]),
            )
        dataframes.append(df)

    # Concatenate all group DataFrames into a single output DataFrame
    return pl.concat(dataframes)


def write_output(interpolated_df, args, grid_metadata, start_time):
    """
    Write interpolated results and metadata to output files.

    Args:
        interpolated_df (pl.DataFrame): Interpolated results for all epochs.
        args (argparse.Namespace): Parsed command line arguments.
        grid_metadata (dict): Grid metadata dictionary.
        start_time (float): Script start time (seconds since epoch).

    Returns:
        Path: Path to the output parquet file.
    """

    args.outdir = os.path.join(
        args.outdir,
        grid_metadata["mission"],
        f"{grid_metadata['gridarea']}_{int(grid_metadata['binsize']/1000)}km_{grid_metadata['mission']}",
    )
    output_path = Path(args.outdir) / "epoch_average_interp.parquet"
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.variables_in != args.variables_out:
        interpolated_df = interpolated_df.rename(
            {args.variables_in[i]: args.variables_out[i] for i in range(len(args.variables_in))}
        )

    interpolated_df.write_parquet(output_path, compression="zstd")
    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    try:
        with open(Path(args.outdir) / "epoch_avg_meta.json", "w", encoding="utf-8") as f_meta:
            json.dump(
                {
                    **vars(args),
                    "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
                },
                f_meta,
                indent=2,
            )
    except OSError as e:
        sys.exit(e)
    return output_path


def main(args):
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
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument(
        "--indir",
        help="Input filename, with path to thefile containing variables of interest",
        required=True,
    )
    parser.add_argument("--outdir", help="Output directory", required=True)
    parser.add_argument(
        "--gridmeta_file",
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
    parser.add_argument(
        "--variables_out",
        help="Comma-separated list of output variable(s) names, if not passed will use variables_in",
        required=False,
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

    args = parser.parse_args(args)
    grid_metadata = load_metadata(args)
    ga = GridArea(grid_metadata["gridarea"], grid_metadata["binsize"])
    ncols, nrows = ga.get_ncols_nrows()
    input_df = pl.read_parquet(Path(args.indir) / "*.parquet")

    interpolated_df = process_epochs(input_df, args, nrows, ncols)
    write_output(interpolated_df, args, grid_metadata, start_time)


if __name__ == "__main__":
    main(sys.argv[1:])

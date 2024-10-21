"""
cryotempo_2d_uncertainty

Purpose: Calculate a 2D uncertainty table using  slope & roughness bins
at 0.1 degree steps from 0 to 2 degrees, and m

Slope & Roughness interpolated from
AIS:
Slopes("rema_100m_900ws_slopes_zarr") or Slopes("cpom_ant_2018_1km_slopes")
Roughness("rema_100m_900ws_roughness_zarr")
Grn:
Slopes("arcticdem_100m_900ws_slopes_zarr")
Roughness("arcticdem_100m_900ws_roughness_zarr")

Input is dh values, lat, lon from CS2-IS2 differences npz files, for example:
[cs2_minus_is2_gt2lgt2r_p2p_diffs_antarctica_icesheets.npz
[cs2_minus_is2_gt2lgt2r_p2p_diffs_greenland.npz

example usage: 
python cryotempo_2d_uncertainty.py -a -m median \
    -dh_file ~/downloads/cs2_minus_is2_gt2lgt2r_p2p_diffs_antarctica_icesheets.npz
python cryotempo_2d_uncertainty.py -g -m median \
    -dh_file ~/downloads/cs2_minus_is2_gt2lgt2r_p2p_diffs_greenland.npz

"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from cpom.roughness.roughness import Roughness
from cpom.slopes.slopes import Slopes

# pylint: disable=too-many-locals,too-many-arguments


def calculate_binned_median(
    delta_elevation: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
) -> pd.DataFrame:
    """Calculate the median absolute elevation differences within slope and roughness bins.

    Args:
        delta_elevation (np.ndarray): Array of elevation differences between
                                      two measurement techniques.
        slope (np.ndarray): Array of surface slopes in meters.
        roughness (np.ndarray): Array of surface roughness values in meters.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.

    Returns:
        pd.DataFrame: A pivot table where rows correspond to slope bins, columns to roughness bins,
                      and values to the median absolute elevation difference within each bin.
    """
    # Create a DataFrame to hold the data
    data = pd.DataFrame(
        {
            "delta_elevation": np.abs(delta_elevation),  # Absolute elevation difference
            "slope": slope,
            "roughness": roughness,
        }
    )

    # Bin the slope and roughness values
    data["slope_bin"] = pd.cut(
        data["slope"], bins=slope_bins, include_lowest=True, labels=slope_bins[:-1]
    )
    data["roughness_bin"] = pd.cut(
        data["roughness"], bins=roughness_bins, include_lowest=True, labels=roughness_bins[:-1]
    )

    # Group by the bins and calculate the median of the absolute elevation difference
    # binned_median = (
    #     data.groupby(["slope_bin", "roughness_bin"])["delta_elevation"].median().reset_index()
    # )

    binned_median = (
        data.assign(abs_delta_elevation=data["delta_elevation"].abs())
        .groupby(["slope_bin", "roughness_bin"])["abs_delta_elevation"]
        .median()
        .reset_index()
    )

    # Pivot the table to create a 2D matrix where rows are slope_bins and columns are roughness_bins
    binned_median_pivot = binned_median.pivot(
        index="slope_bin", columns="roughness_bin", values="abs_delta_elevation"
    )

    return binned_median_pivot


def calculate_binned_mad(
    delta_elevation: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
) -> pd.DataFrame:
    """Calculate the median absolute deviation (MAD) of elevation differences within slope
      and roughness bins.

    Args:
        delta_elevation (np.ndarray): Array of elevation differences between two measurement
                                      techniques.
        slope (np.ndarray): Array of surface slopes in meters.
        roughness (np.ndarray): Array of surface roughness values in meters.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.

    Returns:
        pd.DataFrame: A pivot table where rows correspond to slope bins, columns to roughness bins,
                      and values to the MAD of the elevation differences within each bin.
    """
    # Create a DataFrame to hold the data
    data = pd.DataFrame(
        {
            "delta_elevation": delta_elevation,  # Raw elevation difference
            "slope": slope,
            "roughness": roughness,
        }
    )

    # Bin the slope and roughness values
    data["slope_bin"] = pd.cut(
        data["slope"], bins=slope_bins, include_lowest=True, labels=slope_bins[:-1]
    )
    data["roughness_bin"] = pd.cut(
        data["roughness"], bins=roughness_bins, include_lowest=True, labels=roughness_bins[:-1]
    )

    # Group by the bins and calculate the MAD of the elevation difference
    def mad(x):
        return np.median(np.abs(x - np.median(x)))
        # return median_absolute_deviation(x)

    binned_mad = (
        data.groupby(["slope_bin", "roughness_bin"])["delta_elevation"].apply(mad).reset_index()
    )

    # Pivot the table to create a 2D matrix where rows are slope_bins and columns are roughness_bins
    binned_mad_pivot = binned_mad.pivot(
        index="slope_bin", columns="roughness_bin", values="delta_elevation"
    )

    return binned_mad_pivot


def interpolate_missing_values_with_nearest_neighbour(
    binned_median_pivot: pd.DataFrame,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    secondary_fill: bool = True,
) -> pd.DataFrame:
    """Interpolate missing values in the binned median table using bilinear and
    nearest-neighbor interpolation.

    Args:
        binned_median_pivot (pd.DataFrame): A pivot table of binned median absolute
        elevation differences.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        secondary_fill (bool): perform a second fill

    Returns:
        pd.DataFrame: The pivot table with missing values interpolated.
    """
    # Create a grid of slope and roughness values corresponding to the bin centers
    slope_grid, roughness_grid = np.meshgrid(slope_bins[:-1], roughness_bins[:-1], indexing="ij")

    # Flatten the grid and the binned_median_pivot DataFrame to arrays for interpolation
    points = np.array([slope_grid.flatten(), roughness_grid.flatten()]).T
    values = binned_median_pivot.values.flatten()

    # Create a mask for valid (non-NaN) values
    valid_mask = ~np.isnan(values)

    # First, try bilinear interpolation
    interpolated_values = griddata(points[valid_mask], values[valid_mask], points, method="linear")

    # If bilinear interpolation leaves NaNs, use nearest-neighbor interpolation to fill them in
    if np.any(np.isnan(interpolated_values)) and secondary_fill:
        nearest_values = griddata(points[valid_mask], values[valid_mask], points, method="nearest")
        interpolated_values[np.isnan(interpolated_values)] = nearest_values[
            np.isnan(interpolated_values)
        ]

    # Reshape back to the original 2D shape
    interpolated_values = interpolated_values.reshape(binned_median_pivot.shape)

    # Replace the missing values in the original DataFrame
    binned_median_pivot = pd.DataFrame(
        interpolated_values, index=binned_median_pivot.index, columns=binned_median_pivot.columns
    )

    return binned_median_pivot


def get_binned_values(
    slope_values: np.ndarray,
    roughness_values: np.ndarray,
    binned_table: pd.DataFrame,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
) -> np.ndarray:
    """Retrieve the median absolute elevation difference for arrays of slope and roughness values.

    Args:
        slope_values (np.ndarray): Array of slope values for which to retrieve median differences.
        roughness_values (np.ndarray): Array of roughness values for which to retrieve
                                       median differences.
        binned_table (pd.DataFrame): A pivot table of binned median absolute elevation differences.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.

    Returns:
        np.ndarray: An array of median absolute elevation differences corresponding to the
                    input slope and roughness pairs.
    """
    # Convert slope_values and roughness_values to numpy arrays
    slope_values = np.asarray(slope_values)
    roughness_values = np.asarray(roughness_values)

    # Find the slope bin indices for the array of slope_values
    slope_bin_indices = np.digitize(slope_values, slope_bins) - 1
    slope_bin_indices = np.clip(
        slope_bin_indices, 0, len(slope_bins) - 2
    )  # Ensure indices are within range

    # Find the roughness bin indices for the array of roughness_values
    roughness_bin_indices = np.digitize(roughness_values, roughness_bins) - 1
    roughness_bin_indices = np.clip(
        roughness_bin_indices, 0, len(roughness_bins) - 2
    )  # Ensure indices are within range

    # Convert bin labels to row and column indices in the DataFrame
    row_indices = [binned_table.index.get_loc(slope_bins[idx]) for idx in slope_bin_indices]
    col_indices = [
        binned_table.columns.get_loc(roughness_bins[idx]) for idx in roughness_bin_indices
    ]

    # Retrieve the values using numpy indexing on the DataFrame values
    values = binned_table.values[row_indices, col_indices]

    return values


def calc_2d_uncertainty_table(
    delta_elevation: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    method: str = "median",
    fill=True,
    secondary_fill=True,
) -> pd.DataFrame:
    """Main function to calculate the 2D uncertainty table with interpolated missing values.

    Args:
        delta_elevation (np.ndarray): Array of elevation differences between two
                                      measurement techniques.
        slope (np.ndarray): Array of surface slopes in meters.
        roughness (np.ndarray): Array of surface roughness values in meters.
        method (str): Method to calculate the uncertainty ('median' or 'mad'). Default is 'median'
        fill (bool): Whether to fill missing values using bilinear interpolation.
        secondary_fill (bool): Whether to use nearest neighbor fill if bilinear
                               interpolation leaves NaNs.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.

    Returns:
        pd.DataFrame: A pivot table where rows correspond to slope bins, columns to roughness bins,
                      and values to the interpolated uncertainty metric within each bin.
    """
    if method == "mad":
        binned_table = calculate_binned_mad(
            delta_elevation, slope, roughness, slope_bins, roughness_bins
        )
    else:
        binned_table = calculate_binned_median(
            delta_elevation, slope, roughness, slope_bins, roughness_bins
        )

    if fill:
        # Interpolate missing values
        interpolated_binned_table = interpolate_missing_values_with_nearest_neighbour(
            binned_table,
            slope_bins,
            roughness_bins,
            secondary_fill=secondary_fill,
        )
    else:
        interpolated_binned_table = binned_table

    return interpolated_binned_table


def save_table_as_pickle(binned_table: pd.DataFrame, filename: str) -> None:
    """Save the binned table as a Pickle file.

    Args:
        binned_table (pd.DataFrame): The binned median absolute elevation difference table.
        filename (str): The path to the file where the table will be saved.
    """
    binned_table.to_pickle(filename)


def load_table_from_pickle(filename: str) -> pd.DataFrame:
    """Load the binned table from a Pickle file.

    Args:
        filename (str): The path to the file from which the table will be loaded.

    Returns:
        pd.DataFrame: The binned median absolute elevation difference table.
    """
    return pd.read_pickle(filename)


# ------------------------------------------------------------------------------------------------


def main():
    """main function for command line tool"""
    # initiate the command line parser
    parser = argparse.ArgumentParser()

    # add each argument
    parser.add_argument(
        "--ant",
        "-a",
        help=("process Antarctic 2D uncertainty"),
        action="store_true",
    )

    parser.add_argument(
        "--grn",
        "-g",
        help=("process Greenland 2D uncertainty"),
        action="store_true",
    )

    parser.add_argument(
        "--fill",
        "-f",
        help=("fill with bilinear interpolation"),
        action="store_true",
    )

    parser.add_argument(
        "--fill2",
        "-f2",
        help=("secondary fill with nearest neighbour"),
        action="store_true",
    )

    parser.add_argument(
        "-dh_file",
        "-dh",
        help=(
            "path of elevation difference npz file (for example "
            "/path/to/cs2_minus_is2_gt1lgt1rgt2lgt2rgt3lgt3r_p2p_diffs_greenland.npz or"
            "/path/to/"
            "cs2_minus_is2_gt1lgt1rgt2lgt2rgt3lgt3r_p2p_diffs_antarctica_icesheets.npz)"
        ),
        type=str,
    )

    parser.add_argument(
        "--method",
        "-m",
        choices=["median", "mad"],
        default="median",
        help=("choose the calculation method: 'median' (default) or 'mad'"),
    )

    parser.add_argument(
        "--cpom_ant_slp",
        "-cas",
        help=(
            "use cpom_ant_2018_1km_slopes for Antarctic slope instead of "
            "rema_100m_900ws_slopes_zarr"
        ),
        action="store_true",
    )

    # read arguments from the command line
    args = parser.parse_args()

    if not args.ant and not args.grn:
        sys.exit("Must have either --grn or --ant")

    # Define slope bins in degrees (0.1 degree steps from 0 to 2 degrees)
    the_slope_bins = np.arange(0, 2.1, 0.1)

    # Define roughness bins in meters (0.1 m steps from 0 to 2 meters)
    the_roughness_bins = np.arange(0, 2.1, 0.1)

    if args.ant:
        area = "ant"

        if args.cpom_ant_slp:
            this_slope = Slopes("cpom_ant_2018_1km_slopes")
        else:
            this_slope = Slopes("rema_100m_900ws_slopes_zarr")

        this_roughness = Roughness("rema_100m_900ws_roughness_zarr")

    else:
        area = "grn"
        this_slope = Slopes("arcticdem_100m_900ws_slopes_zarr")
        this_roughness = Roughness("arcticdem_100m_900ws_roughness_zarr")

    # Read npz file to get dh,lat,lon values

    print(f"reading npz file {args.dh_file}...")
    dh_data = np.load(args.dh_file, allow_pickle=True)

    lats = dh_data.get("lats")
    lons = dh_data.get("lons")
    dh = dh_data.get("dh")

    print("interpolating slope ")
    slope = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)
    print("interpolating roughness ")
    roughness = this_roughness.interp_roughness(lats, lons, method="linear", xy_is_latlon=True)

    print("calculating uncertainty table... ")
    binned_table = calc_2d_uncertainty_table(
        dh,
        slope,
        roughness,
        the_slope_bins,
        the_roughness_bins,
        method=args.method,
        fill=args.fill,
        secondary_fill=args.fill2,
    )
    print(binned_table)

    filename = f"/tmp/{area}_2d_uncertainty_table_{args.method}.pickle"
    if args.cpom_ant_slp:
        filename = f"/tmp/{area}_2d_uncertainty_table_{args.method}_cpom_ant_slp.pickle"
    save_table_as_pickle(binned_table, filename)

    retrieved_binned_table = load_table_from_pickle(filename)
    # Example slope and roughness pair
    slope_values = [0.15, 0.4]
    roughness_values = [0.25, 0.3]

    # Extract the corresponding value from the binned_table
    values = get_binned_values(
        slope_values, roughness_values, retrieved_binned_table, the_slope_bins, the_roughness_bins
    )
    print(f"Uncertainty for slope {slope_values} and roughness {roughness_values}: {values} m")


if __name__ == "__main__":
    main()

"""Plot an Antarctic or Greenland map of 2d uncertainty using the 
2d uncertainty LUT, slope and roughness, at a grid of lat/lon points"""

import argparse
import sys

import numpy as np
import pandas as pd

from cpom.areas.area_plot import Polarplot
from cpom.roughness.roughness import Roughness
from cpom.slopes.slopes import Slopes

# pylint: disable=too-many-locals,too-many-statements

# Define slope bins in degrees (0.1 degree steps from 0 to 2 degrees)
slope_bins = np.arange(0, 2.1, 0.1)

# Define roughness bins in meters (0.1 m steps from 0 to 2 meters)
roughness_bins = np.arange(0, 2.1, 0.1)


def load_table_from_pickle(filename: str) -> pd.DataFrame:
    """Load the binned table from a Pickle file.

    Args:
        filename (str): The path to the file from which the table will be loaded.

    Returns:
        pd.DataFrame: The binned median absolute elevation difference table.
    """
    return pd.read_pickle(filename)


def get_binned_values(
    slope_values: np.ndarray, roughness_values: np.ndarray, binned_table: pd.DataFrame
) -> np.ndarray:
    """Retrieve the median absolute elevation difference for arrays of slope and roughness values.

    Args:
        slope_values (np.ndarray): Array of slope values for which to retrieve median differences.
        roughness_values (np.ndarray): Array of roughness values for which to retrieve
                                       median differences.
        binned_table (pd.DataFrame): A pivot table of binned median absolute elevation differences.

    Returns:
        np.ndarray: An array of median absolute elevation differences corresponding to the input
                    slope and roughness pairs.
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


def main():
    """main function for tool"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--ant", "-a", help="process Antarctic 2D uncertainty", action="store_true")
    parser.add_argument("--grn", "-g", help="process Greenland 2D uncertainty", action="store_true")
    parser.add_argument(
        "--show_2m_binary", "-s", help="plot whether uncertainty is < or > 2m", action="store_true"
    )

    parser.add_argument("--density", "-d", help="density of grid", default=800, type=int)
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

    if args.ant:
        # Generate a grid of lat/lon values over Antarctica
        # latmin, latmax, lonmin, lonmax = -90.0, -50.0, 0.0, 360.0
        latmin, latmax, lonmin, lonmax = -90.0, -50.0, 0.0, 360.0

        area = "antarctica_is"
        table_area = "ant"
    else:
        # Generate a grid of lat/lon values over Greenland
        latmin, latmax, lonmin, lonmax = 40.0, 90.0, 0.0, 360.0
        area = "greenland_is"
        table_area = "grn"

    xx, yy = np.meshgrid(
        np.linspace(latmin, latmax, args.density), np.linspace(lonmin, lonmax, args.density)
    )
    lats = xx.flatten()
    lons = yy.flatten()
    # vals = np.zeros(lats.size)

    # Read LUT

    filename = f"/tmp/{table_area}_2d_uncertainty_table_bilinear_{args.method}.pickle"
    if args.cpom_ant_slp:
        filename = (
            f"/tmp/{table_area}_2d_uncertainty_table_bilinear_{args.method}_cpom_ant_slp.pickle"
        )
    retrieved_binned_table = load_table_from_pickle(filename)

    if args.cpom_ant_slp:
        this_slope = Slopes("cpom_ant_2018_1km_slopes")
    else:
        this_slope = Slopes("rema_100m_900ws_slopes_zarr")

    this_roughness = Roughness("rema_100m_900ws_roughness_zarr")

    print(f"Getting slope values from {this_slope.name}......")
    slope_values = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)
    print(f"Getting roughness values from {this_roughness.name}...")
    roughness_values = this_roughness.interp_roughness(
        lats, lons, method="linear", xy_is_latlon=True
    )

    print("Getting uncertainty values from table...")
    # Extract the corresponding value from the binned_table
    vals = get_binned_values(slope_values, roughness_values, retrieved_binned_table)

    name = f"2d uncertainty (method={args.method})"
    if args.cpom_ant_slp:
        name = f"2d uncertainty (method={args.method}), slp=CPOM (TS)"

    if not args.show_2m_binary:
        # Creating the dataset
        dataset = {
            "name": name,
            "units": "m",
            "lats": lats,
            "lons": lons,
            "vals": vals,
            "plot_size_scale_factor": 0.01,
            "min_plot_range": 0.0,
            "max_plot_range": 1.5,
        }

        Polarplot(area, area_overrides={"show_bad_data_map": False}).plot_points(
            dataset,
        )
    else:
        greater_than_2 = np.where(vals > 2.0)[0]
        less_than_2 = np.where(vals <= 2.0)[0]

        lats_greater = lats[greater_than_2]
        lons_greater = lons[greater_than_2]
        vals_greater = np.full_like(lats_greater, fill_value=1, dtype=int)

        lats_less = lats[less_than_2]
        lons_less = lons[less_than_2]
        vals_less = np.full_like(lats_less, fill_value=0, dtype=int)

        lats = np.concatenate((lats_less, lats_greater))
        lons = np.concatenate((lons_less, lons_greater))
        vals = np.concatenate((vals_less, vals_greater))

        # Creating the dataset
        dataset = {
            "name": name,
            "units": "m",
            "lats": lats,
            "lons": lons,
            "vals": vals,
            "plot_size_scale_factor": 0.01,
            "flag_values": [0, 1],  # list of flag values. If used vals treated as flag data
            "flag_names": ["< 2m", "> 2m"],  # list of flag names
            "flag_colors": ["red", "blue"],  # list of flag colors or colormap name to sample
        }

        Polarplot(area, area_overrides={"show_bad_data_map": False}).plot_points(
            dataset,
        )


if __name__ == "__main__":
    main()

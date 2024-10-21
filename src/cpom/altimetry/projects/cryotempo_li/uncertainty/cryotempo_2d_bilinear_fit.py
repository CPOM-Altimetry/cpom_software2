"""Converts 2d uncertainty table to a bilinear model fit

"""

import argparse
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements


def load_table_from_pickle(filename: str) -> pd.DataFrame:
    """Load the binned table from a Pickle file."""
    return pd.read_pickle(filename)


def save_table_as_pickle(binned_table: pd.DataFrame, filename: str) -> None:
    """Save the binned table as a Pickle file.

    Args:
        binned_table (pd.DataFrame): The binned median absolute elevation difference table.
        filename (str): The path to the file where the table will be saved.
    """
    binned_table.to_pickle(filename)


def main():
    """main function for tool

    Returns:
        None
    """
    # initiate the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--ant", "-a", help="process Antarctic 2D uncertainty", action="store_true")
    parser.add_argument("--grn", "-g", help="process Greenland 2D uncertainty", action="store_true")
    parser.add_argument(
        "--file", "-f", help="Path to the 2D uncertainty pickle file", type=str, default=None
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

    if args.ant:
        area = "ant"
    else:
        area = "grn"

    filename = args.file if args.file else f"/tmp/{area}_2d_uncertainty_table_{args.method}.pickle"
    if args.cpom_ant_slp:
        filename = f"/tmp/{area}_2d_uncertainty_table_{args.method}_cpom_ant_slp.pickle"

    # Check if file exists
    if not os.path.exists(filename):
        sys.exit(f"File not found: {filename}")

    df = load_table_from_pickle(filename)

    # Slope and Roughness bins
    slope_bins = df.index.values
    roughness_bins = df.columns.values

    # Reshaping data into long format for fitting
    slope, roughness = np.meshgrid(slope_bins, roughness_bins, indexing="ij")
    slope = slope.flatten()
    roughness = roughness.flatten()
    z_values = df.values.flatten()  # Uncertainty values

    # Remove NaN values before fitting
    mask = ~np.isnan(z_values)
    slope_clean = slope[mask]
    roughness_clean = roughness[mask]
    z_clean = z_values[mask]

    # Create interaction term (slope * roughness)
    interaction_term = slope_clean * roughness_clean

    # Prepare the features matrix
    x_matrix = np.vstack([slope_clean, roughness_clean, interaction_term]).T

    # Perform linear regression
    reg = LinearRegression()
    reg.fit(x_matrix, z_clean)

    # Coefficients of the bilinear model
    a, b, c = reg.coef_
    d = reg.intercept_

    # Bilinear fit function
    def bilinear_fit(slope, roughness):
        return a * slope + b * roughness + c * slope * roughness + d

    # Apply the bilinear fit to the original grid
    bilinear_table = pd.DataFrame(
        bilinear_fit(slope, roughness).reshape(df.shape), index=df.index, columns=df.columns
    )

    # Replace negative values with values from the original table
    bilinear_table = bilinear_table.where(bilinear_table >= 0, df)

    filename = f"/tmp/{area}_2d_uncertainty_table_bilinear_{args.method}.pickle"
    if args.cpom_ant_slp:
        filename = f"/tmp/{area}_2d_uncertainty_table_bilinear_{args.method}_cpom_ant_slp.pickle"
    save_table_as_pickle(bilinear_table, filename)

    # Visualization
    plt.figure(figsize=(10, 8))

    max_range = 10

    # Define the new colormap with an over color
    cmap = plt.get_cmap("viridis", 256)
    # Create a new instance of the colormap
    new_cmap = mcolors.ListedColormap(cmap(np.linspace(0, 1, 256)))  # Correct use of ListedColormap
    new_cmap.set_over("red")

    norm = mcolors.Normalize(vmin=bilinear_table.min().min(), vmax=max_range)
    heatmap = sns.heatmap(
        bilinear_table,
        annot=False,
        fmt=".2f",
        cmap=new_cmap,
        cbar=False,
        xticklabels=[f"{x:.1f}" for x in bilinear_table.columns.astype(float)],
        yticklabels=[f"{y:.1f}" for y in bilinear_table.index.astype(float)],
        vmax=max_range,
    )

    # Create colorbar with an arrow for over-color
    sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable
    cbar = plt.colorbar(sm, ax=heatmap.axes, extend="max")
    cbar.set_label("Elevation Difference (m)")

    plt.title(f"Binned Median Absolute Elevation Difference - {area.upper()}")
    plt.xlabel("Roughness (m)")
    plt.ylabel("Slope (degrees)")
    plt.gca().invert_yaxis()
    plt.show()

    # Visualization
    plt.figure(figsize=(10, 8))

    # Define the new colormap with an over color
    cmap = plt.get_cmap("viridis", 256)  # Create a new instance of the colormap
    new_cmap = mcolors.ListedColormap(cmap(np.linspace(0, 1, 256)))  # Correct use of ListedColormap
    new_cmap.set_over("red")

    norm = mcolors.Normalize(vmin=df.min().min(), vmax=max_range)
    heatmap = sns.heatmap(
        df,
        annot=False,
        fmt=".2f",
        cmap=new_cmap,
        cbar=False,
        xticklabels=[f"{x:.1f}" for x in df.columns.astype(float)],
        yticklabels=[f"{y:.1f}" for y in df.index.astype(float)],
        vmax=max_range,
    )

    # Create colorbar with an arrow for over-color
    sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable
    cbar = plt.colorbar(sm, ax=heatmap.axes, extend="max")
    cbar.set_label("Elevation Difference (m)")

    plt.title(f"Binned Median Absolute Elevation Difference - {area.upper()}")
    plt.xlabel("Roughness (m)")
    plt.ylabel("Slope (degrees)")
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    main()

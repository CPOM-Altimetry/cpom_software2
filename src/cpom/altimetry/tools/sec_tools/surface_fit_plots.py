"""
src.cpom.altimetry.tools.sec_tools.surface_fit_plots
Generate plots from the output of surface fit.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import polars as pl

from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea


def get_objects(args):
    """
    Get Area object from gridded metadata file.
    Get grid data as Polars DataFrame.

    Loads grid area and area name from either command line parameters
    or from the grid metadata JSON file outputted at gridding step.

    Args:
        args (argparse.namespace): Command Line Arguments

    Returns:
        Area: The Area object.
        DataFrame: The grid data as a Polars DataFrame.
    """
    grid_data = pl.read_parquet(Path(args.in_dir) / "grid_data.parquet")
    with open(Path(args.grid_info_json), "r", encoding="utf-8") as f:
        grid_meta = json.load(f)

    if args.grid_area is not None and args.binsize is not None:
        this_grid_area = GridArea(args.grid_area, args.binsize)
    else:
        this_grid_area = GridArea(grid_meta["gridarea"], grid_meta["binsize"])

    if args.area_name is not None:
        this_area = Area(args.area_name)
    else:
        this_area = Area(grid_meta["area"])

    lats, lons = this_grid_area.get_cellcentre_lat_lon_from_col_row(
        grid_data.select(pl.col("x_bin")).to_numpy(), grid_data.select(pl.col("y_bin")).to_numpy()
    )

    grid_data = grid_data.with_columns([pl.Series("latitude", lats), pl.Series("longitude", lons)])

    return this_area, grid_data


# pylint: disable=too-many-arguments, too-many-positional-arguments
def plot(args, data, area, value_column, title, plot_range):
    """
    Generate plots for a given value column, using Polarplot.
    Args:
        args (argparse.namespace): Command Line Arguments
        data (DataFrame): The grid data as a Polars DataFrame.
        area (Area): The Area object.
        value_column (str): The column name to plot.
        title (str): The title for the plot.
        plot_range (tuple): The range for filtering the data (min, max).
    """
    grid_data_f = data.filter(
        (pl.col(value_column) > plot_range[0]) & (pl.col(value_column) < plot_range[1])
    )
    Polarplot(area.name).plot_points(
        {
            "name": title,
            "lats": grid_data_f.select(pl.col("latitude")),
            "lons": grid_data_f.select(pl.col("longitude")),
            "vals": grid_data_f.select(pl.col(value_column)),
        },
        output_dir=Path(args.out_dir) / "plots",
    )


def main(args):
    """
    Main function to generate surface fit plots.
    """

    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument(
        "--in_dir",
        help="Path to the directory containing surface fit data files.",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        help="Path to the output directory.",
        required=True,
    )
    parser.add_argument(
        "--grid_info_json",
        help="Path to the grid info JSON file.",
        required=True,
    )
    parser.add_argument(
        "--area_name",
        help="Name of the area to plot. If not provided, will be read from " "grid info JSON.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--grid_area",
        help="Grid area name. If not provided, will be read from grid info JSON.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--binsize",
        type=float,
        help="Grid bin size. If not provided, will be read from grid info JSON.",
        required=False,
        default=None,
    )
    args = parser.parse_args()

    os.makedirs(Path(args.out_dir) / "plots", exist_ok=True)

    area, grid_data = get_objects(args)

    plot(args, grid_data, area, "slope", "Slope", [0, 2])
    plot(args, grid_data, area, "dhdt", "dh/dt", [-1.0, 1.0])
    plot(args, grid_data, area, "rms", "RMS of linear fit", [0.0, 1.0])
    plot(args, grid_data, area, "dhdt", "Sigma: std of linear fit", [0.0, 2.0])
    return args


if __name__ == "__main__":
    main(sys.argv[1:])

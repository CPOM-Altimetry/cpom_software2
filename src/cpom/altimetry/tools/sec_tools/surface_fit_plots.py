"""
src.cpom.altimetry.tools.sec_tools.surface_fit_plots

Purpose:
    Generate plots from the output of surface fit.
    Uses CPOM Area and Polarplot classes for plotting.
    Can plot any area defined in CPOM areas.

    Loads grid_data.parquet and grid metadata JSON to create plots of:
    - Slope
    - dh/dt
    - RMS of linear fit
    - Sigma (std of linear fit)
Output:
    - Point plots saved to <out_dir>/plots/ for each variable.
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


def get_objects(params: argparse.Namespace) -> tuple[Area, pl.DataFrame]:
    """
    Load grid metadata and construct plotting objects.
    grid_data.parquet into a Polars DataFrame and compute latitude/longitude columns.
    Get GridArea and Area from grid metadata JSON or command line arguments.

    params:
        params (argparse.namespace): Command Line Arguments

    Returns:
        tuple[Area, pl.DataFrame]:
            (Area object, grid data with latitude/longitude columns)
    """
    grid_data = pl.read_parquet(Path(params.in_dir) / "grid_data.parquet")
    with open(Path(params.grid_info_json), "r", encoding="utf-8") as f:
        grid_meta = json.load(f)

    if params.grid_area is not None and params.binsize is not None:
        this_grid_area = GridArea(params.grid_area, params.binsize)
    else:
        this_grid_area = GridArea(grid_meta["gridarea"], grid_meta["binsize"])

    if params.area_name is not None:
        this_area = Area(params.area_name)
    else:
        this_area = Area(grid_meta["area"])

    lats, lons = this_grid_area.get_cellcentre_lat_lon_from_col_row(
        grid_data.select(pl.col("x_bin")).to_numpy(), grid_data.select(pl.col("y_bin")).to_numpy()
    )

    grid_data = grid_data.with_columns([pl.Series("latitude", lats), pl.Series("longitude", lons)])

    return this_area, grid_data


# pylint: disable=too-many-arguments, too-many-positional-arguments
def plot(
    params: argparse.Namespace,
    data: pl.DataFrame,
    area: Area,
    value_column: str,
    title: str,
    plot_range: tuple[float, float],
) -> None:
    """
    Generate and save a point plot for a grid value column.

    Filters the input data to the specified value range, then uses
    Polarplot(area.name).plot_points to create and save the plot.

    params:
        params (argparse.Namespace): Command line arguments,
        data (pl.DataFrame): Surface fit grid data
        area (Area): CPOM Area object
        value_column (str): Column name to plot.
        title (str): Plot title.
        plot_range (tuple[float, float]): Value range (min, max) used to
            filter points prior to plotting.
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
        output_dir=str(Path(params.out_dir) / "plots"),
    )


def main(args):
    """
    Main function to generate standard surface fit plots.
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
    params = parser.parse_args(args)

    os.makedirs(Path(params.out_dir) / "plots", exist_ok=True)

    area, grid_data = get_objects(params)

    plot(params, grid_data, area, "slope", "Slope", (0.0, 2.0))
    plot(params, grid_data, area, "dhdt", "dh/dt", (-1.0, 1.0))
    plot(params, grid_data, area, "rms", "RMS of linear fit", (0.0, 1.0))
    plot(params, grid_data, area, "sigma", "Sigma: std of linear fit", (0.0, 2.0))


if __name__ == "__main__":
    main(sys.argv[1:])

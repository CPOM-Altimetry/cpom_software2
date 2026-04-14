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
import os
import sys
import time
from pathlib import Path

import polars as pl

from cpom.altimetry.tools.sec_tools.metadata_helper import (
    get_algo_name,
    get_grid_params,
    write_metadata,
)
from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea


def parse_arguments(args):
    """
    Parse command line arguments for surface fit plots

    Return:
        parser.parse_args(args)
    """

    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument(
        "--in_dir",
        help="Path to the directory containing surface fit data files.",
        required=True,
    )
    parser.add_argument(
        "--in_meta",
        help="Path to input metadata file",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--out_dir",
        help="Path to the output directory.",
        required=True,
    )
    parser.add_argument(
        "--area_name",
        help="Name of the area to plot. If not provided, will be read from " "grid info JSON.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--gridarea",
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
    return parser.parse_args(args)


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
    grid_params = get_grid_params(
        params,
        ["gridarea", "binsize", "area"],
        "grid_for_elev_change",
    )

    this_grid_area = GridArea(str(grid_params["gridarea"]), float(grid_params["binsize"]))
    this_area = Area(str(grid_params["area"]))

    lats, lons = this_grid_area.get_cellcentre_lat_lon_from_col_row(
        grid_data.select(pl.col("x_bin")).to_numpy(), grid_data.select(pl.col("y_bin")).to_numpy()
    )

    grid_data = grid_data.with_columns([pl.Series("latitude", lats), pl.Series("longitude", lons)])

    return this_area, grid_data


def plot(
    params: argparse.Namespace,
    data: pl.DataFrame,
    area: Area,
    plot_params: tuple[str, str, tuple[float, float]],
) -> None:
    """
    Generate and save a point plot for a grid value column.

    Filters the input data to the specified value range, then uses
    Polarplot(area.name).plot_points to create and save the plot.

    params:
        params (argparse.Namespace): Command line arguments,
        data (pl.DataFrame): Surface fit grid data
        area (Area): CPOM Area object
        plot_params:
            (value_column, title, plot_range)
    """
    value_column, title, plot_range = plot_params

    grid_data_f = data.filter(
        (pl.col(value_column) > plot_range[0]) & (pl.col(value_column) < plot_range[1])
    )
    Polarplot(area.name).plot_points(
        {
            "name": title,
            "lats": grid_data_f["latitude"].to_numpy(),
            "lons": grid_data_f["longitude"].to_numpy(),
            "vals": grid_data_f[value_column].to_numpy(),
        },
        output_dir=str(Path(params.out_dir) / "plots"),
    )


# ---------------------------
# Main Function #
# ---------------------------


def surface_fit_plots(args):
    """
    Main function to generate standard surface fit plots.
    """
    start_time = time.time()

    params = parse_arguments(args)

    os.makedirs(Path(params.out_dir) / "plots", exist_ok=True)

    area, grid_data = get_objects(params)
    plot_params = {
        "slope": (0.0, 2.0),
        "dhdt": (-1.0, 1.0),
        "rms": (0.0, 1.0),
        "sigma": (0.0, 2.0),
    }

    plot_labels = {
        "slope": "Slope",
        "dhdt": "dh/dt",
        "rms": "RMS of linear fit",
        "sigma": "Sigma: std of linear fit",
    }

    for key, limits in plot_params.items():
        plot(params, grid_data, area, (key, plot_labels[key], limits))

    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    write_metadata(
        params,
        get_algo_name(__file__),
        Path(params.out_dir),
        {
            **vars(params),
            **{"plot_params": plot_params},
            "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
        },
    )


if __name__ == "__main__":
    surface_fit_plots(sys.argv[1:])

"""
src.cpom.altimetry.tools.sec_tools.surface_fit_plots

Purpose:
    Generate plots from the output of surface fit model residuals.
    Plots include:
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
    elapsed,
    get_algo_name,
    get_metadata_params,
    write_metadata,
)
from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea


def parse_arguments(args):
    """Parse command line arguments for surface fit plotting."""

    parser = argparse.ArgumentParser()
    # I/O Arguments
    parser.add_argument(
        "--in_step",
        type=str,
        required=False,
        help="Input algorithm step to source metadata from",
    )
    # Required arguments
    parser.add_argument("--in_dir", required=True, help="Input data directory")
    parser.add_argument("--out_dir", required=True, help="Output directory Path")

    # Optional grid metadata overrides
    parser.add_argument(
        "--area",
        default=None,
        help="Name of the area to plot.  Grid metadata fallback",
    )
    parser.add_argument(
        "--gridarea",
        default=None,
        help="Grid area name. Grid metadata fallback",
    )
    parser.add_argument(
        "--binsize",
        type=float,
        default=None,
        help="Grid bin size. Grid metadata fallback",
    )
    return parser.parse_args(args)


def get_objects(params: argparse.Namespace) -> tuple[Area, pl.DataFrame]:
    """
    Load grid metadata and construct Area object for plotting.

    params:
        params (argparse.namespace): Command Line Arguments

    Returns:
        tuple:
            - Area : CPOM Area object for the area being plotted
            - pl.DataFrame : Grid data with added latitude and longitude columns
    """
    grid_data = pl.read_parquet(Path(params.in_dir) / "grid_data.parquet")
    grid_params = get_metadata_params(
        params=params,
        fields=["gridarea", "binsize", "area"],
        algo_name="grid_for_elev_change",
    )

    this_grid_area = GridArea(str(grid_params["gridarea"]), int(grid_params["binsize"]))
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
    Generate ice-sheet-wide Polarplots of surface fit parameters.
    Filters the input data to the specified value range, then uses
    Polarplot(area.name).plot_points to create and save the plot.

    Args:
        params (argparse.Namespace): Command line arguments.
        data (pl.DataFrame): Surface fit grid data
        area (Area): CPOM Area object
        plot_params: (value_column, title, plot_range)
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
    Main entry point for surface fit plotting.
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

    write_metadata(
        params,
        get_algo_name(__file__),
        Path(params.out_dir),
        {
            **vars(params),
            **{"plot_params": plot_params},
            "execution_time": elapsed(start_time),
        },
    )


if __name__ == "__main__":
    surface_fit_plots(sys.argv[1:])

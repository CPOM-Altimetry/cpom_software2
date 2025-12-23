"""
cpom.altimetry.tools.sec_tools.dhdt_plots

Purpose:
    Generate spatial plots of dh/dt (elevation change rate) data from calculate_dhdt.py output.

    Creates scatter plots showing elevation change rates across ice sheet regions or basins,
    with optional glacier boundary overlays from shapefiles.

Output:
    - Basin plots: Saved to <out_dir>/<basin>/plots/<basin>_dhdt_period_{id}_{start}-{end}.png
    - Ice sheet plots: Saved to <out_dir>/plots/dhdt_period_{id}_{start}-{end}.png
"""

import argparse
import os
import sys
from pathlib import Path

import polars as pl
from epoch_average_plots import get_data, get_objects, get_shapefile
from matplotlib import pyplot as plt

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
    get_basins_to_process,
)
from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Generate plots from dhdt.parquet files produced by calculate_dhdt.py"
    )

    # I/O Arguments
    parser.add_argument(
        "--in_dir",
        help="Path to the directory containing dhdt.parquet file.",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        help="Path to the output directory for plots.",
        required=True,
    )
    parser.add_argument(
        "--grid_info_json",
        help="Path to the metadata JSON file from calculate_dhdt.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--parquet_glob",
        type=str,
        default="*dhdt.parquet",
        help="Glob pattern to match parquet files in input directory.",
    )
    # Plotting Arguments
    parser.add_argument(
        "--area_name",
        help="Name of the area to plot. If not provided, will be read from metadata JSON.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--grid_area",
        help="Grid area name. If not provided, will be read from metadata JSON.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--binsize",
        type=float,
        help="Grid bin size. If not provided, will be read from metadata JSON.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--plot_range",
        nargs=2,
        type=float,
        help="Plot range for dhdt values [min max]. Default: -1.0 1.0",
        default=[-1.0, 1.0],
    )
    parser.add_argument(
        "--shapefile", type=str, help="Path to the shapefile to use for clipping", required=False
    )
    parser.add_argument(
        "--shp_file_column",
        type=str,
        help="Column name in the shapefile to use for basin selection",
        required=False,
    )
    parser.add_argument(
        "--mask",
        type=str,
        help="CPOM mask name to use for loading predefined shapefile boundaries.",
        required=False,
    )
    # Column name arguments
    parser.add_argument(
        "--plotting_column",
        type=str,
        default="dhdt",
        help="Column name for dh/dt values in the parquet file.",
    )
    parser.add_argument(
        "--period_column",
        type=str,
        default="period",
        help="Column name for period identifiers in the parquet file.",
    )
    # Shared basin/region selection arguments
    add_basin_selection_arguments(parser)
    return parser.parse_args(args=args)


def plot_basins(
    params: argparse.Namespace,
    this_grid_area: GridArea,
    sub_basins_to_process: list,
):
    """
    Generate spatial scatter plots of dh/dt data for each basin and time period.

    For each basin and period, creates a plot showing:
    - Scatter points colored by dh/dt value
    - Glacier/basin boundaries from shapefile overlay
    - Color scale and axis labels

    Output files: <out_dir>/<basin>/plots/<basin>_dhdt_period_{id}_{start}-{end}.png

    Args:
        params (argparse.Namespace): Command line arguments.
            Includes:
            - in_dir (str): Input directory with dhdt.parquet files
            - out_dir (str): Output directory for plots
            - shapefile (str): Shapefile identifier for boundaries
            - plot_range (list): [min, max] for color scale
        this_area (Area): CPOM Area object (used for fallback plotting)
        this_grid_area (GridArea): CPOM Grid area object
        sub_basins_to_process (list[str]): Basin paths to process (e.g., ["West/H-Hp", "East/A-Ap"])
    """

    def _get_output_path(sub_basin, period_id, period_data):
        """
        Generate output file path with formatted period date range.

        Args:
            sub_basin (str): Sub-basin name
            period_id (int): Period identifier
            period_data (pl.DataFrame): Data for the specific period

        Returns:
            Path: Output file path for the plot
        """
        start_str = period_data.select(
            pl.col("input_dh_start_time").min().dt.strftime("%Y-%m-%d")
        ).item()
        end_str = period_data.select(
            pl.col("input_dh_end_time").max().dt.strftime("%Y-%m-%d")
        ).item()

        plot_dir = Path(params.out_dir) / sub_basin
        plot_dir.mkdir(parents=True, exist_ok=True)
        return (
            plot_dir
            / f"{sub_basin.replace("/", "_")}_dhdt_period_{period_id}_{start_str}-{end_str}.png"
        )

    shp, selector = get_shapefile(params)

    for sub_basin in sub_basins_to_process:
        # Get geometry for this specific basin using the helper function
        data = get_data(params, this_grid_area, sub_basin)
        period_ids = (
            data.select(pl.col(params.period_column)).unique().collect().to_series().to_list()
        )

        for period_id in period_ids:
            period_data = data.filter(pl.col(params.period_column) == period_id).collect()
            if period_data.height == 0:
                continue

            outpath = _get_output_path(sub_basin, period_id, period_data)
            # Create plot
            _, ax = plt.subplots(figsize=(10, 8))

            scatter = ax.scatter(
                period_data["x"],
                period_data["y"],
                c=period_data[params.plotting_column],
                cmap="coolwarm",
                vmin=params.plot_range[0],
                vmax=params.plot_range[1],
                s=20,
                alpha=1.0,
            )

            plt.colorbar(scatter, ax=ax, label="dh/dt (m/yr)")

            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_aspect("equal")

            # Get actual data date range
            data_start = period_data.select(pl.col("input_dh_start_time").min()).item()
            data_end = period_data.select(pl.col("input_dh_end_time").max()).item()
            data_start_str = (
                str(data_start).split()[0] if " " in str(data_start) else str(data_start)
            )
            data_end_str = str(data_end).split()[0] if " " in str(data_end) else str(data_end)
            ax.set_title(f"dhdt period {period_id}: {data_start_str} to {data_end_str}")

            # Overlay glacier boundaries
            if shp is not None:
                shp[shp[selector] == sub_basin].boundary.plot(ax=ax, edgecolor="black", linewidth=1)
            plt.tight_layout()
            plt.savefig(outpath)

            plt.close()


def plot_icesheet(params: argparse.Namespace, this_area: Area, this_grid_area: GridArea):
    """
    Generate ice sheet-wide spatial plots of dh/dt data .

    Creates plots for each time period showing elevation change rates across the entire
    ice sheet using the Polarplot class.

    Output files: <out_dir>/plots/dhdt_period_{id}_{start}-{end}.png

    Args:
        params (argparse.Namespace): Command line arguments.
            Includes:
            - in_dir (str): Input directory with dhdt.parquet file
            - out_dir (str): Output directory for plots
            - plot_range (list): [min, max] for color scale (m/yr)
        this_area (Area): CPOM Area object
        this_grid_area (GridArea): CPOM Grid area object
    """

    data = get_data(params, this_grid_area, sub_basin=None)
    period_ids = sorted(
        data.select(pl.col(params.period_column)).unique().collect().to_series().to_list()
    )
    Path(params.out_dir).mkdir(parents=True, exist_ok=True)
    for period_id in period_ids:
        period_data = data.filter(pl.col(params.period_column) == period_id).collect()

        # Get period boundaries (not actual data times)
        start_time = period_data.select(pl.col("period_lo").min()).item()
        end_time = period_data.select(pl.col("period_hi").max()).item()
        start_time_str = str(start_time).replace(":", "-").replace(" ", "_")
        end_time_str = str(end_time).replace(":", "-").replace(" ", "_")
        Path(params.out_dir).mkdir(parents=True, exist_ok=True)

        Polarplot(this_area.name).plot_points(
            {
                "name": f"dhdt_{period_id}",
                "lats": period_data["latitude"],
                "lons": period_data["longitude"],
                "vals": period_data[params.plotting_column],
                "valid_range": tuple(params.plot_range),
                "units": "m",
                "cmap_name": "coolwarm",
            },
            output_dir=str(Path(params.out_dir)),
            output_file=f"dhdt_period_{period_id}_{start_time_str}-{end_time_str}.png",
        )


def main(params):
    """
    Epoch Average plotting script.

    Steps :
    1. Parses command line arguments.
    2. Determines which sub-basins to process.
    3. Loads Area and GridArea objects.
    4. Plots either entire ice sheet or individual sub-basins based on arguments.

    params:
        params (list): Command line arguments
    """

    params = parse_arguments(params)
    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_file_info=Path(params.out_dir) / "info.log",
        log_file_error=Path(params.out_dir) / "errors.log",
        log_file_warning=Path(params.out_dir) / "warnings.log",
    )

    this_area, this_grid_area = get_objects(params)

    if params.basin_structure is False:
        logger.info("Processing root-level data")
        plot_icesheet(params, this_area, this_grid_area)
    else:
        logger.info("Processing basin/sub-basin level data")
        # Determine which sub-basins to process
        sub_basins_to_process = get_basins_to_process(params, params.in_dir, logger=logger)
        plot_basins(params, this_grid_area, sub_basins_to_process)


if __name__ == "__main__":
    main(sys.argv[1:])

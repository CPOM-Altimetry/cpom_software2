"""
src.cpom.altimetry.tools.sec_tools.dhdt_plots
Generate plots from the output of calculate_dhdt.py (dhdt.parquet files)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl
from matplotlib import pyplot as plt

from cpom.altimetry.tools.sec_tools.clip_to_glaciers import get_shapefile
from cpom.altimetry.tools.sec_tools.cross_calibrate_missions import get_sub_basins
from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea

log = logging.getLogger(__name__)


def get_objects(args):
    """
    Get Area object and GridArea from metadata file.
    Get dhdt data as Polars DataFrame.

    Loads grid area and area name from either command line parameters
    or from the metadata JSON file outputted at dhdt calculation step.

    Args:
        args (argparse.Namespace): Command line arguments

    Returns:
        tuple: A tuple containing:
            - Area: The Area object
            - GridArea: The GridArea object
    """

    # Get grid area and area name from args or metadata
    if args.grid_area is not None and args.binsize is not None:
        this_grid_area = GridArea(args.grid_area, args.binsize)
        this_area = Area(args.area_name)
    else:
        with open(Path(args.metadata_json), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        this_grid_area = GridArea(
            metadata.get("grid_name", "greenland"), metadata.get("binsize", 5000)
        )
        this_area = Area(metadata.get("area", "greenland"))

    return this_area, this_grid_area


def get_data(args, this_grid_area, sub_basin=None):
    """
    Load dhdt data from parquet file.
    Add lat/lon columns based on grid area.

    Args:
        args (argparse.Namespace): Command line arguments with in_dir
        this_grid_area (GridArea): Grid area object for coordinate conversion
        sub_basin (str, optional): Sub-basin name. If None or "None", loads root-level data.
            Defaults to None.

    Returns:
        pl.LazyFrame: The dhdt data as a Polars LazyFrame with latitude and longitude columns.
    """

    if sub_basin is not None and sub_basin != "None":
        dhdt_data = pl.scan_parquet(Path(args.in_dir) / sub_basin / "dhdt.parquet")
    else:
        dhdt_data = pl.scan_parquet(Path(args.in_dir) / "dhdt.parquet")

    # Convert grid coordinates to lat/lon
    lats, lons = this_grid_area.get_cellcentre_lat_lon_from_col_row(
        dhdt_data.select(pl.col("x_bin")).collect().to_numpy(),
        dhdt_data.select(pl.col("y_bin")).collect().to_numpy(),
    )

    dhdt_data = dhdt_data.with_columns([pl.Series("latitude", lats), pl.Series("longitude", lons)])

    return dhdt_data


def plot_basins(args: argparse.Namespace, this_grid_area: GridArea, sub_basins_to_process: list):
    """
    Plot dhdt data for each sub-basin by period.
    Load shapefile for glacier boundaries using get_shapefile function from clip_to_glaciers module.

    Save plots to args.out_dir/plots/dhdt_period_{period_id}_{start_time}-{end_time}.png

    Args:
        args (argparse.Namespace): Command Line Arguments
        this_grid_area (GridArea): GridArea object
        sub_basins_to_process (list): List of sub-basins to process
    """

    def _plot(args, period_data, period_id, shp, outpath):
        """
        Create and save scatter plot for basin dhdt data.

        Args:
            args (argparse.Namespace): Command line arguments with plot_range
            period_data (pl.DataFrame): Filtered period data with x, y, dhdt columns
            period_id (int): Period identifier for plot title
            shp (GeoDataFrame): Shapefile with glacier boundaries
            outpath (Path): Output file path for saved plot
        """
        # Create plot
        _, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(
            period_data["x"],
            period_data["y"],
            c=period_data["dhdt"],
            cmap="coolwarm",
            vmin=args.plot_range[0],
            vmax=args.plot_range[1],
            s=20,
            alpha=1.0,
        )

        plt.colorbar(scatter, ax=ax, label="dh/dt (m/yr)")
        # Overlay glacier boundaries
        shp.boundary.plot(ax=ax, edgecolor="black", linewidth=1)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")

        # Get actual data date range
        data_start = period_data.select(pl.col("input_dh_start_time").min()).item()
        data_end = period_data.select(pl.col("input_dh_end_time").max()).item()
        data_start_str = str(data_start).split()[0] if " " in str(data_start) else str(data_start)
        data_end_str = str(data_end).split()[0] if " " in str(data_end) else str(data_end)
        ax.set_title(f"dhdt period {period_id}: {data_start_str} to {data_end_str}")

        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()

    def _filter_to_period(data, period_id):
        """
        Filter data to specific period and add x, y coordinates.

        Args:
            data (pl.LazyFrame): Dhdt data with period, dhdt, x_bin, y_bin columns
            period_id (int): Period identifier to filter for

        Returns:
            pl.DataFrame: Filtered data with added x, y coordinate columns
        """
        period_data = data.filter(
            (pl.col("period") == period_id)
            & (pl.col("dhdt") >= args.plot_range[0])
            & (pl.col("dhdt") <= args.plot_range[1])
        ).collect()

        x, y = this_grid_area.get_cellcentre_x_y_from_col_row(
            period_data["x_bin"], period_data["y_bin"]
        )
        return period_data.with_columns([pl.Series("x", x), pl.Series("y", y)])

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
        start_time = period_data.select(pl.col("period_lo").min()).item()
        end_time = period_data.select(pl.col("period_hi").max()).item()
        start_str = str(start_time).split()[0] if " " in str(start_time) else str(start_time)
        end_str = str(end_time).split()[0] if " " in str(end_time) else str(end_time)

        plot_dir = Path(args.out_dir) / sub_basin / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        return plot_dir / f"{sub_basin}_dhdt_period_{period_id}_{start_str}-{end_str}.png"

    for sub_basin in sub_basins_to_process:
        shp, col_name = get_shapefile(this_grid_area, args, logger=None)
        shp = shp[shp[col_name] == sub_basin]
        data = get_data(args, this_grid_area, sub_basin)
        period_ids = data.select(pl.col("period")).unique().collect().to_series().to_list()

        for period_id in period_ids:
            period_data = _filter_to_period(data, period_id)
            if period_data.height == 0:
                continue

            outpath = _get_output_path(sub_basin, period_id, period_data)
            _plot(args, period_data, period_id, shp, outpath)


def plot_icesheet(args: argparse.Namespace, this_area: Area, this_grid_area: GridArea):
    """
    Plot dhdt data for the entire ice data input sheet by period,
    using Polarplot from the Area module.

    Save plots to args.out_dir/plots/dhdt_period_{period_id}_{start_time}-{end_time}.png

    Args:
        args (argparse.Namespace): Command Line Arguments
        this_area (Area): Area object
        this_grid_area (GridArea): GridArea object
    """

    data = get_data(args, this_grid_area, sub_basin=None)
    period_ids = sorted(data.select(pl.col("period")).unique().collect().to_series().to_list())
    outpath = Path(args.out_dir) / "plots"
    outpath.mkdir(parents=True, exist_ok=True)
    for period_id in period_ids:
        period_data = data.filter(pl.col("period") == period_id).collect()

        # Get period boundaries (not actual data times)
        start_time = period_data.select(pl.col("period_lo").min()).item()
        end_time = period_data.select(pl.col("period_hi").max()).item()
        start_time_str = str(start_time).replace(":", "-").replace(" ", "_")
        end_time_str = str(end_time).replace(":", "-").replace(" ", "_")
        plot_dir = Path(args.out_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        Polarplot(this_area.name).plot_points(
            {
                "name": f"dhdt_{period_id}",
                "lats": period_data["latitude"],
                "lons": period_data["longitude"],
                "vals": period_data["dhdt"],
                "valid_range": tuple(args.plot_range),
                "units": "m",
                "cmap_name": "coolwarm",
            },
            output_dir=plot_dir,
            output_file=f"dhdt_period_{period_id}_{start_time_str}-{end_time_str}.png",
        )


def main(args):
    """
    Dhdt plotting script.

    Steps :
    1. Parses command line arguments.
    2. Determines which sub-basins to process.
    3. Loads Area and GridArea objects.
    4. Plots either entire ice sheet or individual sub-basins based on arguments.

    Args:
        args (list): Command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Generate plots from dhdt.parquet files produced by calculate_dhdt.py"
    )

    # Required arguments
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
        "--metadata_json",
        help="Path to the metadata JSON file from calculate_dhdt.",
        required=False,
        default=None,
    )
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
        "--sub_basins",
        nargs="+",
        default=["all"],
        help="Specific list of sub-basins to process."
        "If not provided : Will calculate for entire directory. "
        "If set to 'all' will auto-discover sub-basins."
        "To process root-level data, set to [None].",
    )
    parser.add_argument(
        "--shapefile",
        default="mouginot_glaciers",
        help="Shapefile to use for clipping/plotting glacier boundaries.",
    )
    args = parser.parse_args()

    # Determine which sub-basins to proces
    if args.sub_basins is None or args.sub_basins == ["None"]:
        sub_basins_to_process = ["None"]
    elif args.sub_basins in ("all", ["all"]):
        sub_basins_to_process = get_sub_basins(args.mission_mapper, logger=None)
    else:
        sub_basins_to_process = args.sub_basins

    this_area, this_grid_area = get_objects(args)

    if args.sub_basins is None or args.sub_basins == ["None"]:
        plot_icesheet(args, this_area, this_grid_area)
    else:
        plot_basins(args, this_grid_area, sub_basins_to_process)


if __name__ == "__main__":
    main(sys.argv[1:])

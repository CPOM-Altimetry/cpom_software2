"""
cpom.altimetry.tools.sec_tools.dhdt_plots

Purpose:
    Generate spatial plots of dh/dt (elevation change rate) data from calculate_dhdt.py output.

    Creates scatter plots showing elevation change rates across ice sheet regions or basins,
    with optional glacier boundary overlays from shapefiles.

Output:
    - Basin plots: Saved to <out_dir>/<basin>/plots/<basin>_dhdt_period_{id}_{start}-{end}.png
    - Ice sheet plots: Saved to <out_dir>/plots/dhdt_period_{id}_{start}-{end}.png

Supports:
    - Root-level (entire ice sheet) data
    - Single-tier basin structure (e.g., individual glaciers)
    - Two-tier region/subregion structure (e.g., Antarctic IMBIE2 basins)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import polars as pl
from matplotlib import pyplot as plt

from cpom.altimetry.tools.sec_tools.clip_to_basins import (
    get_basin_geometry_from_shapefile,
    get_shapefile,
)
from cpom.altimetry.tools.sec_tools.cross_calibrate_missions import (
    get_basins_to_process,
)
from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers


def get_objects(args):
    """
    Load Area and GridArea objects for plotting.

    Retrieves grid area and area name from either:
    1. Command line arguments (--grid_area, --binsize, --area_name), or
    2. Metadata JSON file (--metadata_json)

    Args:
        args (argparse.Namespace): Command line arguments with either:
            - grid_area (str), binsize (float), area_name (str), or
            - metadata_json (str): Path to dhdt_metadata.json

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
    Load dh/dt data from parquet file and add geographic coordinates.

    Loads dhdt.parquet from either root directory or basin subdirectory,
    then converts grid cell coordinates (x_bin, y_bin) to latitude/longitude.

    Args:
        args (argparse.Namespace): Command line arguments.
            Includes:
            - in_dir (str): Input directory containing dhdt.parquet file(s)
        this_grid_area (GridArea): Grid area object for coordinate conversion
        sub_basin (str, optional): Basin path (e.g., "basin1" or "West/H-Hp").
            If None or "None", loads from root directory. Defaults to None.

    Returns:
        pl.LazyFrame: Elevation change rate data with added latitude and longitude columns.
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
    Generate spatial scatter plots of dh/dt data for each basin and time period.

    For each basin and period, creates a plot showing:
    - Scatter points colored by dh/dt value
    - Glacier/basin boundaries from shapefile overlay
    - Color scale and axis labels

    Output files: <out_dir>/<basin>/plots/<basin>_dhdt_period_{id}_{start}-{end}.png

    Args:
        args (argparse.Namespace): Command line arguments.
            Includes:
            - in_dir (str): Input directory with dhdt.parquet files
            - out_dir (str): Output directory for plots
            - shapefile (str): Shapefile identifier for boundaries
            - plot_range (list): [min, max] for color scale
        this_grid_area (GridArea): CPOM Grid area object
        sub_basins_to_process (list[str]): Basin paths to process (e.g., ["West/H-Hp", "East/A-Ap"])
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

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")

        # Get actual data date range
        data_start = period_data.select(pl.col("input_dh_start_time").min()).item()
        data_end = period_data.select(pl.col("input_dh_end_time").max()).item()
        data_start_str = str(data_start).split()[0] if " " in str(data_start) else str(data_start)
        data_end_str = str(data_end).split()[0] if " " in str(data_end) else str(data_end)
        ax.set_title(f"dhdt period {period_id}: {data_start_str} to {data_end_str}")

        # Overlay glacier boundaries
        shp.boundary.plot(ax=ax, edgecolor="black", linewidth=1)
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
        return (
            plot_dir
            / f"{sub_basin.replace("/", "_")}_dhdt_period_{period_id}_{start_str}-{end_str}.png"
        )

    # Load shapefile once
    full_shp, selector = get_shapefile(this_grid_area, args, logger=None)

    for sub_basin in sub_basins_to_process:
        # Get geometry for this specific basin using the helper function
        basin_shp = get_basin_geometry_from_shapefile(full_shp, selector, sub_basin)
        data = get_data(args, this_grid_area, sub_basin)
        period_ids = data.select(pl.col("period")).unique().collect().to_series().to_list()

        for period_id in period_ids:
            period_data = _filter_to_period(data, period_id)
            if period_data.height == 0:
                continue

            outpath = _get_output_path(sub_basin, period_id, period_data)
            _plot(args, period_data, period_id, basin_shp, outpath)


def plot_icesheet(args: argparse.Namespace, this_area: Area, this_grid_area: GridArea):
    """
    Generate ice sheet-wide spatial plots of dh/dt data .

    Creates plots for each time period showing elevation change rates across the entire
    ice sheet using the Polarplot class.

    Output files: <out_dir>/plots/dhdt_period_{id}_{start}-{end}.png

    Args:
        args (argparse.Namespace): Command line arguments.
            Includes:
            - in_dir (str): Input directory with dhdt.parquet file
            - out_dir (str): Output directory for plots
            - plot_range (list): [min, max] for color scale (m/yr)
        this_area (Area): CPOM Area object
        this_grid_area (GridArea): CPOM Grid area object
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
            output_dir=str(plot_dir),
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
        "--shapefile",
        default="mouginot_glaciers",
        help="Shapefile to use for clipping/plotting glacier boundaries.",
    )
    parser.add_argument(
        "--structure",
        type=str,
        default="root",
        choices=["root", "single-tier", "two-tier"],
        help="Directory structure: root (icesheet level data), "
        "single-tier (basins), or two-tier (regions/subregions)",
    )
    parser.add_argument(
        "--region_selector",
        nargs="+",
        default=["all"],
        help="Select regions to process. Use 'all' to process all available regions. "
        "Ignored for root level data.",
    )
    parser.add_argument(
        "--subregion_selector",
        nargs="+",
        default=["all"],
        help="For two-tier structure only (e.g., Antarctic IMBIE2 basins): "
        "Select specific subregions within each region (e.g., H-Hp, F-G, A-Ap) "
        "or 'all' for all subregions. "
        "Ignored for single-tier structure.",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logger = set_loggers(
        log_file_info=Path(args.out_dir) / "info.log",
        log_file_error=Path(args.out_dir) / "errors.log",
        log_file_warning=Path(args.out_dir) / "warnings.log",
    )

    this_area, this_grid_area = get_objects(args)

    if args.structure == "root":
        logger.info("Processing root-level data")
        plot_icesheet(args, this_area, this_grid_area)
    else:
        # Determine which sub-basins to process
        sub_basins_to_process = get_basins_to_process(args, args.in_dir, logger=logger)
        plot_basins(args, this_grid_area, sub_basins_to_process)


if __name__ == "__main__":
    main(sys.argv[1:])

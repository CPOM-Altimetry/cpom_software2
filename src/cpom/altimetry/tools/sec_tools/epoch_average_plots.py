"""
cpom.altimetry.tools.sec_tools.epoch_average_plots

Purpose:
    Generate spatial plots of epoch-averaged elevation data from epoch_average.py output.

    Creates scatter plots showing elevation values across ice sheet regions or basins,
    with optional glacier boundary overlays from shapefiles.

Output:
    - Basin plots: Saved to <out_dir>/<basin>/<dh_column>_<basin>_epoch_{id}_{start}_{end}.png
    - Ice sheet plots: Saved to <out_dir>/plots/<dh_column>_epoch_{id}_{start}-{end}.png

Supports:
    - Root-level (entire ice sheet) data
    - Basin-level (individual glaciers or drainage basins)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import polars as pl
from geopandas import gpd
from matplotlib import pyplot as plt

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
    get_basins_to_process,
)
from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers
from cpom.masks.masks import Mask


def parse_arguments(params: list[str]) -> argparse.Namespace:
    """
    Parse command line arguments for epoch average plotting.

    params:
        params: List of command-line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments including mask name,
                           input/output directories, and region selection.
    """
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
        "--parquet_glob",
        type=str,
        default="**/*.parquet",
        help="Glob pattern to match parquet files in input directory.",
    )
    parser.add_argument(
        "--grid_info_json",
        help="Path to the metadata JSON file from calculate_dhdt.",
        required=False,
        default=None,
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
    # Columns to use
    parser.add_argument(
        "--epoch_plotting_column",
        type=str,
        default="epoch_number",
        help="Column name to use for epoch identification in plots (default: epoch_number).",
    )
    parser.add_argument(
        "--plotting_column",
        type=str,
        default="dh_ave",
        help="Column name to use for elevation change values in basin plots (default: dh_ave).",
    )
    parser.add_argument(
        "--dh_column",
        type=str,
        default="dh_ave",
        help="Column name to use for elevation change values in ice sheet plots (default: dh_ave).",
    )
    # Standardize basin selection arguments across tools
    add_basin_selection_arguments(parser)

    # check either mask or

    return parser.parse_args(params)


def get_objects(params):
    """
    Load Area and GridArea objects for plotting.

    Retrieves grid area and area name from either:
    1. Command line arguments (--grid_area, --binsize, --area_name), or
    2. Metadata JSON file (--grid_info_json)

    params:
        params (argparse.Namespace): Command line arguments with either:
            - grid_area (str), binsize (float), area_name (str), or
            - grid_info_json (str): Path to dhdt_metadata.json

    Returns:
        tuple: A tuple containing:
            - Area: The Area object
            - GridArea: The GridArea object
    """

    # Get grid area and area name from params or metadata
    if params.grid_area is not None and params.binsize is not None:
        this_grid_area = GridArea(params.grid_area, params.binsize)
        this_area = Area(params.area_name)
    else:
        with open(Path(params.grid_info_json), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        this_grid_area = GridArea(
            metadata.get("grid_name", "greenland"), metadata.get("binsize", 5000)
        )
        this_area = Area(metadata.get("area", "greenland"))

    return this_area, this_grid_area


def get_data(
    params: argparse.Namespace,
    this_grid_area: GridArea,
    sub_basin: str | None = None,
    logger: logging.Logger | None = None,
) -> pl.LazyFrame:
    """
    Load epoch averaged data from parquet file and add geographic coordinates.

    Loads epoch averaged data from either root directory or basin subdirectory,
    then converts grid cell coordinates (x_bin, y_bin) to latitude/longitude.

    params:
        params (argparse.Namespace): Command line arguments.
            Includes:
            - in_dir (str): Input directory containing dhdt.parquet file(s)
            - parquet_glob (str): Parquet file glob pattern
        this_grid_area (GridArea): Grid area object for coordinate conversion
        sub_basin (str, optional): Basin path to load data from.
            If None or "None", loads from root directory. Defaults to None.

    Returns:
        pl.LazyFrame: data with added 'latitude' and 'longitude' columns.
    """
    if sub_basin is not None and sub_basin != "None":
        logger.info(
            "Loading data for sub-basin: %s from path %s",
            sub_basin,
            Path(params.in_dir) / sub_basin / params.parquet_glob,
        )
        epoch_data = pl.scan_parquet(Path(params.in_dir) / sub_basin / params.parquet_glob)

        row_count = epoch_data.select(pl.len()).collect().item()
        logger.info("Loaded %d rows for sub-basin %s", row_count, sub_basin)
        if row_count == 0:
            logger.info("Skipping sub-basin %s with no data", sub_basin)
            return epoch_data

        # Apply plot range
        epoch_data = epoch_data.filter(
            (pl.col(params.plotting_column) >= params.plot_range[0])
            & (pl.col(params.plotting_column) <= params.plot_range[1])
        )
        # Convert grid coordinates to x/y
        x, y = this_grid_area.get_cellcentre_x_y_from_col_row(
            epoch_data.select(pl.col("x_bin")).collect().to_numpy(),
            epoch_data.select(pl.col("y_bin")).collect().to_numpy(),
        )
        return epoch_data.with_columns([pl.Series("x", x), pl.Series("y", y)])

    logger.info("Loading root-level data from %s", Path(params.in_dir) / params.parquet_glob)
    epoch_data = pl.scan_parquet(Path(params.in_dir) / params.parquet_glob).collect()

    row_count = epoch_data.select(pl.len()).collect().item()
    logger.info("Loaded %d rows for root-level data", row_count)
    if row_count == 0:
        logger.info("No root-level data found; skipping plot")
        return epoch_data
    # Convert grid coordinates to lat/lon
    lats, lons = this_grid_area.get_cellcentre_lat_lon_from_col_row(
        epoch_data.select(pl.col("x_bin")).collect().to_numpy(),
        epoch_data.select(pl.col("y_bin")).collect().to_numpy(),
    )
    return epoch_data.with_columns([pl.Series("latitude", lats), pl.Series("longitude", lons)])


def get_shapefile(params: argparse.Namespace):
    """
    Load the shapefile for basin plots

    params:
        params (argparse.Namespace):
            Includes :
                - shapefile (str): Path to the shapefile
                - shp_file_column (str): Column name in the shapefile to use for basin selection
            or :
                - mask (str): Optional mask name to load predefined shapefile from masks

    Returns:
        gpd.GeoDataFrame: Loaded shapefile GeoDataFrame
    """
    try:
        if params.mask is not None:
            mask = Mask(params.mask)
            shp = gpd.read_file(mask.shapefile_path)
            selector = mask.shapefile_column_name
            # Ensure shapefile is in the same projection as the grid/mask
            if shp.crs is None or shp.crs != mask.crs_bng:
                shp = shp.to_crs(mask.crs_bng)

        shp = gpd.read_file(params.shapefile)
        selector = params.shp_file_column

        return shp, selector

    except (TypeError, ValueError):
        return None, None


def plot_basins(
    params: argparse.Namespace,
    this_grid_area: GridArea,
    sub_basins_to_process: list,
    logger: logging.Logger,
):
    """
    Generate spatial scatter plots of dh/dt data for each basin and time period.

    For each basin and period, creates a plot showing:
    - Scatter points colored by dh/dt value
    - Glacier/basin boundaries from shapefile overlay
    - Color scale and axis labels

    Output files: <out_dir>/<basin>/area_plots/<basin>_dhdt_period_{id}_{start}-{end}.png

    params:
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

    shp, selector = get_shapefile(params)

    for sub_basin in sub_basins_to_process:
        clipped_data = get_data(
            params, this_grid_area, sub_basin=sub_basin, logger=logger
        ).collect()

        if clipped_data.select(pl.len()).item() == 0:
            logger.info("No data points found for basin %s; skipping", sub_basin)
            continue

        plot_timeseries(
            params,
            clipped_data,
            out_path=Path(params.out_dir)
            / sub_basin
            / f"{sub_basin}_{params.dh_column}_epoch_average_timeseries.png",
        )

        # Loop through and plot for each Epoch
        for epoch in sorted(
            clipped_data.select(pl.col(params.epoch_plotting_column)).unique().to_series().to_list()
        ):
            epoch_data = clipped_data.filter(pl.col(params.epoch_plotting_column) == epoch)
            _, ax = plt.subplots(figsize=(10, 8))

            scatter = ax.scatter(
                epoch_data.select(pl.col("x")).to_series().to_numpy(),
                epoch_data.select(pl.col("y")).to_series().to_numpy(),
                c=epoch_data.select(pl.col(params.plotting_column)).to_series().to_numpy(),
                cmap="coolwarm",  # Reverse colormap to match reference colors
                vmin=-1,
                vmax=1,
                s=10,
                alpha=0.7,
            )

            plt.colorbar(scatter, ax=ax, label="dh (m)")

            # Overlay glacier boundaries
            if shp is not None:
                shp[shp[selector] == sub_basin].boundary.plot(ax=ax, edgecolor="black", linewidth=1)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            min_date = epoch_data.select(pl.col("epoch_lo_dt").min().dt.date()).to_series().item()
            max_date = epoch_data.select(pl.col("epoch_hi_dt").max().dt.date()).to_series().item()

            ax.set_title(f"Clipped Data for {sub_basin} - Epoch {epoch} : {min_date} to {max_date}")

            plt.tight_layout()
            plot_path = (
                Path(params.out_dir)
                / sub_basin
                / "area_plots"
                / f"{params.dh_column}_{sub_basin}_epoch_{epoch}_{min_date}_to_{max_date}.png"
            )
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300)
            plt.close()


def plot_icesheet(
    params: argparse.Namespace, this_area: Area, this_grid_area: GridArea, logger: logging.Logger
):
    """
    Generate ice sheet-wide spatial plots of dh/dt data .

    Creates plots for each time period showing elevation change rates across the entire
    ice sheet using the Polarplot class.

    Output files: <out_dir>/plots/dhdt_period_{id}_{start}-{end}.png

    params:
        params (argparse.Namespace): Command line arguments.
            Includes:
            - in_dir (str): Input directory with dhdt.parquet file
            - out_dir (str): Output directory for plots
            - plot_range (list): [min, max] for color scale (m/yr)
        this_area (Area): CPOM Area object
        this_grid_area (GridArea): CPOM Grid area object
        logger (Logger): Logger object
    """

    data = get_data(params, this_grid_area, sub_basin=None, logger=logger)
    row_count = data.select(pl.len()).collect().item()
    if row_count == 0:
        logger.info("No root-level data found; skipping icesheet plots")
        return

    logger.info("Loaded %d rows for icesheet plots", row_count)
    plot_timeseries(
        params,
        data.collect(),
        out_path=Path(params.out_dir) / f"icesheet_{params.dh_column}_epoch_average_timeseries.png",
    )
    epochs = sorted(
        data.select(pl.col(params.epoch_plotting_column)).unique().collect().to_series().to_list()
    )
    logger.info("Found %d epochs for icesheet plotting: %s", len(epochs), epochs)
    for epoch_id in epochs:
        epoch_data = data.filter(pl.col(params.epoch_plotting_column) == epoch_id).collect()

        # Get period boundaries (not actual data times)
        min_date = epoch_data.select(pl.col("epoch_lo_dt").min().dt.date()).to_series().item()
        max_date = epoch_data.select(pl.col("epoch_hi_dt").max().dt.date()).to_series().item()
        plot_dir = Path(params.out_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        Polarplot(this_area.name).plot_points(
            {
                "name": f"dh_{epoch_id}",
                "lats": epoch_data["latitude"],
                "lons": epoch_data["longitude"],
                "vals": epoch_data[params.dh_column],
                "valid_range": tuple(params.plot_range),
                "units": "m",
                "cmap_name": "coolwarm",
            },
            output_dir=str(plot_dir),
            output_file=f"{params.dh_column}_epoch_{epoch_id}_{min_date}-{max_date}.png",
        )


def plot_timeseries(params, data: pl.DataFrame, out_path: Path) -> None:
    """
    Plot the epoch-averaged time series for the selected `dh_column`.
    Filter data based on `plot_range`.
    Computes mean, median, standard error, and count per `epoch_midpoint_dt`,
    then plots the mean series shifted to start at zero.

    Args:
        params (argparse.Namespace): Command line arguments.
        data (pl.DataFrame): Input data with `epoch_midpoint_dt` and `dh_column`.
        out_path (Path | str): Output path for saving the plot.
    """
    out_path = Path(out_path)

    # Aggregate stats
    data_ts = (
        data.filter(
            (pl.col(params.dh_column) >= params.plot_range[0])
            & (pl.col(params.dh_column) <= params.plot_range[1])
        )
        .group_by("epoch_midpoint_dt")
        .agg(
            [
                pl.col(params.dh_column).mean().alias("mean_dh"),
                pl.col(params.dh_column).median().alias("median_dh"),
                pl.col(params.dh_column).std(ddof=1).alias("dh_std"),
                pl.col(params.dh_column).count().alias("count"),
            ]
        )
        .with_columns(
            pl.when(pl.col("count") > 1)
            .then(pl.col("dh_std") / pl.col("count").cast(pl.Float64).sqrt())
            .otherwise(None)
            .alias("se_dh")
        )
        .sort("epoch_midpoint_dt")
    )

    if data_ts.height == 0:
        print("No data available to plot.")
        return

    # Shift time series to start at zero
    data_ts = data_ts.with_columns(
        [
            (pl.col("mean_dh") - data_ts[0, "mean_dh"]).alias("shifted_mean"),
            (pl.col("median_dh") - data_ts[0, "median_dh"]).alias("shifted_median"),
        ]
    )
    _, ax = plt.subplots(figsize=(12, 6))
    # Mean and rolling mean
    ax.plot(
        data_ts["epoch_midpoint_dt"].to_list(),
        data_ts["shifted_mean"].to_list(),
        alpha=0.8,
        linewidth=2,
        color="blue",
        label=f"Mean {params.dh_column} (m)",
    )

    ax.plot(
        data_ts["epoch_midpoint_dt"].to_list(),
        data_ts["shifted_median"].to_list(),
        label=f"Median {params.dh_column} (m)",
        color="red",
        linestyle="--",
        alpha=0.7,
    )

    ax.fill_between(
        data_ts["epoch_midpoint_dt"].to_list(),
        (data_ts["shifted_mean"] - data_ts["se_dh"]).to_list(),
        (data_ts["shifted_mean"] + data_ts["se_dh"]).to_list(),
        alpha=0.3,
        color="blue",
        label="Standard Error",
    )

    ax.set_xlabel("Epoch date")
    ax.set_ylabel("Height change (m)")

    ax.set_title(
        f"Epoch-averaged {params.dh_column} time series",
        fontsize=14,
    )

    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close()


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
        plot_icesheet(params, this_area, this_grid_area, logger=logger)
    else:
        logger.info("Processing basin/sub-basin level data")
        # Determine which sub-basins to process
        sub_basins_to_process = get_basins_to_process(params, params.in_dir, logger=logger)
        plot_basins(params, this_grid_area, sub_basins_to_process, logger=logger)


if __name__ == "__main__":
    main(sys.argv[1:])

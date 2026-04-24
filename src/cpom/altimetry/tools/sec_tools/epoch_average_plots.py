"""
cpom.altimetry.tools.sec_tools.epoch_average_plots

Purpose:
    Generate spatial plots of epoch-averaged elevation data from epoch_average.py output.
    Supports both ice-sheet-wide polar plots and per-basin scatter plots, with optional
    glacier boundary overlays from shapefiles.

    - Ice-sheet-wide: single Polarplot per period (--basin_structure False).
         <out_dir>/<basin>/area_plots/<dh_column>_<basin>_epoch_{id}_{start}_to_{end}.png
    - Basin-level: scatter plot per basin per period (--basin_structure True).
        <out_dir>/plots/<dh_column>_epoch_{id}_{start}_to_{end}.png
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import polars as pl
from matplotlib import pyplot as plt

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
    get_basins_to_process,
)
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    elapsed,
    get_algo_name,
    get_metadata_params,
    write_metadata,
)
from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers
from cpom.masks.masks import Mask


def parse_arguments(params: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for epoch-average plotting."""
    parser = argparse.ArgumentParser(
        description="Generate plots from epoch-averaged parquet files."
    )

    # I/O Arguments
    parser.add_argument(
        "--in_step",
        type=str,
        required=False,
        help="Input algorithm step to source metadata from",
    )
    parser.add_argument("--in_dir", type=str, required=True, help="Input data directory")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory Path",
    )
    parser.add_argument(
        "--parquet_glob",
        type=str,
        default="**/*.parquet",
        help="Glob pattern for selecting input files relative to --in_dir.",
    )
    # Plotting Arguments
    parser.add_argument(
        "--plot_range",
        nargs=2,
        type=float,
        default=[-1.0, 1.0],
        help="Colour scale limits for elevation change values [min max]. Default: -1.0 1.0.",
    )
    parser.add_argument(
        "--shapefile",
        type=str,
        help="Path to a shapefile for glacier boundary overlays.",
    )
    parser.add_argument(
        "--shp_file_column",
        type=str,
        help="Path to a shapefile for glacier boundary overlays.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        help="CPOM Mask class name for predefined shapefile boundaries.",
    )
    # Columns to use
    parser.add_argument(
        "--epoch_plotting_column",
        type=str,
        default="epoch_number",
        help="Column name used to identify epochs in plots.",
    )
    parser.add_argument(
        "--plotting_column",
        type=str,
        default="dh_ave",
        help="Column name for dh values to plot.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging",
    )
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
    # Standardize basin selection arguments across tools
    add_basin_selection_arguments(parser)
    return parser.parse_args(params)


# --------------------------
# Helper functions
# --------------------------
def _load_grid_params(
    params: argparse.Namespace, basin_name: str | None, logger: logging.Logger
) -> dict:
    """Load gridarea and binsize (plus area for root-level runs) from metadata."""
    keys = ["gridarea", "binsize"] if basin_name is not None else ["gridarea", "binsize", "area"]
    return get_metadata_params(
        params, keys, algo_name="grid_for_elev_change", basin_name=basin_name, logger=logger
    )


def _iter_epochs(data: pl.DataFrame, epoch_col: str) -> list:
    """Return a sorted list of unique epoch identifiers from the dataset."""
    return sorted(data.select(pl.col(epoch_col)).unique().to_series().to_list())


def get_data(
    params: argparse.Namespace,
    this_grid_area: GridArea,
    logger: logging.Logger,
    sub_basin: str | None = None,
) -> pl.LazyFrame:
    """
    Load epoch-averaged data and append projected coordinates.

    For basin data: appends projected x/y columns (metres).
    For root data: appends latitude/longitude columns.
    Data is filtered to plot_range before coordinates are appended.

    Args:
        params (argparse.Namespace): Command line arguments.
        this_grid_area (GridArea): Grid area object for coordinate conversion
        sub_basin (str | None): Basin subdirectory to load, or None for root-level data.
    Returns:
        pl.LazyFrame: Input data with coordinate columns appended.
    """
    if sub_basin is not None and sub_basin != "None":
        path = Path(params.in_dir) / sub_basin / params.parquet_glob
        logger.info("Loading basin data: %s", path)
        epoch_data = pl.scan_parquet(path)

        row_count = epoch_data.select(pl.len()).collect().item()
        if row_count == 0:
            logger.info("No data found for sub-basin %s; skipping", sub_basin)
            return epoch_data
        logger.info("Loaded %d rows for sub-basin %s", row_count, sub_basin)

        # Apply plot range
        epoch_data = epoch_data.filter(
            pl.col(params.dh_column).is_between(params.plot_range[0], params.plot_range[1])
        )

        # Convert grid coordinates to x/y
        x, y = this_grid_area.get_cellcentre_x_y_from_col_row(
            epoch_data.select("x_bin").collect().to_numpy(),
            epoch_data.select("y_bin").collect().to_numpy(),
        )
        return epoch_data.with_columns([pl.Series("x", x), pl.Series("y", y)])

    path = Path(params.in_dir) / params.parquet_glob
    logger.info("Loading root-level data from %s", path)
    epoch_data = pl.scan_parquet(path)

    row_count = epoch_data.select(pl.len()).collect().item()
    if row_count == 0:
        logger.info("No root-level data found; skipping plot")
        return epoch_data
    logger.info("Loaded %d rows for root-level data", row_count)

    # Convert grid coordinates to lat/lon
    collected_data = epoch_data.collect()
    lats, lons = this_grid_area.get_cellcentre_lat_lon_from_col_row(
        collected_data.select("x_bin").to_numpy(),
        collected_data.select("y_bin").to_numpy(),
    )
    return epoch_data.with_columns([pl.Series("latitude", lats), pl.Series("longitude", lons)])


def get_shapefile(params: argparse.Namespace, logger: logging.Logger | None = None):
    """
    Load a shapefile for glacier boundary overlays.

    Attempts to load from a CPOM Mask class (--mask) or an explicit path
    (--shapefile + --shp_file_column). Returns (None, None) if neither is
    provided or if loading fails.

    Args:
        params (argparse.Namespace): Command line arguments (uses mask, shapefile, shp_file_column).
        logger (logging.Logger): Logger object

    Returns:
        gpd.GeoDataFrame: (GeoDataFrame, column_name) or (None, None) if unavailable.
    """
    try:
        if params.mask is not None:
            mask = Mask(params.mask)
            shp = gpd.read_file(mask.shapefile_path)
            if shp.crs is None or shp.crs != mask.crs_bng:
                shp = shp.to_crs(mask.crs_bng)
            return shp, mask.shapefile_column_name

        if not params.shapefile:
            if logger is not None:
                logger.info("No shapefile or mask provided; skipping boundary overlay")
            return None, None

        if params.shp_file_column is None:
            if logger is not None:
                logger.warning(
                    "--shp_file_column not provided; boundary filtering will be skipped."
                )

        return gpd.read_file(params.shapefile), params.shp_file_column

    except (TypeError, ValueError, OSError) as exc:
        if logger is not None:
            logger.warning(
                "Could not load shapefile overlay (%s); continuing without boundaries", exc
            )
        return None, None


def plot_timeseries(params, data: pl.DataFrame, out_path: Path) -> None:
    """
    Plot the epoch-averaged elevation change time series.
    Computes mean, median, and standard error per epoch midpoint, shifts each
    series to start at zero, and saves a single figure with all three overlaid.

    Args:
        params (argparse.Namespace): Command line arguments.
        data (pl.DataFrame): Input data with `epoch_midpoint_dt` and `dh_column`.
        out_path (Path): Output path for saving the plot.

    """
    out_path = Path(out_path)

    # Aggregate stats
    data_ts = (
        data.filter(pl.col(params.dh_column).is_between(params.plot_range[0], params.plot_range[1]))
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
        return

    # Shift time series to start at zero
    data_ts = data_ts.with_columns(
        [
            (pl.col("mean_dh") - data_ts[0, "mean_dh"]).alias("shifted_mean"),
            (pl.col("median_dh") - data_ts[0, "median_dh"]).alias("shifted_median"),
        ]
    )
    _, ax = plt.subplots(figsize=(12, 6))
    times = data_ts["epoch_midpoint_dt"].to_list()

    # Mean and rolling mean
    ax.plot(
        times,
        data_ts["shifted_mean"].to_list(),
        alpha=0.8,
        linewidth=2,
        color="blue",
        label=f"Mean {params.dh_column} (m)",
    )

    ax.plot(
        times,
        data_ts["shifted_median"].to_list(),
        label=f"Median {params.dh_column} (m)",
        color="red",
        linestyle="--",
        alpha=0.7,
    )

    ax.fill_between(
        times,
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

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close()


def plot_basins(
    params: argparse.Namespace,
    sub_basins_to_process: list,
    logger: logging.Logger,
) -> dict[str, str]:
    """
    Generate per-epoch scatter plots and a time series for each basin.
    For each basin, produces one scatter plot per epoch and a single time series
        plot. Basins with no data are skipped. An optional glacier boundary overlay
        is applied if a shapefile is available.

    Args:
        params (argparse.Namespace): Command line arguments.
        sub_basins_to_process (list[str]): List of basin subdirectory paths to plot.
        logger: Logger object.

    Returns:
        dict[str, str]: Map of basin name to elapsed processing time string.

    Output:
        <out_dir>/<basin>/area_plots/<dh_column>_<basin>_epoch_{id}_{start}_to_{end}.png
    """

    shp, selector = get_shapefile(params, logger=logger)
    basin_execution_times: dict[str, str] = {}

    for sub_basin in sub_basins_to_process:
        basin_start_time = time.time()

        grid_params = _load_grid_params(params, sub_basin, logger)
        this_grid_area = GridArea(grid_params["gridarea"], grid_params["binsize"])
        clipped_data = get_data(params, this_grid_area, logger, sub_basin).collect()

        if clipped_data.is_empty():
            logger.info("No data for basin %s; skipping.", sub_basin)
            basin_execution_times[sub_basin] = elapsed(basin_start_time)
            continue

        plot_timeseries(
            params,
            clipped_data,
            out_path=Path(params.out_dir)
            / sub_basin
            / f"{sub_basin}_{params.dh_column}_epoch_average_timeseries.png",
        )

        # Loop through and plot for each Epoch
        for epoch in _iter_epochs(clipped_data, params.epoch_plotting_column):
            epoch_data = clipped_data.filter(pl.col(params.epoch_plotting_column) == epoch)
            min_date = epoch_data.select(pl.col("epoch_lo_dt").min().dt.date()).item()
            max_date = epoch_data.select(pl.col("epoch_hi_dt").max().dt.date()).item()
            _, ax = plt.subplots(figsize=(10, 8))

            scatter = ax.scatter(
                epoch_data["x"].to_numpy(),
                epoch_data["y"].to_numpy(),
                c=epoch_data[params.dh_column].to_numpy(),
                cmap="coolwarm",
                vmin=params.plot_range[0],
                vmax=params.plot_range[1],
                s=10,
                alpha=0.7,
            )

            plt.colorbar(scatter, ax=ax, label="dh (m)")

            # Overlay glacier boundaries
            if shp is not None and selector is not None:
                shp[shp[selector] == sub_basin].boundary.plot(ax=ax, edgecolor="black", linewidth=1)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            ax.set_title(f"Clipped Data for {sub_basin} - Epoch {epoch} : {min_date} to {max_date}")
            plt.tight_layout()

            plot_path = (
                Path(params.out_dir)
                / sub_basin
                / f"{params.dh_column}_{sub_basin}_epoch_{epoch}_{min_date}_to_{max_date}.png"
            )
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300)
            plt.close()

            basin_execution_times[sub_basin] = elapsed(basin_start_time)

    return basin_execution_times


def plot_icesheet(params: argparse.Namespace, logger: logging.Logger):
    """
    Generate ice-sheet-wide Polarplots and a time series for each epoch.

    Args:
        params (argparse.Namespace): Command line arguments.
        logger (Logger): Logger object

    Output:
        <out_dir>/plots/<dh_column>_epoch_{id}_{start}-{end}.png
    """

    grid_params = _load_grid_params(params, None, logger)
    this_grid_area = GridArea(grid_params["gridarea"], grid_params["binsize"])
    data_lf = get_data(params, this_grid_area, logger, None)

    if data_lf.select(pl.len()).collect().item() == 0:
        logger.info("No root-level data; skipping ice sheet plots.")
        return

    data = data_lf.collect()
    logger.info("Loaded %d rows for ice sheet plots.", len(data))
    plot_timeseries(
        params,
        data,
        Path(params.out_dir) / f"icesheet_{params.dh_column}_epoch_average_timeseries.png",
    )

    area_name = Area(str(grid_params["area"])).name

    for epoch_id in _iter_epochs(data, params.epoch_plotting_column):
        epoch_data = data.filter(pl.col(params.epoch_plotting_column) == epoch_id)

        # Get period boundaries (not actual data times)
        min_date = epoch_data.select(pl.col("epoch_lo_dt").min().dt.date()).item()
        max_date = epoch_data.select(pl.col("epoch_hi_dt").max().dt.date()).item()

        Polarplot(area_name).plot_points(
            {
                "name": f"dh_{epoch_id}",
                "lats": epoch_data["latitude"],
                "lons": epoch_data["longitude"],
                "vals": epoch_data[params.dh_column],
                "valid_range": tuple(params.plot_range),
                "units": "m",
                "cmap_name": "coolwarm",
            },
            output_dir=str(params.out_dir),
            output_file=f"{params.dh_column}_epoch_{epoch_id}_{min_date}-{max_date}.png",
        )


# -----------------------
# Main Function #
# -----------------------


def epoch_average_plots(params):
    """
    Main entry point for epoch-average plotting.

    Parses arguments, then either:
    - Plots the full ice sheet (--basin_structure False), or
    - Plots each sub-basin individually (--basin_structure True)

    Args:
        args (list[str]): Arguments.
    """

    params = parse_arguments(params)
    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )

    start_time = time.time()
    basin_execution_times: dict[str, str] = {}

    if params.basin_structure is False:
        logger.info("Processing root-level (ice sheet) data.")
        plot_icesheet(params, logger=logger)
        metadata_targets: list[str | None] = [None]
    else:
        logger.info("Processing basin/sub-basin level data")
        sub_basins_to_process = get_basins_to_process(params, params.in_dir, logger=logger)
        basin_execution_times = plot_basins(params, sub_basins_to_process, logger)
        metadata_targets = sub_basins_to_process

    total_time = elapsed(start_time)

    for basin_name in metadata_targets:
        out_meta_dir = (
            Path(params.out_dir) if basin_name is None else Path(params.out_dir) / basin_name
        )
        write_metadata(
            params,
            get_algo_name(__file__),
            out_meta_dir,
            {
                **vars(params),
                "execution_time": (
                    basin_execution_times.get(basin_name, total_time) if basin_name else total_time
                ),
                **({"basin_name": basin_name} if basin_name else {}),
            },
            basin_name=basin_name,
            logger=logger,
        )


if __name__ == "__main__":
    epoch_average_plots(sys.argv[1:])

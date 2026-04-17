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
import logging
import os
import sys
import time
from pathlib import Path
from typing import cast

import polars as pl
from matplotlib import pyplot as plt

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
    get_basins_to_process,
)
from cpom.altimetry.tools.sec_tools.epoch_average_plots import (
    _load_grid_params,
    get_data,
    get_shapefile,
)
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    elapsed,
    get_algo_name,
    write_metadata,
)
from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers


def parse_arguments(args):
    """
    Parse command line arguments for dhdt plotting script.
    Args:
        args (list): Command line arguments
    Returns:
        argparse.Namespace: Parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="Generate plots from dhdt.parquet files produced by calculate_dhdt.py"
    )

    # I/O Arguments
    parser.add_argument(
        "--in_step",
        type=str,
        required=False,
        help="Input algorithm step to source metadata from",
    )
    parser.add_argument(
        "--in_dir",
        required=True,
        help="Path to the directory containing dhdt.parquet file.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Path to the output directory for plots.",
    )
    parser.add_argument(
        "--parquet_glob",
        type=str,
        required=True,
        help="Glob pattern to match parquet files in input directory. e.g. '**/dhdt.parquet'",
    )
    # Plotting Arguments
    parser.add_argument(
        "--area",
        help="Name of the area to plot. If not provided, will be read from metadata JSON.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--gridarea",
        required=False,
        default=None,
        help="Grid area name. If not provided, will be read from metadata JSON.",
    )
    parser.add_argument(
        "--binsize",
        type=float,
        required=False,
        default=None,
        help="Grid bin size. If not provided, will be read from metadata JSON.",
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
        help="Column name in shapefile for basin selection.",
        required=False,
    )
    parser.add_argument(
        "--mask",
        type=str,
        help="CPOM mask name for predefined shapefile boundaries.",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging",
    )
    # Shared basin/region selection arguments
    add_basin_selection_arguments(parser)
    return parser.parse_args(args=args)


def _period_date_strings(period_data: pl.DataFrame) -> tuple[str, str]:
    """Return (start_str, end_str) from actual data times in a period DataFrame."""
    start = period_data.select(pl.col("input_dh_start_time").min()).item()
    end = period_data.select(pl.col("input_dh_end_time").max()).item()
    fmt = lambda dt: str(dt).split()[0]  # noqa: E731
    return fmt(start), fmt(end)


def _basin_plot_path(
    out_dir: str, sub_basin: str, period_id: int, period_data: pl.DataFrame
) -> Path:
    """Build the output path for a basin period plot."""
    start_str, end_str = _period_date_strings(period_data)
    plot_dir = Path(out_dir) / sub_basin
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_basin = sub_basin.replace("/", "_")
    return plot_dir / f"{safe_basin}_dhdt_period_{period_id}_{start_str}-{end_str}.png"


def plot_basins(
    params: argparse.Namespace,
    sub_basins_to_process: list,
    logger: logging.Logger,
) -> dict[str, str]:
    """
    Generate scatter plots of dh/dt data for each basin and time period.

    Output files: <out_dir>/<basin>/plots/<basin>_dhdt_period_{id}_{start}-{end}.png

    Args:
        params (argparse.Namespace): Command line arguments.
            Includes:
            - in_dir (str): Input directory with dhdt.parquet files
            - out_dir (str): Output directory for plots
            - shapefile (str): Shapefile identifier for boundaries
            - plot_range (list): [min, max] for color scale
        this_area (Area): CPOM Area object
        this_grid_area (GridArea): CPOM GridArea object
        sub_basins_to_process (list[str]): Basin paths to process
        logger (logging.Logger): Logger object
    """

    shp, selector = get_shapefile(params, logger)
    basin_execution_times: dict[str, str] = {}

    for sub_basin in sub_basins_to_process:
        basin_start = time.time()
        grid_params = _load_grid_params(params, sub_basin, logger)
        this_grid_area = GridArea(grid_params["gridarea"], grid_params["binsize"])
        basin_lazy = get_data(params, this_grid_area, logger=logger, sub_basin=sub_basin)

        if basin_lazy.select(pl.len()).collect().item() == 0:
            logger.info("No data points found for basin %s; skipping", sub_basin)
            continue

        period_ids = (
            basin_lazy.select(pl.col(params.period_column)).unique().collect().to_series().to_list()
        )

        # Loop through unique periods
        for period_id in period_ids:
            period_data = basin_lazy.filter(pl.col(params.period_column) == period_id).collect()
            if period_data.is_empty():
                continue
            start_str, end_str = _period_date_strings(period_data)

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
            ax.set_title(f"dhdt period {period_id}: {start_str} to {end_str}")

            # Overlay glacier boundaries
            if shp is not None:
                shp[shp[selector] == sub_basin].boundary.plot(ax=ax, edgecolor="black", linewidth=1)

            plt.tight_layout()
            plt.savefig(_basin_plot_path(params.out_dir, sub_basin, period_id, period_data))

            plt.close()
        basin_execution_times[sub_basin] = elapsed(basin_start)

    return basin_execution_times


def plot_icesheet(params: argparse.Namespace, logger: logging.Logger) -> str:
    """
    Generate ice sheet-wide scatter plots of dh/dt for each time period using Polarplot.
    Output files: <out_dir>/plots/dhdt_period_{id}_{start}-{end}.png

    Args:
        params (argparse.Namespace): Command line arguments.
            Includes:
            - in_dir (str): Input directory with dhdt.parquet file
            - out_dir (str): Output directory for plots
            - plot_range (list): [min, max] for color scale (m/yr)
        this_area (Area): CPOM Area object
        this_grid_area (GridArea): CPOM Grid area object
        logger (logging.Logger): Logger object
    """

    fn_start = time.time()
    grid_params = _load_grid_params(params, None, logger)
    this_grid_area = GridArea(grid_params["gridarea"], grid_params["binsize"])
    this_area = Area(str(grid_params["area"]))

    data = get_data(params, this_grid_area, logger, None)
    row_count = data.select(pl.len()).collect().item()
    if row_count == 0:
        logger.info("No root-level data found; skipping icesheet plots")
        return elapsed(fn_start)

    logger.info("Loaded %d rows for icesheet plots", row_count)
    Path(params.out_dir).mkdir(parents=True, exist_ok=True)

    period_ids = sorted(
        data.select(pl.col(params.period_column)).unique().collect().to_series().to_list()
    )
    for period_id in period_ids:
        period_data = data.filter(pl.col(params.period_column) == period_id).collect()

        # Get period boundaries (not actual data times)
        period_start = period_data.select(pl.col("period_lo").min()).item()
        period_end = period_data.select(pl.col("period_hi").max()).item()
        start_str = str(period_start).replace(":", "-").replace(" ", "_")
        end_str = str(period_end).replace(":", "-").replace(" ", "_")

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
            output_file=f"dhdt_period_{period_id}_{start_str}-{end_str}.png",
        )
    return elapsed(fn_start)


# ----------------
# Main Function
# ---------------
def dhdt_plots(args: list[str] | None = None) -> None:
    """
    Main entry point for dh/dt plotting.

    Parses arguments, then either:
    - Plots the full ice sheet (--basin_structure False), or
    - Plots each sub-basin individually (--basin_structure True)

    args:
        args (list): Command line arguments
    """

    params = parse_arguments(args)
    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )

    start_time = time.time()

    if not params.basin_structure:
        logger.info("Processing root-level data")
        icesheet_time = plot_icesheet(params, logger=logger)
        metadata_targets: list[str | None] = [None]
        basin_execution_times: dict[str | None, str] = {None: icesheet_time}
    else:
        logger.info("Processing basin/sub-basin level data")
        sub_basins_to_process = get_basins_to_process(params, params.in_dir, logger=logger)
        basin_execution_times = cast(
            dict[str | None, str],
            plot_basins(params, sub_basins_to_process, logger=logger),
        )
        metadata_targets = cast(list[str | None], sub_basins_to_process)

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
                "execution_time": basin_execution_times.get(basin_name, total_time),
                **({"basin_name": basin_name} if basin_name else {}),
            },
            basin_name=basin_name,
            logger=logger,
        )


if __name__ == "__main__":
    dhdt_plots(sys.argv[1:])

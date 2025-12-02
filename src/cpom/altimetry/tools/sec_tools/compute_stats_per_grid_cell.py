"""
cpom.altimetry.tools.sec_tools.compute_stats_per_gridcell.py

Purpose:
  - Works on either:
     (A) A "compacted" Parquet dataset partitioned by x_part=NN/y_part=MM/data.parquet, or
     (B) A "year-partitioned" dataset (possibly year=YYYY/month=MM/x_part=NN/y_part=MM/*.parquet),
       i.e. multiple files per spatial chunk if they span multiple years/months.

  - Reads each spatial chunk in parallel with duckDB groups by (x_bin, y_bin) to compute both:
       - mean(elevation)
       - coverage_yrs = (max(time) - min(time)) / (secs in 1 year)

- Optionally plots a chosen variable
    (e.g. mean elevation or coverage in years) using cpom.areas.area_plot.Polarplot.

Usage:
    python compute_stats_per_gridcell.py \
        --grid_dir /path/to/dataset \
        --output_file /path/to/results.parquet \
        --plot_to_file /tmp/coverage_plot.png \
        --plot_var coverage_yrs \
"""

import argparse
import json
import logging
import sys
from typing import Dict, Tuple

import duckdb
import polars as pl

from cpom.areas.area_plot import Polarplot
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers

log = logging.getLogger(__name__)
SECONDS_PER_YEAR = 3600.0 * 24.0 * 365.25  # approximate


def get_stats_per_grid_cell(griddir: str, conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """
    Get statistics per grid cell from the parquet files.
    """
    if "compacted" in griddir:
        parquet_glob = f"{griddir}/**/*.parquet"
    else:
        parquet_glob = f"{griddir}/year=*/**/*.parquet"

    tbl = conn.execute(
        f"""
                SELECT
                x_bin,
                y_bin,
                mean(elevation) as mean_elev,
                (MAX(time) - MIN(time)) / {SECONDS_PER_YEAR} AS coverage_yrs,
                count(*) as n_elev,
                x_part,
                y_part,
                FROM parquet_scan('{parquet_glob}')
                group by x_part, y_part, x_bin , y_bin, 
                order by x_bin, y_bin
    """
    ).pl()

    return tbl


def get_grid_and_metadata(griddir: str) -> Tuple[GridArea, Dict]:
    """Get grid area and metadata from the grid directory.

    Args:
        griddir (str): Path to the grid directory.

    Returns:
        Tuple[GridArea, Dict]: A tuple containing the GridArea object and the metadata dictionary.
    """
    with open(griddir + "grid_meta.json", "r", encoding="utf-8") as f:
        grid_meta = json.load(f)

    if "grid_name" in grid_meta and "bin_size" in grid_meta:
        grid = GridArea(grid_meta["grid_name"], int(grid_meta["bin_size"]))
        grid.info()

    return grid, grid_meta


def attach_xy_latlon(df: pl.DataFrame, grid: GridArea) -> pl.DataFrame:
    """
    Attach x_center, y_center, lat_center, lon_center columns to a DataFrame
    that has [x_bin, y_bin].
    """
    if len(df) == 0 or grid is None:
        return df

    df = df.with_columns(
        (pl.col("x_bin") * grid.binsize + grid.minxm + (grid.binsize / 2)).alias("x_centre"),
        (pl.col("y_bin") * grid.binsize + grid.minym + (grid.binsize / 2)).alias("y_centre"),
    )

    lat_arr, lon_arr = grid.transform_x_y_to_lat_lon(
        df["x_centre"].to_numpy(), df["y_centre"].to_numpy()
    )

    df = df.with_columns(pl.Series("lat_centre", lat_arr), pl.Series("lon_centre", lon_arr))

    return df


def plot_variable(
    df: pl.DataFrame,
    grid_meta: dict,
    var_name: str = "mean_elev",
    plot_file: str = "",
):
    """
    Plot a chosen variable (column in df), e.g. "mean_elev" or "coverage_yrs",
    as points using Polarplot.
    """
    if len(df) == 0:
        log.info("No data to plot.")
        return

    if var_name not in df.columns:
        log.error(
            "Requested plot_var '%s' not found in DataFrame columns: %s", var_name, df.columns
        )
        return

    area_name = grid_meta.get("area_filter", "antarctica_is")  # fallback if missing
    dataset_for_plot = {
        "lats": df.select(["lat_centre"]).to_numpy(),
        "lons": df.select(["lon_centre"]).to_numpy(),
        "vals": df.select([var_name]).to_numpy(),
        "name": var_name,
        "plot_size_scale_factor": 0.1,
    }

    log.info("Plotting '%s' for %d grid cells.", var_name, len(df))
    polar = Polarplot(area_name)
    polar.plot_points(dataset_for_plot, output_file=plot_file)

    if plot_file:
        log.info("Saved plot to: %s", plot_file)


def process(args):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean(elevation) and coverage time in each grid cell from a partitioned "
            "Parquet dataset (either 'compacted' or 'year-partitioned'."
            "Optionally plot the chosen variable."
        )
    )

    parser.add_argument(
        "--debug",
        "-d",
        help="Output debug log messages to console",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--grid_dir",
        "-gd",
        help=(
            "Path of the dataset directory containing parquet files. "
            "This can be 'compacted' or 'year-partitioned'."
        ),
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_file",
        "-o",
        help="Optional path to write final aggregated results (e.g. a Parquet of stats).",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--plot_to_file",
        "-pf",
        help="Output plot to this file (e.g. PNG). If omitted, default is no file output.",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--plot_var",
        help=(
            "Which variable to plot: 'mean_elev', 'coverage_yrs', 'n_elev', or other "
            "columns found in final DF."
        ),
        type=str,
        default="mean_elev",
        required=False,
    )

    args = parser.parse_args(args)

    # -----------------------------------------------------------------------------
    #  Set up logging
    # -----------------------------------------------------------------------------
    default_log_level = logging.INFO
    if args.debug:
        default_log_level = logging.DEBUG

    logfile = "/tmp/compute_stats_per_gridcell.log"
    set_loggers(
        log_file_info=logfile[:-3] + "info.log",
        log_file_warning=logfile[:-3] + "warning.log",
        log_file_error=logfile[:-3] + "errors.log",
        log_file_debug=logfile[:-3] + "debug.log",
        log_format="%(levelname)s : %(asctime)s %(name)s : %(message)s",
        default_log_level=default_log_level,
    )

    conn = duckdb.connect("sec_tools.duckdb")

    # 1) Compute stats (mean_elev, coverage_yrs) for each cell in parallel
    tbl = get_stats_per_grid_cell(args.griddir, conn)
    grid_obj, grid_meta = get_grid_and_metadata(args.griddir)

    # 2) (Optional) Write the final DataFrame
    if args.output_file and not len(tbl) == 0:
        tbl.write_parquet(args.output_file, compression="zstd")
        log.info("Wrote aggregated DataFrame to %s", args.output_file)

    # 3) (Optional) Plot using Polarplot
    if len(tbl) == 0:
        log.info("No data, so skipping plot.")
        return

    if grid_obj is not None:
        tbl = attach_xy_latlon(tbl, grid_obj)
        plot_variable(tbl, grid_meta, var_name=args.plot_var, plot_file=args.plot_to_file)
    else:
        log.warning("No grid info (grid_obj) available, cannot do lat/lon. Skipping plot.")


if __name__ == "__main__":
    process(sys.argv[1:])

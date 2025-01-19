#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cpom.altimetry.tools.sec_tools.compute_stats_per_gridcell.py

Purpose:
  - Takes a "compacted" Parquet dataset partitioned by x_part=NN/y_part=MM.
  - Reads each chunk in parallel (multiprocessing), groups by (x_bin, y_bin),
    computes both:
       - mean(elevation)
       - coverage_yrs = (max(time) - min(time)) / (secs in 1 year)
  - Logs progress, merges data, and then plots a chosen variable (either mean elevation
    or coverage in years) via cpom.areas.area_plot.Polarplot.

Usage:
    python compute_stats_per_gridcell.py \
        --grid_dir /path/to/compacted_grid \
        --output_file /path/to/results.parquet \
        --plot_to_file /tmp/coverage_plot.png \
        --plot_var coverage_yrs \
        --max_workers 4
"""

import argparse
import concurrent.futures
import glob
import json
import logging
import os
import sys
import time
from typing import Tuple

import pandas as pd

from cpom.areas.area_plot import Polarplot
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers

log = logging.getLogger(__name__)

SECONDS_PER_YEAR = 3600.0 * 24.0 * 365.25  # approximate


def process_chunk(chunk_info: Tuple[int, int, str]) -> pd.DataFrame:
    """Processes a single chunk of partitioned Parquet data.

    Reads the Parquet file corresponding to a single spatial partition, then groups
    by (x_bin, y_bin) to compute:

    1. mean(elevation)
    2. coverage_yrs = (max(time) - min(time)) / SECONDS_PER_YEAR

    Args:
        chunk_info (Tuple[int, int, str]): A 3-element tuple of the form
            ``(x_part_val, y_part_val, parquet_path)``. The first two integers
            identify the spatial partition indices, and the string is the path
            to the Parquet file.

    Returns:
        pd.DataFrame: A DataFrame with columns:

        - x_bin (int): The x-bin index of the grid cell.
        - y_bin (int): The y-bin index of the grid cell.
        - mean_elev (float): The mean elevation within that cell.
        - coverage_yrs (float): Time coverage in years, computed as
          ``(max(time) - min(time)) / SECONDS_PER_YEAR``.
        - x_part (int): The partition index in x-direction.
        - y_part (int): The partition index in y-direction.

        If the Parquet file is empty, returns an empty DataFrame with the above columns.
    """

    (x_part_val, y_part_val, parquet_path) = chunk_info

    df_chunk = pd.read_parquet(parquet_path)

    if df_chunk.empty:
        return pd.DataFrame(
            columns=["x_bin", "y_bin", "mean_elev", "coverage_yrs", "x_part", "y_part"]
        )

    # Group by (x_bin, y_bin)
    grouped = df_chunk.groupby(["x_bin", "y_bin"], as_index=False)

    # We can compute multiple aggregations in one go using .agg():
    #   mean elevation
    #   min time
    #   max time
    agg_df = grouped.agg(
        mean_elev=("elevation", "mean"),
        time_min=("time", "min"),
        time_max=("time", "max"),
    ).reset_index(drop=True)

    # Convert coverage in seconds to years
    agg_df["coverage_yrs"] = (agg_df["time_max"] - agg_df["time_min"]) / SECONDS_PER_YEAR

    # Keep track of chunk partition (optional)
    agg_df["x_part"] = x_part_val
    agg_df["y_part"] = y_part_val

    return agg_df


def compute_stats_per_cell(data_set: dict, max_workers: int):
    """
    Process a 'compacted' dataset partitioned by:
      x_part=NN/y_part=MM/data.parquet
    using multiprocessing to handle multiple chunks in parallel.

    Returns:
      (final_df, grid_object, grid_meta_dict)

    Where final_df has columns:
      [x_bin, y_bin, mean_elev, coverage_yrs, x_part, y_part]
    """
    start_time = time.time()

    grid_dir = data_set["grid_dir"]
    if not os.path.isdir(grid_dir):
        log.error("grid directory not found or not a directory: %s", grid_dir)
        sys.exit(1)

    meta_json = os.path.join(grid_dir, "grid_meta.json")
    if os.path.exists(meta_json):
        with open(meta_json, "r", encoding="utf-8") as f:
            grid_meta = json.load(f)

    else:
        grid_meta = {}
        log.warning("No grid_meta.json found in %s.", grid_dir)

    # Build a GridArea object if info is available
    grid = None
    if "grid_name" in grid_meta and "bin_size" in grid_meta:
        grid = GridArea(grid_meta["grid_name"], grid_meta["bin_size"])
        grid.info()

    # --------------------------------------------------------------------------
    # 1) Discover all chunk paths
    # --------------------------------------------------------------------------
    all_chunk_paths = []
    x_part_dirs = glob.glob(os.path.join(grid_dir, "x_part=*"))
    if not x_part_dirs:
        log.error("No x_part=* directories found in %s", grid_dir)
        sys.exit(1)

    for x_part_dir in sorted(x_part_dirs):
        if not os.path.isdir(x_part_dir):
            continue

        x_part_name = os.path.basename(x_part_dir)  # e.g. "x_part=2"
        try:
            x_part_val = int(x_part_name.split("=")[1])
        except ValueError:
            log.warning("Skipping invalid x_part directory: %s", x_part_dir)
            continue

        y_part_dirs = glob.glob(os.path.join(x_part_dir, "y_part=*"))
        for y_part_dir in sorted(y_part_dirs):
            if not os.path.isdir(y_part_dir):
                continue

            y_part_name = os.path.basename(y_part_dir)  # e.g. "y_part=3"
            try:
                y_part_val = int(y_part_name.split("=")[1])
            except ValueError:
                log.warning("Skipping invalid y_part directory: %s", y_part_dir)
                continue

            parquet_file = os.path.join(y_part_dir, "data.parquet")
            if os.path.isfile(parquet_file):
                all_chunk_paths.append((x_part_val, y_part_val, parquet_file))

    total_chunks = len(all_chunk_paths)
    if total_chunks == 0:
        log.warning("No parquet data files found in any partition.")
        return pd.DataFrame(), grid, grid_meta

    log.info("Found %d chunk files to process in parallel.", total_chunks)

    # --------------------------------------------------------------------------
    # 2) Process each chunk in parallel using ProcessPoolExecutor
    # --------------------------------------------------------------------------
    results_all = []
    start_parallel = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for i, chunk_info in enumerate(all_chunk_paths, start=1):
            future = executor.submit(process_chunk, chunk_info)
            future_map[future] = (i, chunk_info)

        for future in concurrent.futures.as_completed(future_map):
            i, chunk_info = future_map[future]
            (xp, yp, _) = chunk_info

            try:
                chunk_result = future.result()
            except Exception as e:  # pylint: disable=broad-exception-caught
                log.error("Error processing chunk x_part=%d, y_part=%d: %s", xp, yp, e)
                continue

            # Progress logging
            progress_pct = 100.0 * i / total_chunks
            log.info(
                "Completed chunk %d/%d (%.1f%%): x_part=%d, y_part=%d, rows=%d",
                i,
                total_chunks,
                progress_pct,
                xp,
                yp,
                len(chunk_result),
            )
            if not chunk_result.empty:
                results_all.append(chunk_result)

    log.info("Parallel chunk processing time: %.1f sec", (time.time() - start_parallel))

    if not results_all:
        log.warning("No data from any chunk (all empty).")
        return pd.DataFrame(), grid, grid_meta

    final_df = pd.concat(results_all, ignore_index=True)
    log.info("Concatenated results: %d total rows.", len(final_df))

    elapsed = time.time() - start_time
    log.info("Computation complete in %.1f sec.", elapsed)

    return final_df, grid, grid_meta


def attach_xy_latlon(df: pd.DataFrame, grid: GridArea) -> pd.DataFrame:
    """
    Attach x_center, y_center, lat_center, lon_center columns to a DataFrame
    that has [x_bin, y_bin].
    """
    if df.empty or grid is None:
        return df

    x_center = df["x_bin"] * grid.binsize + grid.minxm + (grid.binsize / 2)
    y_center = df["y_bin"] * grid.binsize + grid.minym + (grid.binsize / 2)

    df["x_center"] = x_center
    df["y_center"] = y_center

    lat_arr, lon_arr = grid.transform_x_y_to_lat_lon(x_center.values, y_center.values)
    df["lat_center"] = lat_arr
    df["lon_center"] = lon_arr

    return df


def plot_variable(
    df: pd.DataFrame,
    grid_meta: dict,
    var_name: str = "mean_elev",
    plot_file: str = "",
):
    """
    Plot a chosen variable (column in df), e.g. "mean_elev" or "coverage_yrs",
    as points using Polarplot.
    """
    if df.empty:
        log.info("No data to plot.")
        return

    if var_name not in df.columns:
        log.error(
            "Requested plot_var '%s' not found in DataFrame columns: %s", var_name, df.columns
        )
        return

    area_name = grid_meta.get("area_filter", "antarctica_is")  # fallback if missing
    dataset_for_plot = {
        "lats": df["lat_center"].values,
        "lons": df["lon_center"].values,
        "vals": df[var_name].values,
        "name": var_name,
        "plot_size_scale_factor": 0.1,
    }

    log.info("Plotting '%s' for %d grid cells.", var_name, len(df))
    polar = Polarplot(area_name)
    polar.plot_points(dataset_for_plot, output_file=plot_file)

    if plot_file:
        log.info("Saved plot to: %s", plot_file)


def main(args):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean(elevation) and coverage time in each grid cell from a 'compacted' "
            "Parquet dataset, using multiprocessing to process chunks in parallel. "
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
            "Path of the 'compacted' grid dir containing parquet files under x_part=NN/y_part=MM."
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
            "Which variable to plot: 'mean_elev' or 'coverage_yrs' or other "
            "columns found in final DF."
        ),
        type=str,
        default="mean_elev",
        required=False,
    )

    parser.add_argument(
        "--max_workers",
        "-mw",
        help="Number of parallel worker processes (default=4).",
        type=int,
        default=4,
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

    data_set = {
        "grid_dir": args.grid_dir,
        "output_file": args.output_file,
        "plot_to_file": args.plot_to_file,
        "plot_var": args.plot_var,
    }

    # 1) Compute stats (mean_elev, coverage_yrs) for each cell in parallel
    df, grid_obj, grid_meta = compute_stats_per_cell(data_set, max_workers=args.max_workers)

    # 2) (Optional) Write the final DataFrame
    if args.output_file and not df.empty:
        df.to_parquet(args.output_file, index=False)
        log.info("Wrote aggregated DataFrame to %s", args.output_file)

    # 3) (Optional) Plot using Polarplot
    if df.empty:
        log.info("No data, so skipping plot.")
        return

    if grid_obj is not None:
        df = attach_xy_latlon(df, grid_obj)
        plot_variable(df, grid_meta, var_name=args.plot_var, plot_file=args.plot_to_file)
    else:
        log.warning("No grid info (grid_obj) available, cannot do lat/lon. Skipping plot.")


if __name__ == "__main__":
    main(sys.argv[1:])

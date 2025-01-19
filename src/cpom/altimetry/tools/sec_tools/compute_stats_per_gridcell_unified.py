#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cpom.altimetry.tools.sec_tools.compute_stats_per_gridcell.py

Purpose:
  - Works on either:
     (A) A "compacted" Parquet dataset partitioned by x_part=NN/y_part=MM/data.parquet, or
     (B) A "year-partitioned" dataset (possibly year=YYYY/month=MM/x_part=NN/y_part=MM/*.parquet),
       i.e. multiple files per spatial chunk if they span multiple years/months.
  - Reads each spatial chunk in parallel (multiprocessing). Each "chunk" is all files that match a
    given (x_part, y_part), merges them in memory, then groups by (x_bin, y_bin) to compute both:
       - mean(elevation)
       - coverage_yrs = (max(time) - min(time)) / (secs in 1 year)
  - Logs progress, merges chunk results, and optionally plots a chosen variable (e.g. mean elevation
    or coverage in years) using cpom.areas.area_plot.Polarplot.

Usage:
    python compute_stats_per_gridcell.py \
        --grid_dir /path/to/dataset \
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
from typing import List, Tuple

import pandas as pd

from cpom.areas.area_plot import Polarplot
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers

log = logging.getLogger(__name__)

SECONDS_PER_YEAR = 3600.0 * 24.0 * 365.25  # approximate


def process_chunk(chunk_info: Tuple[int, int, List[str]]) -> pd.DataFrame:
    """Processes all files for a single spatial chunk (x_part, y_part).

    For 'compacted' datasets, this list typically has 1 Parquet file.
    For 'year-partitioned' datasets, it may contain multiple files
    (e.g., one per year or month) for the same (x_part, y_part).

    We read all files, concatenate them, then group by (x_bin, y_bin) to compute:
      1. mean(elevation)
      2. coverage_yrs = (max(time) - min(time)) / SECONDS_PER_YEAR

    Args:
        chunk_info: A 3-element tuple of the form
            (x_part_val, y_part_val, list_of_paths).
          - x_part_val, y_part_val are integers identifying the spatial partition.
          - list_of_paths is one or more Parquet files for this chunk.

    Returns:
        pd.DataFrame: A DataFrame with columns:
          - x_bin (int): The x-bin index of the grid cell.
          - y_bin (int): The y-bin index of the grid cell.
          - mean_elev (float): The mean elevation within that cell.
          - coverage_yrs (float): Time coverage in years, i.e.
            (max(time) - min(time)) / SECONDS_PER_YEAR.
          - x_part (int): The partition index in x-direction.
          - y_part (int): The partition index in y-direction.
    """
    (x_part_val, y_part_val, file_paths) = chunk_info

    if not file_paths:
        return pd.DataFrame(
            columns=["x_bin", "y_bin", "mean_elev", "coverage_yrs", "x_part", "y_part"]
        )

    # Read and concatenate all Parquet files for this chunk
    df_list = []
    for fpath in file_paths:
        df_part = pd.read_parquet(fpath)
        if not df_part.empty:
            df_list.append(df_part)
    if not df_list:
        return pd.DataFrame(
            columns=["x_bin", "y_bin", "mean_elev", "coverage_yrs", "x_part", "y_part"]
        )
    df_chunk = pd.concat(df_list, ignore_index=True)

    if df_chunk.empty:
        return pd.DataFrame(
            columns=["x_bin", "y_bin", "mean_elev", "coverage_yrs", "x_part", "y_part"]
        )

    # Group by (x_bin, y_bin)
    grouped = df_chunk.groupby(["x_bin", "y_bin"], as_index=False)

    # Compute multiple aggregations in one pass:
    #   mean(elevation)
    #   min(time)
    #   max(time)
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
    Gathers all Parquet files under grid_dir, detects (x_part=NN, y_part=MM) from the
    directory structure, and groups them so that all files for the same chunk
    are combined in one worker call. This works for both:
      - "compacted" layout: one data.parquet per chunk
      - "year-partitioned" layout: possibly multiple files per chunk across different years/months

    Returns:
      (final_df, grid_object, grid_meta_dict)

    Where final_df has columns:
      [x_bin, y_bin, mean_elev, coverage_yrs, x_part, y_part].
    """
    start_time = time.time()

    grid_dir = data_set["grid_dir"]
    if not os.path.isdir(grid_dir):
        log.error("grid directory not found or not a directory: %s", grid_dir)
        sys.exit(1)

    # Attempt to load grid_meta.json if present
    meta_json = os.path.join(grid_dir, "grid_meta.json")
    if os.path.exists(meta_json):
        with open(meta_json, "r", encoding="utf-8") as f_obj:
            grid_meta = json.load(f_obj)
    else:
        grid_meta = {}
        log.warning("No grid_meta.json found in %s.", grid_dir)

    # Build a GridArea object if info is available
    grid = None
    if "grid_name" in grid_meta and "bin_size" in grid_meta:
        grid = GridArea(grid_meta["grid_name"], grid_meta["bin_size"])
        grid.info()

    # --------------------------------------------------------------------------
    # Gather all *.parquet recursively. Then parse out x_part, y_part from the path.
    # We'll do a pattern: **/x_part=*/y_part=*/*.parquet
    # That covers both:
    #   - x_part=NN/y_part=MM/data.parquet  (compacted)
    #   - year=YYYY/month=MM/x_part=NN/y_part=MM/foo.parquet (year-partitioned)
    # --------------------------------------------------------------------------
    parquet_files = glob.glob(
        os.path.join(grid_dir, "**", "x_part=*", "y_part=*", "*.parquet"), recursive=True
    )
    if not parquet_files:
        log.error("No parquet files found under %s with x_part=..., y_part=....", grid_dir)
        sys.exit(1)

    log.info("Found %d total Parquet files matching x_part=.../y_part=...", len(parquet_files))

    # We want a dictionary: (x_part, y_part) -> list of files
    chunk_map: dict = {}
    for fpath in parquet_files:
        # path might look like:
        #   /root/compacted/x_part=1/y_part=2/data.parquet
        # or
        #   /root/year=2015/x_part=3/y_part=10/file.parquet
        # or
        #   /root/year=2015/month=01/x_part=7/y_part=12/file.parquet
        # We'll parse out x_part=NN and y_part=NN from the subdirs.
        # The simplest approach is to walk backwards from the directory names
        # or we can do a small function to find them from the entire path.
        path_parts = fpath.split(os.sep)

        # we look for 'x_part=' and 'y_part='
        x_part_val = None
        y_part_val = None

        for part in path_parts:
            if part.startswith("x_part="):
                try:
                    x_part_val = int(part.split("=")[1])
                except ValueError:
                    pass
            elif part.startswith("y_part="):
                try:
                    y_part_val = int(part.split("=")[1])
                except ValueError:
                    pass

        if x_part_val is None or y_part_val is None:
            # not a valid path
            log.warning("Skipping file that lacks x_part,y_part: %s", fpath)
            continue

        chunk_map.setdefault((x_part_val, y_part_val), []).append(fpath)

    if not chunk_map:
        log.error("No valid x_part=..., y_part=... found in file paths.")
        sys.exit(1)

    # We'll create a list of chunk_info: (x_part, y_part, [list_of_files])
    all_chunk_info = []
    for (xp, yp), fpaths in chunk_map.items():
        all_chunk_info.append((xp, yp, fpaths))

    total_chunks = len(all_chunk_info)
    log.info("Identified %d unique (x_part,y_part) chunks to process.", total_chunks)
    if total_chunks == 0:
        log.warning("No chunk data available; returning empty DataFrame.")
        return pd.DataFrame(), grid, grid_meta

    # --------------------------------------------------------------------------
    # Process in parallel
    # --------------------------------------------------------------------------
    results_all = []
    start_parallel = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for i, chunk_info in enumerate(all_chunk_info, start=1):
            future = executor.submit(process_chunk, chunk_info)
            future_map[future] = (i, chunk_info)

        for future in concurrent.futures.as_completed(future_map):
            i, chunk_info = future_map[future]
            (xp, yp, fpaths) = chunk_info
            try:
                chunk_result = future.result()
            except Exception as err:  # pylint: disable=broad-exception-caught
                log.error("Error processing x_part=%d, y_part=%d: %s", xp, yp, err)
                continue

            progress_pct = 100.0 * i / total_chunks
            log.info(
                "Completed chunk %d/%d (%.1f%%): x_part=%d, y_part=%d, files=%d, rows=%d",
                i,
                total_chunks,
                progress_pct,
                xp,
                yp,
                len(fpaths),
                len(chunk_result),
            )
            if not chunk_result.empty:
                results_all.append(chunk_result)

    log.info("Parallel chunk processing time: %.1f sec", (time.time() - start_parallel))

    if not results_all:
        log.warning("All chunks were empty.")
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
            "Compute mean(elevation) and coverage time in each grid cell from a partitioned "
            "Parquet dataset (either 'compacted' or 'year-partitioned'), using multiprocessing. "
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
            "This can be either 'compacted' (x_part=NN/y_part=MM) or year-partitioned "
            "(year=YYYY/month=MM/x_part=NN/y_part=MM)."
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

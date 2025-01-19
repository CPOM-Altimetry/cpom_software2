#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cpom.altimetry.tools.sec_tools.compute_stats_per_gridcell.py

Purpose:
  - Works on either:
     (A) A "compacted" Parquet dataset partitioned by x_part=NN/y_part=MM/data.parquet, or
     (B) A "year-partitioned" dataset (possibly year=YYYY/month=MM/x_part=NN/y_part=MM/*.parquet),
       i.e. multiple files per spatial chunk if they span multiple years/months.

  - If the dataset includes a 'partition_index.json' (written by grid_altimetry_data.py),
    we use that to discover the (x_part, y_part) pairs instead of a big glob. This can
    dramatically speed up scanning.

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
from typing import Dict, List, Tuple

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


def build_chunk_map_from_partition_index(grid_dir: str) -> Dict[Tuple[int, int], List[str]]:
    """
    If partition_index.json is found in grid_dir, read its (x_part, y_part) pairs
    and build a map of chunk -> list of parquet files by searching only those subpaths.

    Returns dict {(x_part, y_part): [list_of_files]}
    or an empty dict if something fails.
    """
    partition_index_path = os.path.join(grid_dir, "partition_index.json")
    if not os.path.isfile(partition_index_path):
        return {}

    log.info("Reading partition_index.json: %s", partition_index_path)
    with open(partition_index_path, "r", encoding="utf-8") as f_idx:
        partition_data = json.load(f_idx)
    xypart_list = partition_data.get("xypart_list", [])

    if not xypart_list:
        log.warning("partition_index.json is present but empty.")
        return {}

    chunk_map: Dict[Tuple[int, int], List[str]] = {}
    # For each (x_part, y_part), gather relevant parquet files
    # We'll do a small glob for each chunk: "**/x_part=X/y_part=Y/*.parquet"
    # But this is narrower than a full recursion for everything.
    for xp, yp in xypart_list:
        # xp, yp are ints
        chunk_glob = os.path.join(grid_dir, "**", f"x_part={xp}", f"y_part={yp}", "*.parquet")
        files_for_chunk = glob.glob(chunk_glob, recursive=True)
        if files_for_chunk:
            chunk_map.setdefault((xp, yp), []).extend(files_for_chunk)

    return chunk_map


def build_chunk_map_from_glob(grid_dir: str) -> Dict[Tuple[int, int], List[str]]:
    """
    Fallback approach: recursively search for "x_part=NN/y_part=MM/*.parquet"
    under grid_dir, building a dict {(x_part, y_part): list_of_files}.
    """
    parquet_files = glob.glob(
        os.path.join(grid_dir, "**", "x_part=*", "y_part=*", "*.parquet"), recursive=True
    )
    if not parquet_files:
        return {}

    chunk_map: Dict[Tuple[int, int], List[str]] = {}
    for fpath in parquet_files:
        path_parts = fpath.split(os.sep)
        x_part_val, y_part_val = None, None
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
            log.warning("Skipping file that lacks x_part,y_part: %s", fpath)
            continue
        chunk_map.setdefault((x_part_val, y_part_val), []).append(fpath)

    return chunk_map


def compute_stats_per_cell(data_set: dict, max_workers: int):
    """
    Gathers all Parquet files under grid_dir, detects (x_part=NN, y_part=MM) from either:
      - A partition_index.json (if present),
      - Otherwise does a recursive glob.

    Then groups them so that all files for the same chunk
    are combined in one worker call. This works for both:
      - "compacted" layout
      - "year-partitioned" layout

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
        grid = GridArea(grid_meta["grid_name"], int(grid_meta["bin_size"]))
        grid.info()

    # --------------------------------------------------------------------------
    # 1) Attempt to build chunk map from partition_index.json
    # --------------------------------------------------------------------------
    chunk_map = build_chunk_map_from_partition_index(grid_dir)

    if chunk_map:
        log.info(
            "Using partition_index.json. Found %d (x_part,y_part) combos from index.",
            len(chunk_map),
        )
    else:
        # Fallback: do the full glob
        log.info("No valid partition_index.json or it's empty; falling back to glob scan.")
        chunk_map = build_chunk_map_from_glob(grid_dir)

    if not chunk_map:
        log.error("No valid parquet files found (x_part=..., y_part=...). Exiting.")
        sys.exit(1)

    # Build the list for parallel processing
    all_chunk_info = []
    for (xp, yp), file_list in chunk_map.items():
        all_chunk_info.append((xp, yp, file_list))

    total_chunks = len(all_chunk_info)
    log.info("Identified %d unique (x_part,y_part) chunks to process.", total_chunks)
    if total_chunks == 0:
        log.warning("No chunk data available; returning empty DataFrame.")
        return pd.DataFrame(), grid, grid_meta

    # --------------------------------------------------------------------------
    # 2) Process in parallel
    # --------------------------------------------------------------------------
    results_all = []
    start_parallel = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for i, chunk_info in enumerate(all_chunk_info, start=1):
            xp, yp, fpaths = chunk_info
            future = executor.submit(process_chunk, chunk_info)
            future_map[future] = (i, xp, yp, fpaths)

        for future in concurrent.futures.as_completed(future_map):
            i, xp, yp, fpaths = future_map[future]
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
            "Optionally plot the chosen variable. Will use partition_index.json if present "
            "for fast chunk discovery."
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

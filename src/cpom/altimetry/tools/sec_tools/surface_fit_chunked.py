#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surface_fit.py

Purpose:
  - Takes a "compacted" Parquet dataset partitioned by x_part=NN/y_part=MM.
  - Reads each chunk, groups by (x_bin, y_bin), computes mean(elevation).
  - At the end, plots the resulting mean-elevation grid in a polar stereographic map,
    using cpom.areas.area_plot.Polarplot.

Usage:
    python surface_fit.py \
        --grid_dir /path/to/compacted_grid \
        --output_file /path/to/mean_elevations.parquet \
        --plot_to_file /tmp/mean_elev_plot.png

"""

import argparse
import glob
import json
import logging
import os
import sys
import time

import pandas as pd

from cpom.areas.area_plot import Polarplot
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers

log = logging.getLogger(__name__)


def compute_mean_elevation_per_cell(data_set: dict):
    """
    Process a 'compacted' dataset partitioned by:
      x_part=NN/y_part=MM/data.parquet
    to compute mean(elevation) in each actual grid cell (x_bin, y_bin).

    - Loops over each partition subdirectory.
    - Groups by (x_bin, y_bin) in memory.
    - Aggregates to get mean elevation.

    Returns a final pandas DataFrame with columns:
      [x_bin, y_bin, mean_elev, x_part, y_part]
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

    # We'll use grid info if it exists
    grid = None
    if "grid_name" in grid_meta and "bin_size" in grid_meta:
        grid = GridArea(grid_meta["grid_name"], grid_meta["bin_size"])
        grid.info()

    results_all = []  # store chunk-level aggregates

    # Discover all x_part, y_part subdirectories
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
        if not y_part_dirs:
            log.warning("No y_part directories inside %s", x_part_dir)
            continue

        for y_part_dir in sorted(y_part_dirs):
            if not os.path.isdir(y_part_dir):
                continue
            y_part_name = os.path.basename(y_part_dir)  # e.g. "y_part=3"
            try:
                y_part_val = int(y_part_name.split("=")[1])
            except ValueError:
                log.warning("Skipping invalid y_part directory: %s", y_part_dir)
                continue

            parquet_path = os.path.join(y_part_dir, "data.parquet")
            if not os.path.isfile(parquet_path):
                log.info("No data.parquet in %s, skipping", y_part_dir)
                continue

            log.info(
                "Processing chunk: x_part=%d, y_part=%d => %s", x_part_val, y_part_val, parquet_path
            )
            df_chunk = pd.read_parquet(parquet_path)
            if df_chunk.empty:
                log.debug("No rows in %s", parquet_path)
                continue

            # Group by actual grid cell, compute mean of 'elevation'
            grouped = df_chunk.groupby(["x_bin", "y_bin"], as_index=False)
            agg_df = grouped["elevation"].mean(numeric_only=True)
            agg_df.rename(columns={"elevation": "mean_elev"}, inplace=True)

            # Keep track of which chunk these belong to (optional)
            agg_df["x_part"] = x_part_val
            agg_df["y_part"] = y_part_val

            results_all.append(agg_df)

    if not results_all:
        log.warning("No data found in any partition.")
        return pd.DataFrame(columns=["x_bin", "y_bin", "mean_elev", "x_part", "y_part"])

    # Combine the chunk-level results into one final DataFrame
    final_df = pd.concat(results_all, ignore_index=True)
    log.info("Concatenated results: %d total rows.", len(final_df))

    elapsed = time.time() - start_time
    log.info("Mean elevation aggregation complete in %.1f sec.", elapsed)

    return final_df, grid, grid_meta


def attach_xy_latlon(df: pd.DataFrame, grid: GridArea) -> pd.DataFrame:
    """
    Attach x_center, y_center, lat_center, lon_center columns to a DataFrame
    that has [x_bin, y_bin].
    """
    if df.empty:
        return df

    # Compute the center of each bin.
    # e.g. x_center = (x_bin * binsize) + (minxm) + (binsize/2)
    x_center = df["x_bin"] * grid.binsize + grid.minxm + (grid.binsize / 2)
    y_center = df["y_bin"] * grid.binsize + grid.minym + (grid.binsize / 2)
    df["x_center"] = x_center
    df["y_center"] = y_center

    # Transform to lat/lon
    lat_arr, lon_arr = grid.transform_x_y_to_lat_lon(x_center.values, y_center.values)
    df["lat_center"] = lat_arr
    df["lon_center"] = lon_arr

    return df


def plot_mean_elevation(df: pd.DataFrame, grid_meta: dict, plot_file: str | None = None):
    """
    Simple example: plot the mean_elev as point data using Polarplot.
    """

    if df.empty:
        print("No data to plot.")
        return

    area_name = grid_meta.get("area_filter", "antarctica_is")  # fallback if missing
    # Build the dataset dict expected by Polarplot
    dataset_for_plot = {
        "lats": df["lat_center"].values,
        "lons": df["lon_center"].values,
        "vals": df["mean_elev"].values,
        "name": "Mean Elevation (all cells)",
        # Optional: control marker size
        "plot_size_scale_factor": 0.1,
    }

    print(f"Plotting mean elevation for {len(df)} grid cells.")
    # If you have a 'Polarplot' for antarctica, greenland, etc.:
    polar = Polarplot(area_name)

    # plot_points can take an optional output filename
    if plot_file:
        polar.plot_points(dataset_for_plot, output_file=plot_file)
    else:
        polar.plot_points(dataset_for_plot)
    if plot_file:
        print(f"Saved plot to: {plot_file}")


def main(args):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean(elevation) per grid cell from a 'compacted' Parquet dataset, and plot."
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
        help="Optional path to write final aggregated results (e.g. mean elevations).",
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--plot_to_file",
        "-pf",
        help="Output plot to this file (e.g. PNG). If omitted, default is no file output.",
        type=str,
        default=None,
        required=False,
    )

    args = parser.parse_args(args)

    # -----------------------------------------------------------------------------
    #  Set up logging
    # -----------------------------------------------------------------------------
    default_log_level = logging.INFO
    if args.debug:
        default_log_level = logging.DEBUG

    logfile = "/tmp/surface_fit.log"
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
    }

    # 1) Compute mean elevation for each cell, chunk by chunk
    df, grid_obj, grid_meta = compute_mean_elevation_per_cell(data_set)

    # 2) (Optional) Write the final aggregated results
    if args.output_file and not df.empty:
        df.to_parquet(args.output_file, index=False)
        log.info("Wrote aggregated mean_elev DataFrame to %s", args.output_file)

    # 3) (Optional) Plot using Polarplot
    if df.empty:
        log.info("No data, so skipping plot.")
        return

    if grid_obj is not None:
        df = attach_xy_latlon(df, grid_obj)
        # Now we have lat_center, lon_center
        plot_mean_elevation(df, grid_meta, plot_file=args.plot_to_file)
    else:
        log.warning("No grid info (grid_obj) available, cannot do lat/lon. Skipping plot.")


if __name__ == "__main__":
    main(sys.argv[1:])

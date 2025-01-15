#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cpom.altimetry.tools.sec_tools.read_grid_means.py

Purpose
-------
Example tool to read partitioned parquet files produced by the revised
grid_altimetry_data.py (which partitions by [year, x_part, y_part])
and calculate the mean elevation in each 5km grid cell.
Optionally, plot all years, per-year, or a single specified year.
"""

import argparse
import json
import os

import duckdb
import pandas as pd
import yaml

from cpom.areas.area_plot import Polarplot
from cpom.gridding.gridareas import GridArea


def read_mean_elevation_all_years(parquet_path: str) -> pd.DataFrame:
    """
    Query all partitioned parquet files to get a single average (mean) elevation
    per (x_bin, y_bin), combining ALL years.

    Returns a DataFrame with columns: [x_bin, y_bin, mean_elev]
    """
    parquet_glob = f"{parquet_path}/**/*.parquet"

    query = f"""
        SELECT 
            x_bin,
            y_bin,
            AVG(elevation) AS mean_elev
        FROM parquet_scan('{parquet_glob}')
        GROUP BY x_bin, y_bin
        ORDER BY x_bin, y_bin
    """

    con = duckdb.connect()
    df_means = con.execute(query).df()
    con.close()
    return df_means


def read_mean_elevation_by_year(parquet_path: str) -> pd.DataFrame:
    """
    Query all partitioned parquet files to get average (mean) elevation per
    (year, x_bin, y_bin). This shows how to leverage the 'year' column
    from the partitioned dataset.

    Returns a DataFrame with columns: [year, x_bin, y_bin, mean_elev]
    """
    parquet_glob = f"{parquet_path}/**/*.parquet"

    query = f"""
        SELECT 
            year,
            x_bin,
            y_bin,
            AVG(elevation) AS mean_elev
        FROM parquet_scan('{parquet_glob}')
        GROUP BY year, x_bin, y_bin
        ORDER BY year, x_bin, y_bin
    """

    con = duckdb.connect()
    df_means = con.execute(query).df()
    con.close()
    return df_means


def read_mean_elevation_for_single_year(parquet_path: str, year_val: int) -> pd.DataFrame:
    """
    Query all partitioned parquet files to get average (mean) elevation per
    (x_bin, y_bin) *only for the specified year*.

    Returns a DataFrame with columns: [x_bin, y_bin, mean_elev]
    """
    parquet_glob = f"{parquet_path}/**/*.parquet"

    # We'll filter using a WHERE clause for the specified year.

    query = f"""
        SELECT
            x_bin,
            y_bin,
            AVG(elevation) AS mean_elev
        FROM parquet_scan('{parquet_glob}')
        WHERE year = {year_val}
        GROUP BY x_bin, y_bin
        ORDER BY x_bin, y_bin
    """

    con = duckdb.connect()
    df_means = con.execute(query).df()
    con.close()
    return df_means


def read_mean_elevation_for_single_month(
    parquet_path: str, year_val: int, month_val: int
) -> pd.DataFrame:
    """
    Query all partitioned parquet files to get average (mean) elevation per
    (x_bin, y_bin) *only for the specified year AND month*.

    Returns a DataFrame with columns: [x_bin, y_bin, mean_elev].
    """
    parquet_glob = f"{parquet_path}/**/*.parquet"

    query = f"""
        SELECT
            x_bin,
            y_bin,
            AVG(elevation) AS mean_elev
        FROM parquet_scan('{parquet_glob}')
        WHERE year = {year_val}  -- integer comparison, no quotes
          AND month = {month_val}
        GROUP BY x_bin, y_bin
        ORDER BY x_bin, y_bin
    """

    con = duckdb.connect()
    df_means = con.execute(query).df()
    con.close()
    return df_means


def attach_xy_centers(df_means: pd.DataFrame, grid: GridArea) -> pd.DataFrame:
    """
    Given a DataFrame that has (x_bin, y_bin), compute the bin-center coordinates
    (x_center, y_center) based on the grid's minxm/minym and binsize, and attach
    them as new columns.
    """
    df_means["x_center"] = df_means["x_bin"] * grid.binsize + grid.minxm + (grid.binsize / 2)
    df_means["y_center"] = df_means["y_bin"] * grid.binsize + grid.minym + (grid.binsize / 2)
    return df_means


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Read partitioned Parquet (potentially partitioned by year, month, x_part, y_part) "
            "and calculate the mean elevation per grid cell."
        )
    )

    parser.add_argument(
        "--grid_dir",
        "-gd",
        help=(
            "Path of grid directory to load grid and meta-data from."
            "Used instead of rcf name. No need for both as meta-data loaded automatically"
        ),
        required=False,
    )

    parser.add_argument(
        "--rcf_filename",
        "-r",
        help=("Path of run control file (YAML). Use instead of --grid_dir"),
        required=False,
    )
    parser.add_argument(
        "--by_year",
        help="If set, compute per-year means for all years (group by year).",
        action="store_true",
    )
    parser.add_argument(
        "--year",
        "-y",
        help="If specified, compute mean only for this single year (e.g. 2015).",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--month",
        "-m",
        help=(
            "If specified, compute mean only for this single month of year (e.g. 01/2015)."
            "Use with --year"
        ),
        required=False,
        type=int,
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------------------------------------
    # 1) Load configuration (optionally from a YAML run-control file).
    #    If not provided, we fall back to a hard-coded default dictionary.
    # ----------------------------------------------------------------------------------------------
    if args.grid_dir:
        with open(os.path.join(args.grid_dir, "grid_meta.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

    elif args.rcf_filename:
        with open(args.rcf_filename, "r", encoding="utf-8") as f:
            raw_yaml = f.read()

        # Substitute environment variables if present in placeholders like ${MY_VAR}
        for key, value in os.environ.items():
            placeholder = f"${{{key}}}"
            raw_yaml = raw_yaml.replace(placeholder, value)

        config = yaml.safe_load(raw_yaml)
    else:
        # Hard-coded fallback configuration
        config = {
            "grid_name": "greenland",
            "bin_size": 5000,
            "area_filter": "greenland_is",
            "grid_output_dir": "/cpnet/altimetry/landice/gridded_altimetry",
            "mission": "cs2",
            "dataset": "cryotempo_c001",
        }

    # ----------------------------------------------------------------------------------------------
    # 2) Construct the path to your partitioned dataset
    # ----------------------------------------------------------------------------------------------
    grid = GridArea(config["grid_name"], int(config["bin_size"]))
    parquet_path = os.path.join(
        config["grid_output_dir"],
        config["mission"],
        f'{config["grid_name"]}_{int(config["bin_size"] / 1000)}km_{config["dataset"]}',
    )

    print(f"Reading dataset from: {parquet_path}")

    # ----------------------------------------------------------------------------------------------
    # 3) Decide which read function to call:
    #    - If --year is specified, read only that year.
    #    - Else if --by_year, group by year.
    #    - Else read across all years (combined).
    # ----------------------------------------------------------------------------------------------
    if args.year and args.month:
        print(f"Computing means for single month: {args.month:02d}/{args.year}")
        df_means = read_mean_elevation_for_single_month(parquet_path, args.year, args.month)
        plot_name = f"{args.month:02d}/{args.year}"
    elif args.year:
        print(f"Computing means for single year: {args.year}")
        df_means = read_mean_elevation_for_single_year(parquet_path, args.year)
        plot_name = f"{args.year}"
    elif args.by_year:
        print("Computing means PER YEAR (all years) ...")
        df_means = read_mean_elevation_by_year(parquet_path)
        plot_name = "by year"
    else:
        print("Computing means ACROSS ALL YEARS (combined)...")
        df_means = read_mean_elevation_all_years(parquet_path)
        plot_name = "all years"

    print(f"Number of rows in the aggregated DataFrame: {len(df_means)}")

    # ----------------------------------------------------------------------------------------------
    # 4) Attach x_center and y_center coordinates
    # ----------------------------------------------------------------------------------------------
    df_means = attach_xy_centers(df_means, grid)

    # If you're reading a single year, there's no 'year' column in your query.
    # If you're reading per-year or all-years, you may or may not have 'year'.
    # For the single-year query, we skip year in the SELECT.
    # If you prefer, you can also SELECT 'year' in that query.

    # ----------------------------------------------------------------------------------------------
    # 5) Convert (x_center, y_center) -> lat, lon for each bin center, then plot
    # ----------------------------------------------------------------------------------------------
    lat_arr, lon_arr = grid.transform_x_y_to_lat_lon(
        df_means["x_center"].values, df_means["y_center"].values
    )
    df_means["lat_center"] = lat_arr
    df_means["lon_center"] = lon_arr

    # ----------------------------------------------------------------------------------------------
    # 6) Example: Plot the data as points using Polarplot
    # ----------------------------------------------------------------------------------------------

    if args.by_year:
        # df_means has columns: ['year', 'x_bin', 'y_bin', 'mean_elev', 'x_center', 'y_center']
        # and maybe lat_center/lon_center if you already attached them
        for year_val, subdf in df_means.groupby("year"):
            sub_ds = {
                "lats": subdf["lat_center"].values,
                "lons": subdf["lon_center"].values,
                "vals": subdf["mean_elev"].values,
                "name": f"{year_val}",
            }
            out_path = f"/tmp/mean_plot_{year_val}.png"
            print(f"Plotting year {year_val} to {out_path}")
            Polarplot(config["area_filter"]).plot_points(
                sub_ds, output_dir="/tmp", output_file=f"mean_plot_{year_val}.png"  # or wherever
            )
    else:
        # do the single combined plot as before
        dataset_for_plot = {
            "lats": df_means["lat_center"].values,
            "lons": df_means["lon_center"].values,
            "vals": df_means["mean_elev"].values,
            "name": plot_name,
        }
        Polarplot(config["area_filter"]).plot_points(
            dataset_for_plot,
            # output_dir="/tmp"
        )


if __name__ == "__main__":
    main()

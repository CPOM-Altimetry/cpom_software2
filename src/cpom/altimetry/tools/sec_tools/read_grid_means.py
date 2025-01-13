"""cpom.altimetry.tools.sec_tools.read_grid_means.py

# Purpose

Example tool to read partitioned parquet files produced by grid_altimetry_data.py
and calculate the mean elevation in each grid cell over the example Greenland grid.

"""

import argparse
import os

import duckdb
import pandas as pd
import yaml

from cpom.areas.area_plot import Polarplot
from cpom.gridding.gridareas import GridArea


def read_mean_elevation(parquet_path: str) -> pd.DataFrame:
    """
    Query all partitioned parquet files to get average (mean) elevation
    per (x_bin, y_bin). Returns a DataFrame with columns:
       x_bin, y_bin, mean_elev
    """
    parquet_glob = f"{parquet_path}/**/*.parquet"

    query = f"""
        SELECT 
            x_bin,
            y_bin,
            AVG(elevation) AS mean_elev
        FROM parquet_scan('{parquet_glob}')
        GROUP BY x_bin, y_bin
    """
    con = duckdb.connect()
    df_means = con.execute(query).df()
    con.close()
    return df_means


def attach_xy_centers(df_means: pd.DataFrame, grid: GridArea) -> pd.DataFrame:
    """
    Given a DataFrame with (x_bin, y_bin), compute the bin-center coordinates
    (x_center, y_center) based on the grid's minxm/minym and binsize.
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
            "Convert altimetry data into partitioned Parquet with ragged layout "
            "and coarser partitioning."
        )
    )

    parser.add_argument("--rcf_filename", "-r", help="path of run control file", required=True)

    args = parser.parse_args()

    # ----------------------------------------------------------------------------------------------
    # Read run control file (rcf)
    # ----------------------------------------------------------------------------------------------
    # Read the raw YAML file as a string
    with open(args.rcf_filename, "r", encoding="utf-8") as f:
        raw_yaml = f.read()

    # 2) Substitute environment variables in the string
    #   we denote env vars with a simple syntax like: ${MY_VAR}
    for key, value in os.environ.items():
        placeholder = f"${{{key}}}"
        raw_yaml = raw_yaml.replace(placeholder, value)

    # 3) Now load the substituted YAML string
    config = yaml.safe_load(raw_yaml)

    # Create (or load) your GridArea
    grid = GridArea(config["grid_name"], int(config["bin_size"]))

    parquet_path = os.path.join(
        config["grid_output_dir"],
        config["mission"],
        f'{config["grid_name"]}_{int(config["bin_size"]/1000)}km_{config["dataset"]}',
    )

    # 1) Read mean elevation per bin from your partitioned parquet
    df_means = read_mean_elevation(parquet_path)

    # 2) Attach x_center and y_center columns
    df_means = attach_xy_centers(df_means, grid)

    # 3) Convert x_center,y_center -> lat,lon for each bin
    lat_arr, lon_arr = grid.transform_x_y_to_lat_lon(
        df_means["x_center"].values, df_means["y_center"].values
    )

    # 4) Prepare data for plotting as individual points
    dataset = {"lats": lat_arr, "lons": lon_arr, "vals": df_means["mean_elev"].values}

    # 5) Plot the data (all points) with Polarplot
    #    Note: This works as long as 'lats', 'lons', and 'vals' all have the same shape (1D).
    Polarplot(config["area_filter"]).plot_points(dataset)


if __name__ == "__main__":
    main()

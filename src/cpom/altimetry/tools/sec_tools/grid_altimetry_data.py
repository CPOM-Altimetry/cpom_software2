#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cpom.altimetry.tools.sec_tools.grid_altimetry_data.py

# Purpose

Grid altimetry data for SEC (model fit) processing,
but store each measurement (ragged arrays) in partitioned Parquet
approach (by x_part, y_part = x_bin//20, y_bin//20).

# Inputs

# Outputs

Parquet directory containing partitioned Parquet files

# Usage

Example: gridding CryoTEMPO Baseline C001 over the Greenland ice-sheet
```
  python grid_altimetry_data.py --gridarea greenland \
    -o test2.parquet --max_files 100  --regrid --binsize 5e3 --area greenland_is \
    -elev elevation -lat latitude -lon longitude -power backscatter \
    -time time --l2_dir /cpdata/SATS/RA/CRY/Cryo-TEMPO/BASELINE-C/001/LAND_ICE \
    -pat **/CS_OFFL_SIR_TDP_LI*C001*.nc --unique_string -32 -3
```

# TODO

- Split into writing data frames every n rows
- Check time parameter
- Check x,y params - should they be diffs
- Check direction param
- add meta data to grid directory
- separate logging for each stage? 

"""

import argparse
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from netCDF4 import Dataset, Variable  # pylint: disable=E0611

from cpom.areas.areas import Area

# CPOM imports
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers

log = logging.getLogger(__name__)


def load_processed_files(output_dir: str) -> set:
    """
    Load the set of already processed unique_str's from a JSON file.
    Returns an empty set if no file found.

    Args:
        output_dir (str): path of top-level parquet directory
    Returns:
         (set) : set containing previously processed unique strings or empty set
    """
    processed_file_path = os.path.join(output_dir, "processed_files.json")
    if os.path.exists(processed_file_path):
        with open(processed_file_path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_processed_files(output_dir: str, processed_files: set):
    """
    Save the set of processed unique_str's to a JSON file.

    Args:
        output_dir (str): path of top-level parquet directory
        processed_files (set): containing processed unique strings
    Returns:
         None
    """
    processed_file_path = os.path.join(output_dir, "processed_files.json")

    log.info("saving processed file list to: %s", processed_file_path)
    with open(processed_file_path, "w", encoding="utf-8") as f:
        json.dump(list(processed_files), f, indent=2)


def get_variable(nc: Dataset, nc_var_path: str) -> Variable:
    """
    Retrieve variable from NetCDF file, handling groups if necessary.

    Args:
        nc (Dataset) : netcdf Dataset object
        nc_var_path (str) : netcdf variable name. If netcdf contains groups use /mygroup/param_name
    Returns:
        Dataset Variable (Variable) : the selected Dataset Variable
    """
    parts = nc_var_path.split("/")
    var = nc
    for part in parts:
        var = var[part]
    return var


def grid_dataset(data_set: dict, regrid: bool = False):
    """
    Read NetCDF altimetry data, bin them into (x_bin, y_bin) grid cells,
    and store the resulting table in partitioned Parquet format with
    coarser partitioning (x_part, y_part) = (x_bin//20, y_bin//20).
    """

    # Form path of grid parquet directory as
    # <grid_output_dir>/<mission>/<grid_name>_<binsize/1000>km_<dataset>

    output_dir = os.path.join(
        data_set["grid_output_dir"],
        data_set["mission"],
        f'{data_set["grid_name"]}_{int(data_set["bin_size"]/1000)}km_{data_set["dataset"]}',
    )

    # Load the set of processed file identifiers
    processed_files = load_processed_files(output_dir)
    if regrid:
        processed_files = set()

    # Prepare the grid
    grid = GridArea(data_set["grid_name"], data_set["bin_size"])
    thisarea = Area(data_set["area_filter"])

    # We'll gather data in a list of DataFrames (one per file).
    # If you have memory concerns with large data, you can write in batches.
    all_rows = []

    # Find matching files
    search_dir = data_set["l2_directory"]
    pattern = data_set["search_pattern"]
    matching_files = glob.glob(f"{search_dir}/{pattern}", recursive=True)

    for n_file, file_path in enumerate(matching_files):
        log.info("%d : reading %s", n_file, file_path)

        if data_set["max_files"] is not None and n_file > data_set["max_files"]:
            break

        unique_str = file_path[data_set["unique_string"][0] : data_set["unique_string"][1]]
        log.debug("unique_str=%s", unique_str)

        # --- Check if already processed ---
        if unique_str in processed_files:
            log.info("Skipping file %s (unique_str=%s) - already processed", file_path, unique_str)
            continue

        with Dataset(file_path) as nc:
            # Extract lat/lon
            try:
                lats = get_variable(nc, data_set["latitude_param"])[:].data
                lons = get_variable(nc, data_set["longitude_param"])[:].data % 360.0
            except (KeyError, IndexError):
                log.error(
                    "No latitude or longitude parameter %s found in %s",
                    data_set["latitude_param"],
                    file_path,
                )
                continue

            # Convert lat/lon -> (x, y)
            x_coords, y_coords = thisarea.latlon_to_xy(lats, lons)

            # Area filter
            bool_mask, n_inside = thisarea.inside_area(lats, lons)
            if n_inside == 0:
                log.debug("No points inside area: %s", thisarea.name)
                continue

            # Elevation
            try:
                elevations = get_variable(nc, data_set["elevation_param"])[:].data
            except (IndexError, KeyError):
                log.error(
                    "No elevation parameter %s found in %s", data_set["elevation_param"], file_path
                )
                continue

            try:
                fill_value = get_variable(  # pylint: disable=protected-access
                    nc, data_set["elevation_param"]
                )._FillValue
                elevations[elevations == fill_value] = np.nan
            except AttributeError:
                log.debug("No _FillValue attribute found for elevation param")

            # Power
            try:
                power = get_variable(nc, data_set["power_param"])[:].data
            except (IndexError, KeyError):
                log.error("No power parameter %s found in %s", data_set["power_param"], file_path)
                continue

            try:
                fill_value = get_variable(  # pylint: disable=protected-access
                    nc, data_set["power_param"]
                )._FillValue
                power[power == fill_value] = np.nan
            except AttributeError:
                log.debug("No _FillValue attribute found for power param")

            # Combine mask with valid data
            bool_mask = bool_mask & np.isfinite(elevations) & np.isfinite(power)
            if bool_mask.sum() == 0:
                log.warning("No valid measurements after elevation & power ingest: %s", file_path)
                continue

            # Subset
            x_valid = x_coords[bool_mask]
            y_valid = y_coords[bool_mask]
            elev_valid = elevations[bool_mask]
            pwr_valid = power[bool_mask]

            # Convert to grid bins
            x_bin = ((x_valid - grid.minxm) / grid.binsize).astype(int)
            y_bin = ((y_valid - grid.minym) / grid.binsize).astype(int)

            # Read ascending info (example logic, adapt as needed)
            ascending_locs = np.ones_like(x_valid, dtype=bool)
            try:
                if nc.ascending_start_record == "None" and nc.descending_start_record != "None":
                    ascending_locs[:] = False
                elif nc.descending_start_record == "None" and nc.ascending_start_record != "None":
                    ascending_locs[:] = True
                else:
                    if nc.ascending_start_record == 0 and nc.descending_start_record > 0:
                        ascending_locs[nc.descending_start_record :] = False
                    elif nc.descending_start_record == 0 and nc.ascending_start_record > 0:
                        ascending_locs[: nc.ascending_start_record] = False
            except AttributeError:
                log.error("Error reading ascending_start_record in %s", file_path)
                continue

            # Possibly read time or other parameters
            try:
                time_data = get_variable(nc, data_set["time_param"])[:].data
                time_valid = time_data[bool_mask]
            except (KeyError, IndexError):
                log.error("No time_param found or error reading time in %s", file_path)
                continue

            # Build DataFrame for this file
            df = pd.DataFrame(
                {
                    "x_bin": x_bin,
                    "y_bin": y_bin,
                    "xi": x_valid,
                    "yi": y_valid,
                    "elevation": elev_valid,
                    "power": pwr_valid,
                    "ascending": ascending_locs,
                    "time": time_valid,
                }
            )

            # -----------------------------
            # Coarser partitioning (factor=20)
            #   x_part = x_bin // 20
            #   y_part = y_bin // 20
            # You can pick a factor that suits your data size vs. number of partitions.
            # -----------------------------
            df["x_part"] = df["x_bin"] // 20
            df["y_part"] = df["y_bin"] // 20

            all_rows.append(df)

            processed_files.add(unique_str)

    # Concatenate all data into a single DataFrame
    if len(all_rows) == 0:
        log.info("No data found. Exiting.")
        return

    log.info("Number of rows %d", len(all_rows))

    final_df = pd.concat(all_rows, ignore_index=True)

    # Convert to an Arrow table
    arrow_table = pa.Table.from_pandas(final_df)

    # append behaviour
    if regrid:
        existing_data_behavior_str = "delete_matching"
    else:
        existing_data_behavior_str = "overwrite_or_ignore"

    # Write to a partitioned dataset.
    # We'll partition by (x_part, y_part),
    # so that each partition includes many (x_bin, y_bin) pairs.
    pq.write_to_dataset(
        table=arrow_table,
        root_path=output_dir,
        partition_cols=["x_part", "y_part"],  # coarser partitioning
        use_dictionary=True,
        compression="snappy",  # or "zstd", "gzip", etc.
        existing_data_behavior=existing_data_behavior_str,  # "delete_matching",
        # "overwrite_or_ignore"
    )

    # Save the updated set of processed files
    save_processed_files(output_dir, processed_files)

    log.info("Partitioned Parquet dataset written to: %s", output_dir)
    log.debug("  * Table shape: %s rows, %s columns", *final_df.shape)
    log.debug("  * Partition columns: x_part, y_part (20x coarser than x_bin, y_bin)")


def main(args):
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert altimetry data into partitioned Parquet with ragged layout "
            "and coarser partitioning."
        )
    )

    parser.add_argument(
        "--area",
        "-a",
        help=(
            "area for data filtering/masking of input measurements. "
            "Valid area names in cpom.areas.definition.<area_name>.py. Example: greenland_is"
        ),
        required=True,
    )

    parser.add_argument(
        "--binsize",
        "-b",
        help=("grid binsize in m, default=5e3 == 5km"),
        required=True,
        type=float,
    )

    parser.add_argument(
        "--debug",
        "-d",
        help="Output debug log messages to console",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--elevation_param",
        "-elev",
        help=(
            "name of elevation parameter to use in L2 netcdf files. If netcdf contains groups, "
            "use / to seperate. example data/ku/elevation"
        ),
        required=True,
    )

    parser.add_argument(
        "--gridarea",
        "-g",
        help=(
            "output grid area specification name. Valid names in "
            "cpom.gridding.gridarea.py:all_grid_areas"
        ),
        required=True,
    )

    parser.add_argument(
        "--l2_dir",
        "-dir",
        help="L2 base directory to recursively search for L2 files using the search pattern: "
        "example for CryoTEMPO: /cpdata/SATS/RA/CRY/Cryo-TEMPO/BASELINE-C/001/LAND_ICE",
        required=True,
    )

    # 'l2_directory': '/cpdata/SATS/RA/CRY/Cryo-TEMPO/BASELINE-C/001/LAND_ICE',

    parser.add_argument(
        "--lat_param",
        "-lat",
        help=(
            "name of latitude parameter to use in L2 netcdf files. If netcdf contains groups use / "
            "to seperate. example data/ku/latitude"
        ),
        required=True,
    )

    parser.add_argument(
        "--lon_param",
        "-lon",
        help=(
            "name of longitude parameter to use in L2 netcdf files. If netcdf contains groups use /"
            "to seperate. example data/ku/longitude"
        ),
        required=True,
    )

    parser.add_argument(
        "--max_files",
        "-mf",
        help="Restrict number of input L2 files (int)",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Path of output partitioned Parquet dataset",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--power_param",
        "-power",
        help=(
            "name of power (backscatter) parameter to use in L2 netcdf files. If netcdf contains "
            "groups, use / to seperate. example data/ku/backscatter"
        ),
        required=True,
    )

    parser.add_argument("--regrid", "-r", help="[optional]", required=False, action="store_true")

    parser.add_argument(
        "--search_pattern",
        "-pat",
        help=(
            "search pattern for L2 file discovery. Example for CryoTEMPO C001: "
            "**/CS_OFFL_SIR_TDP_LI*C001*.nc"
        ),
        required=True,
    )

    #'search_pattern': '**/CS_OFFL_SIR_TDP_LI*C001*.nc',

    # add long and short argument
    parser.add_argument(
        "--scenario",
        "-s",
        default="operational",
        help="id name used to name the grid processing chain. Default is : operational",
    )

    parser.add_argument(
        "--time_param",
        "-time",
        help=(
            "name of time parameter to use in L2 netcdf files. If netcdf contains "
            "groups, use / to seperate. example data/ku/time"
        ),
        required=True,
    )

    parser.add_argument(
        "--unique_string",
        "-us",
        help=("offsets from end of file name to provide a unique string. example: -32 -3"),
        required=True,
        nargs=2,
        type=int,
    )

    args = parser.parse_args(args)

    # ----------------------------------------------------------------------------------------------
    # Create a logger for this tool to output to console
    # ----------------------------------------------------------------------------------------------

    default_log_level = logging.INFO
    if args.debug:
        default_log_level = logging.DEBUG
    logfile = "/tmp/grid.log"
    set_loggers(
        log_file_info=logfile[:-3] + "info.log",
        log_file_warning=logfile[:-3] + "warning.log",
        log_file_error=logfile[:-3] + "errors.log",
        log_file_debug=logfile[:-3] + "debug.log",
        log_format="%(levelname)s : %(asctime)s %(name)s : %(message)s",
        default_log_level=default_log_level,
    )

    # Example data_set dictionary
    data_set = {
        "grid_name": args.gridarea,  # GridArea name: is 'greenland'
        "area_filter": args.area,  # CPOM area name, used for masking input measurements
        "bin_size": args.binsize,  # grid binsize to use, example: 5e3 (for 5km grid)
        "l2_directory": args.l2_dir,  # L2 base dir to recursively search for L2 files,
        "search_pattern": args.search_pattern,  # search pattern for L2 file discovery
        "unique_string": args.unique_string,  # offsets from end of fname to
        # create a unique string per L2 file, eg -32 -3
        "latitude_param": args.lat_param,  # latitude name in netcdf. example: data/ku/latitude
        "longitude_param": args.lon_param,  # longitude name in netcdf. example: data/ku/longitude
        "elevation_param": args.elevation_param,  # elevation parameter in netcdf. use / for groups
        "power_param": args.power_param,  # power (backscatter) parameter in netcdf. / for groups
        "time_param": args.time_param,  # time parameter in netcdf. / for groups
        "max_files": args.max_files,  # limit to this number of input files ingested
        "grid_output_dir": args.output_dir,  # path of grid output dir to be created or updated
    }

    grid_dataset(data_set, regrid=args.regrid)


if __name__ == "__main__":
    main(sys.argv[1:])

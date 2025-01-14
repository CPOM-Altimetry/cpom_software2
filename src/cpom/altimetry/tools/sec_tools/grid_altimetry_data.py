#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cpom.altimetry.tools.sec_tools.grid_altimetry_data.py

Purpose
-------
Grid altimetry data for SEC (model fit) processing,
storing each measurement in a partitioned Parquet dataset, 
partitioned by (year, x_part, y_part).

Supported modes:
  1) Full regrid: remove entire dataset and rewrite from scratch for *all* years
     (use --regrid)
  2) Update/add a single year: only overwrite that year partition in the existing dataset
     (use --update_year YYYY)

If neither is provided, the script exits with a usage message.

Example usage:
--------------
1) Full regrid (5km grid):
   python grid_altimetry_data.py --regrid --binsize 5e3 --area greenland_is \
       -o /cpnet/altimetry/landice/gridded_altimetry -g greenland \
       -mi cs2 -da cryotempo_c001 \
       -dir /cpdata/SATS/RA/CRY/Cryo-TEMPO/BASELINE-C/001/LAND_ICE/GREENL \
       -pat '**/CS_OFFL_SIR_TDP_LI*C001*.nc' \
       --year_str_fname_indices -48 -44 \
       --lat latitude \
       --lon longitude \
       --elev elevation \
       --power backscatter \
       --time time

2) Update year 2015 only:
   python grid_altimetry_data.py --update_year 2015 --binsize 5e3 --area greenland_is \
       -o /cpnet/altimetry/landice/gridded_altimetry -g greenland \
       -mi cs2 -da cryotempo_c001 \
       -dir /cpdata/SATS/RA/CRY/Cryo-TEMPO/BASELINE-C/001/LAND_ICE/GREENL \
       -pat '**/CS_OFFL_SIR_TDP_LI*C001*.nc' \
       --year_str_fname_indices -48 -44 \
       --lat latitude \
       --lon longitude \
       --elev elevation \
       --power backscatter \
       --time time

# TODO
- add month column?
- Check time parameter
- Check x,y params - should they be diffs
- Check direction param
- add meta data to grid directory

remove load_processed_files(), etc. No need to do this any more.

"""

import argparse
import glob

# import json
import logging
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from netCDF4 import Dataset, Variable  # pylint: disable=E0611

from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers

log = logging.getLogger(__name__)


def get_variable(nc: Dataset, nc_var_path: str) -> Variable:
    """Retrieve variable from NetCDF file, handling groups if necessary."""
    parts = nc_var_path.split("/")
    var = nc
    for part in parts:
        var = var[part]
    return var


def grid_dataset(data_set: dict, regrid: bool, update_year: str) -> None:
    """
    Read NetCDF altimetry data, bin them into (x_bin, y_bin) grid cells,
    and store them in partitioned Parquet by (year, x_part, y_part).

    Two modes:
      1) regrid=True => remove entire dataset folder, write all years.
      2) update_year=YYYY => only overwrite that year partition, keep other years as-is.
    """

    # ----------------------------------------------------------------------
    # Determine output directory path
    # ----------------------------------------------------------------------
    output_dir = os.path.join(
        data_set["grid_output_dir"],
        data_set["mission"],
        f'{data_set["grid_name"]}_{int(data_set["bin_size"]/1000)}km_{data_set["dataset"]}',
    )

    log.info("output_dir=%s", output_dir)

    # ----------------------------------------------------------------------
    # Decide how to handle existing data folder + set "existing_data_behavior"
    # ----------------------------------------------------------------------
    if regrid:
        # Mode 1: Full regrid => remove entire directory, then create fresh
        if os.path.exists(output_dir):
            if output_dir != "/" and data_set["mission"] in output_dir:  # safety check
                log.info("Removing previous grid dir: %s ...", output_dir)
                shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        existing_data_behavior_str = "delete_matching"

    else:
        # Mode 2: Update a single year => do *not* remove the entire dataset
        if not os.path.exists(output_dir):
            log.error(
                "Output directory %s does not exist. "
                "Cannot update a year in a non-existent dataset!",
                output_dir,
            )
            sys.exit(1)

        existing_data_behavior_str = "delete_matching"

    # ----------------------------------------------------------------------
    # Setup grid and area
    # ----------------------------------------------------------------------
    grid = GridArea(data_set["grid_name"], data_set["bin_size"])
    thisarea = Area(data_set["area_filter"])

    log.info("grid specification=%s", data_set["grid_name"])

    # ----------------------------------------------------------------------
    # Gather all matching files, identify unique years
    # ----------------------------------------------------------------------
    search_dir = data_set["l2_directory"]
    pattern = data_set["search_pattern"]

    log.info("finding matching L2 files from %s/%s", search_dir, pattern)

    matching_files = glob.glob(f"{search_dir}/{pattern}", recursive=True)
    if not matching_files:
        log.error("No matching L2 files found in %s/%s", search_dir, pattern)
        sys.exit()

    start_idx, end_idx = data_set["year_str_fname_indices"]
    unique_years = set()

    log.info("finding unique years..")

    for file_path in matching_files:
        year_str = file_path[start_idx:end_idx]
        if len(year_str) != 4 or not year_str.isdigit():
            log.error("Invalid year string '%s' from file %s", year_str, file_path)
            sys.exit(1)
        unique_years.add(year_str)

    unique_years_list = sorted(unique_years)
    log.info("Unique years found in files: %s", unique_years_list)

    # Decide which years to process
    if regrid:
        # regrid => all years
        years_to_process = unique_years_list
    else:
        # update_year => only that year
        if update_year not in unique_years_list:
            log.error(
                "Requested update_year=%s but no files exist for that year! Exiting.", update_year
            )
            sys.exit(1)
        years_to_process = [update_year]

    # ----------------------------------------------------------------------
    # Process each year
    # ----------------------------------------------------------------------
    n_files_ingested = 0
    max_files = data_set.get("max_files")

    for year_str in years_to_process:
        log.info("Processing year: %s", year_str)
        files_this_year = [fp for fp in matching_files if fp[start_idx:end_idx] == year_str]
        if not files_this_year:
            log.info("No files for year=%s, skipping", year_str)
            continue

        n_files_this_year = len(files_this_year)

        log.info("Number of files found for year: %d", n_files_this_year)

        year_rows = []

        for n_file, file_path in enumerate(files_this_year):
            if max_files is not None and n_files_ingested >= max_files:
                break

            log.info(
                "Processing file:%d/%d  %s (year=%s)",
                n_file,
                n_files_this_year,
                file_path,
                year_str,
            )
            n_files_ingested += 1

            with Dataset(file_path) as nc:
                # lat/lon
                try:
                    lats = get_variable(nc, data_set["latitude_param"])[:].data
                    lons = get_variable(nc, data_set["longitude_param"])[:].data % 360.0
                except (KeyError, IndexError):
                    log.warning("No lat/lon in %s, skipping", file_path)
                    continue

                x_coords, y_coords = thisarea.latlon_to_xy(lats, lons)
                bool_mask, n_inside = thisarea.inside_area(lats, lons)
                if n_inside == 0:
                    log.debug("No points inside area for %s", file_path)
                    continue

                # elevation
                try:
                    elevs = get_variable(nc, data_set["elevation_param"])[:].data
                except (KeyError, IndexError):
                    log.warning("No elevation in %s, skipping", file_path)
                    continue
                try:
                    fill_elev = get_variable(  # pylint: disable=protected-access
                        nc, data_set["elevation_param"]
                    )._FillValue
                    elevs[elevs == fill_elev] = np.nan
                except AttributeError:
                    pass

                # power
                try:
                    pwr = get_variable(nc, data_set["power_param"])[:].data
                except (KeyError, IndexError):
                    log.warning("No power in %s, skipping", file_path)
                    continue
                try:
                    fill_pwr = get_variable(  # pylint: disable=protected-access
                        nc, data_set["power_param"]
                    )._FillValue
                    pwr[pwr == fill_pwr] = np.nan
                except AttributeError:
                    pass

                # time
                try:
                    time_data = get_variable(nc, data_set["time_param"])[:].data
                except (KeyError, IndexError):
                    log.warning("No time in %s, skipping", file_path)
                    continue

                # combine mask
                bool_mask = bool_mask & np.isfinite(elevs) & np.isfinite(pwr)
                if bool_mask.sum() == 0:
                    log.debug("No valid data left in %s after mask", file_path)
                    continue

                # subset
                x_valid = x_coords[bool_mask]
                y_valid = y_coords[bool_mask]
                elev_valid = elevs[bool_mask]
                pwr_valid = pwr[bool_mask]
                time_valid = time_data[bool_mask]

                # bin
                x_bin = ((x_valid - grid.minxm) / grid.binsize).astype(int)
                y_bin = ((y_valid - grid.minym) / grid.binsize).astype(int)

                # ascending logic
                ascending_locs = np.ones_like(x_valid, dtype=bool)
                try:
                    if nc.ascending_start_record == "None" and nc.descending_start_record != "None":
                        ascending_locs[:] = False
                    elif (
                        nc.descending_start_record == "None" and nc.ascending_start_record != "None"
                    ):
                        ascending_locs[:] = True
                except AttributeError:
                    pass

                # build DataFrame
                df = pd.DataFrame(
                    {
                        "year": year_str,
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
                df["x_part"] = df["x_bin"] // 20
                df["y_part"] = df["y_bin"] // 20

                year_rows.append(df)

        if not year_rows:
            log.info("No data for year=%s", year_str)
            continue

        year_df = pd.concat(year_rows, ignore_index=True)
        log.info(
            "Writing year=%s => %d total rows from %d files",
            year_str,
            len(year_df),
            len(files_this_year),
        )

        arrow_table = pa.Table.from_pandas(year_df)
        pq.write_to_dataset(
            table=arrow_table,
            root_path=output_dir,
            partition_cols=["year", "x_part", "y_part"],
            use_dictionary=True,
            compression="snappy",
            existing_data_behavior=existing_data_behavior_str,
        )

        del year_rows
        del year_df
        del arrow_table

    log.info("Partitioned Parquet dataset written to: %s", output_dir)


def main(args):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert altimetry data into partitioned Parquet with ragged layout. "
            "Supports regridding everything or updating a single year."
        )
    )

    # Required arguments
    parser.add_argument("--area", "-a", required=True)
    parser.add_argument("--binsize", "-b", required=True, type=float)
    parser.add_argument("--dataset", "-da", required=True)
    parser.add_argument("--elevation_param", "-elev", required=True)
    parser.add_argument("--gridarea", "-g", required=True)
    parser.add_argument("--l2_dir", "-dir", required=True)
    parser.add_argument("--lat_param", "-lat", required=True)
    parser.add_argument("--lon_param", "-lon", required=True)
    parser.add_argument("--mission", "-mi", required=True)
    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument("--power_param", "-power", required=True)
    parser.add_argument("--search_pattern", "-pat", required=True)
    parser.add_argument("--time_param", "-time", required=True)
    parser.add_argument("--year_str_fname_indices", "-ys", nargs=2, type=int, required=True)

    # Optional
    parser.add_argument("--max_files", "-mf", type=int)

    # The two mutually exclusive modes
    parser.add_argument(
        "--regrid",
        "-r",
        action="store_true",
        help="Remove entire dataset and re-write from scratch for all years.",
    )
    parser.add_argument(
        "--update_year", help="Overwrite data for only this single year (e.g. 2015)."
    )

    parser.add_argument("--debug", "-d", action="store_true")

    args = parser.parse_args(args)

    # 1) Check that user provided EXACTLY one of: regrid or update_year
    if args.regrid and args.update_year:
        log.error("Cannot specify both --regrid and --update_year simultaneously.")
        sys.exit(1)

    if not args.regrid and not args.update_year:
        log.error(
            "You must specify one of --regrid (full rewrite) or --update_year YEAR (update only)."
        )
        sys.exit(1)

    # 2) Set up logging
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

    # 3) Build data_set dict
    data_set = {
        "dataset": args.dataset,
        "year_str_fname_indices": args.year_str_fname_indices,
        "mission": args.mission,
        "grid_name": args.gridarea,
        "area_filter": args.area,
        "bin_size": args.binsize,
        "l2_directory": args.l2_dir,
        "search_pattern": args.search_pattern,
        "latitude_param": args.lat_param,
        "longitude_param": args.lon_param,
        "elevation_param": args.elevation_param,
        "power_param": args.power_param,
        "time_param": args.time_param,
        "max_files": args.max_files,
        "grid_output_dir": args.output_dir,
    }

    # 4) Run gridding in the chosen mode
    grid_dataset(data_set, regrid=args.regrid, update_year=args.update_year)


if __name__ == "__main__":
    main(sys.argv[1:])

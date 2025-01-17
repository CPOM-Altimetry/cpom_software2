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
       --yyyymm_str_fname_indices -48 -44 \
       --lat latitude \
       --lon longitude \
       --elev elevation \
       --power backscatter \
       --time time \

2) Update year 2015 only:
   python grid_altimetry_data.py --update_year 2015 --binsize 5e3 --area greenland_is \
       -o /cpnet/altimetry/landice/gridded_altimetry -g greenland \
       -mi cs2 -da cryotempo_c001 \
       -dir /cpdata/SATS/RA/CRY/Cryo-TEMPO/BASELINE-C/001/LAND_ICE/GREENL \
       -pat '**/CS_OFFL_SIR_TDP_LI*C001*.nc' \
       --yyyymm_str_fname_indices -48 -44 \
       --lat latitude \
       --lon longitude \
       --elev elevation \
       --power backscatter \
       --time time

# TODO
- direction (ascending) calc for other missions
- Check other params required?


"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from netCDF4 import Dataset, Variable  # pylint: disable=E0611

from cpom.altimetry.datasets.altdatasets import AltDataset
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


def grid_dataset(
    data_set: dict, regrid: bool, update_year: int | None = None, confirm_regrid=True
) -> None:
    """
    Read NetCDF altimetry data, bin them into (x_bin, y_bin) grid cells,
    and store them in partitioned Parquet by (year, x_part, y_part).

    Two modes:
      1) regrid=True => remove entire dataset folder, write all years or if update_year
         also specified just that year
      2) update_year=YYYY => only overwrite that year partition, keep other years as-is.

    Args:
        data_set (dict) : dictionary of gridding parameters.

            data_set = {
                "dataset": str, # cryotempo_c001
                "yyyymm_str_fname_indices": (int,int), # neg indices in fname of start YYYY
                "mission": str, # cs2, e1,e2,ev,s3a,s3b,is2
                "grid_name": str,  # GridArea name: is 'greenland'
                "area_filter": str,  # CPOM area name, used for masking input measurements
                "bin_size": int,  # grid binsize to use, example: 5e3 (for 5km grid)
                "l2_directory": str,  # L2 base dir to recursively search for L2 files,
                "search_pattern": str,  # search pattern for L2 file discovery
                "latitude_param": str,  # latitude name in netcdf. example: data/ku/latitude
                "longitude_param": str,  # longitude name in netcdf. example: data/ku/longitude
                "elevation_param": str,  # elevation parameter in netcdf. use / for groups
                "power_param": str,  # power (backscatter) parameter in netcdf. / for groups
                "time_param": str,  # time parameter in netcdf. / for groups
                "max_files": int,  # limit to this number of input files ingested
                "grid_output_dir": str,  # base path of grid output dir to be created or updated
                "partition_by_month": bool, # if True partition parquet files by year AND month
            }
        regrid (bool) : if True remove any previous grid parquet and regrid from scratch.
                        if False, then will try updating the year given in update_year
        update_year (int|None) : YYYY. year to update. Is set only this year will be ingested
        confirm_regrid (bool): if True user is prompted to confirm regrid before any previous
                            data set is overwritten.

    """

    start_time = time.time()  # Record the start time

    if not regrid and update_year is None:
        log.error("update_year must be provided if regrid is False")
        sys.exit(1)

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
                if confirm_regrid:
                    response = (
                        input("Confirm removal of previous grid archive? (y/n): ").strip().lower()
                    )
                    if response == "y":
                        shutil.rmtree(output_dir)
                    else:
                        print("Exiting as user requested not to overwrite grid archive")
                        sys.exit(0)
            else:
                log.error("invalid output_dir path: %s", output_dir)
                sys.exit(1)
        os.makedirs(output_dir, exist_ok=True)

    else:
        # Mode 2: Update a single year => do *not* remove the entire dataset
        if not os.path.exists(output_dir):
            log.error(
                "Output directory %s does not exist. "
                "Cannot update a year in a non-existent dataset!",
                output_dir,
            )
            sys.exit(1)

    # ----------------------------------------------------------------------
    # Setup grid and area
    # ----------------------------------------------------------------------
    grid = GridArea(data_set["grid_name"], data_set["bin_size"])
    thisarea = Area(data_set["area_filter"])
    this_dataset = AltDataset(data_set["dataset"])

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

    start_idx, end_idx = data_set["yyyymm_str_fname_indices"]
    yr_start_idx = start_idx
    yr_end_idx = end_idx - 2
    month_start_idx = start_idx + 4
    month_end_idx = end_idx

    unique_years = set()

    log.info("finding unique years..")

    for file_path in matching_files:
        year_val = int(file_path[yr_start_idx:yr_end_idx])
        month_val = int(file_path[month_start_idx:month_end_idx])

        if not 1900 <= year_val <= 2100:
            log.error("Invalid year: %d in file %s", year_val, file_path)
            sys.exit(1)
        if not 1 <= month_val <= 12:
            log.error("Invalid month: %d in file %s", month_val, file_path)
            sys.exit(1)

        unique_years.add(year_val)

    unique_years_list = sorted(unique_years)
    log.info("Unique years found in files: %s", unique_years_list)

    # Decide which years to process

    if regrid and update_year is not None:
        if update_year not in unique_years_list:
            log.error(
                "Requested update_year=%s but no files exist for that year! Exiting.", update_year
            )

            sys.exit(1)
        years_to_process = [update_year]
    elif regrid:
        # regrid => all years
        years_to_process = unique_years_list
    else:
        # update_year => only that year
        if update_year not in unique_years_list:
            log.error(
                "Requested update_year =%d but no files exist for that year! Exiting.", update_year
            )

            sys.exit(1)
        years_to_process = [update_year]

    log.info("Years to process: %s", str(years_to_process))
    # ----------------------------------------------------------------------
    # Process each year
    # ----------------------------------------------------------------------
    total_files_ingested = 0
    total_files_rejected = 0
    years_ingested = []
    years_files_ingested = []
    years_files_rejected = []

    max_files = data_set.get("max_files")

    for year in years_to_process:
        files_ingested = 0
        files_rejected = 0
        log.info("Processing year: %d", year)
        files_this_year = [fp for fp in matching_files if int(fp[yr_start_idx:yr_end_idx]) == year]
        if not files_this_year:
            log.info("No files for year=%d, skipping", year)
            continue

        n_files_this_year = len(files_this_year)

        years_ingested.append(year)

        log.info("Number of files found for year: %d", n_files_this_year)

        year_rows = []

        for n_file, file_path in enumerate(files_this_year):
            if max_files is not None and total_files_ingested >= max_files:
                break

            # parse month from the file path
            month = int(file_path[month_start_idx:month_end_idx])  # e.g. int('03')
            if month < 1 or month > 12:
                log.warning("Invalid month '%d' in file %s", month, file_path)
                files_rejected += 1
                total_files_rejected += 1
                continue

            log.info(
                "Processing file from %d: %d/%d  %s ",
                year,
                n_file,
                n_files_this_year,
                file_path,
            )
            total_files_ingested += 1
            files_ingested += 1

            with Dataset(file_path) as nc:
                # lat/lon
                try:
                    lats = get_variable(nc, data_set["latitude_param"])[:].data
                    lons = get_variable(nc, data_set["longitude_param"])[:].data % 360.0
                except (KeyError, IndexError):
                    log.warning("No lat/lon in %s, skipping", file_path)
                    total_files_rejected += 1
                    files_rejected += 1
                    continue

                x_coords, y_coords = thisarea.latlon_to_xy(lats, lons)
                bool_mask, n_inside = thisarea.inside_area(lats, lons)
                if n_inside == 0:
                    log.debug("No points inside area for %s", file_path)
                    total_files_rejected += 1
                    files_rejected += 1
                    continue

                # elevation
                try:
                    elevs = get_variable(nc, data_set["elevation_param"])[:].data
                except (KeyError, IndexError):
                    log.warning("No elevation in %s, skipping", file_path)
                    total_files_rejected += 1
                    files_rejected += 1
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
                    total_files_rejected += 1
                    files_rejected += 1
                    continue
                try:
                    fill_pwr = get_variable(  # pylint: disable=protected-access
                        nc, data_set["power_param"]
                    )._FillValue
                    pwr[pwr == fill_pwr] = np.nan
                except AttributeError:
                    pass

                # time parameter
                #    - Note a constant is added to the returned time in seconds from
                #      data_set["time_secs_to_1950"] so that all times are referenced to
                #      00:00 1/1/1950
                #    - CryoTEMPO units of time are UTC seconds since 00:00:00 1-Jan-2000
                #    - S3 Thematic LI, time_20_ku:units = "seconds since 2000-01-01 00:00:00.0" ;
                #    - FDGR4ALT: expert/time:units = "seconds since 1950-01-01 00 :00 :00.00 " ;
                try:
                    time_data = get_variable(nc, data_set["time_param"])[:].data
                except (KeyError, IndexError):
                    log.warning("No time parameter in %s, skipping", file_path)
                    total_files_rejected += 1
                    files_rejected += 1
                    continue

                # Add the time constant so that times are referenced to 00:00 1/1/1950
                time_data += data_set["time_secs_to_1950"]

                # Direction parameter (nadir ascending/descending) at each measurement point

                ascending_points = this_dataset.find_measurement_directions(nc)

                # combine mask
                bool_mask = bool_mask & np.isfinite(elevs) & np.isfinite(pwr)
                if bool_mask.sum() == 0:
                    log.debug("No valid data left in %s after mask", file_path)
                    total_files_rejected += 1
                    files_rejected += 1
                    continue

                # subset
                x_valid = x_coords[bool_mask]
                y_valid = y_coords[bool_mask]
                elev_valid = elevs[bool_mask]
                pwr_valid = pwr[bool_mask]
                time_valid = time_data[bool_mask]
                ascending_valid = ascending_points[bool_mask]

                # bin
                # x_bin = ((x_valid - grid.minxm) / grid.binsize).astype(int)
                # y_bin = ((y_valid - grid.minym) / grid.binsize).astype(int)

                # Get the nKm grid cell indices for the track x,y locations
                x_bin, y_bin = grid.get_col_row_from_x_y(x_valid, y_valid)

                # Store the difference to cell centre, rather than actual x,y
                # Get the x,y offsets from grid point center coordinates
                xoffset, yoffset = grid.get_xy_relative_to_cellcentres(x_valid, y_valid)

                # Build the base dictionary that is always included
                data_dict = {
                    "year": year,  # 'year' comes from your outer loop variable (an int)
                    "x_bin": x_bin,
                    "y_bin": y_bin,
                    "x_cell_offset": xoffset,
                    "y_cell_offset": yoffset,
                    "elevation": elev_valid,
                    "power": pwr_valid,
                    "ascending": ascending_valid,
                    "time": time_valid,
                }

                # Conditionally add the 'month' key if partitioning by month is enabled
                if data_set["partition_by_month"]:
                    data_dict["month"] = month

                # Build the DataFrame
                df = pd.DataFrame(data_dict)

                df["x_part"] = df["x_bin"] // data_set["partition_xy_chunking"]
                df["y_part"] = df["y_bin"] // data_set["partition_xy_chunking"]

                year_rows.append(df)

        years_files_ingested.append(files_ingested)
        years_files_rejected.append(files_rejected)

        if not year_rows:
            log.info("No data for year=%d", year)
            continue

        year_df = pd.concat(year_rows, ignore_index=True)
        log.info(
            "Writing year=%d => %d total rows from %d files",
            year,
            len(year_df),
            len(files_this_year),
        )

        arrow_table = pa.Table.from_pandas(year_df)

        if data_set["partition_by_month"]:
            partition_cols = ["year", "month", "x_part", "y_part"]
        else:
            partition_cols = ["year", "x_part", "y_part"]

        pq.write_to_dataset(
            table=arrow_table,
            root_path=output_dir,
            partition_cols=partition_cols,
            use_dictionary=True,
            compression="snappy",
            existing_data_behavior="delete_matching",
        )

        del year_rows
        del year_df
        del arrow_table

    log.info("Partitioned Parquet dataset written to: %s", output_dir)

    # ----------------------------------------------------------------------
    # Save data_set dictionary as JSON in the top-level parquet directory
    # ----------------------------------------------------------------------

    data_set["years_ingested"] = years_ingested
    data_set["years_files_ingested"] = years_files_ingested
    data_set["years_files_rejected"] = years_files_rejected
    data_set["total_l2_files_rejected"] = total_files_rejected
    data_set["total_l2_files_ingested"] = total_files_ingested
    data_set["grid_xmin"] = grid.minxm
    data_set["grid_ymin"] = grid.minym
    data_set["grid_crs"] = grid.coordinate_reference_system
    data_set["grid_x_size"] = grid.grid_x_size
    data_set["grid_y_size"] = grid.grid_y_size

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time

    # Convert elapsed time to HH:MM:SS
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    data_set["gridding_time"] = f"{hours:02}:{minutes:02}:{seconds:02}"

    meta_json_path = os.path.join(output_dir, "grid_meta.json")
    try:
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(data_set, f, indent=2)
        log.info("Wrote data_set metadata to %s", meta_json_path)
    except OSError as e:
        log.error("Failed to write grid_meta.json: %s", e)


def main(args):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert altimetry data into partitioned Parquet with ragged layout. "
            "Supports regridding everything or updating a single year."
        )
    )

    # Required arguments
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
        "--dataset",
        "-da",
        help=("string identifying L2 data set: valid values are: cryotempo_c001"),
        required=True,
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
        "--mission",
        "-mi",
        help=("mission of data set: e1, e2, ev, s3a, s3b, cs2, is2"),
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        help=(
            "Base path to create or update the gridded data set Parquet format files"
            "Actual parquet directory is formed from: <output_dir>/<mission>/<grid_name>_"
            "<binsize in km>km_<dataset name>"
        ),
        required=True,
        type=str,
    )

    parser.add_argument(
        "--partition_by_month",
        "-pm",
        help=(
            "If set to True, the Parquet dataset will be partitioned by both year and month"
            " (in addition to the coarser spatial partitions defined below). "
            "This results in a folder hierarchy such as year=YYYY/month=MM/x_part=.../y_part=..."
            "  . Note that finer partitioning (i.e. by month) may significantly increase the number"
            "  of output files, which can increase write times and, for very large datasets,"
            " potentially affect read performance depending upon the query types."
        ),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--partition_xy_chunking",
        "-px",
        help=(
            "This parameter sets the chunking factor for spatial partitioning. "
            "Individual grid cells (identified by x_bin and y_bin) are grouped into larger "
            "partitions by dividing them by this factor. For example, a value of 20 means"
            " that the coarser partition columns (x_part, y_part) will be computed as x_bin//20 "
            "and y_bin//20. Adjust this value to control the trade-off between file size and "
            "number of partitions.."
        ),
        required=False,
        default=20,
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

    parser.add_argument(
        "--regrid",
        "-r",
        action="store_true",
        help="Remove entire dataset and re-write from scratch for all years.",
    )

    parser.add_argument(
        "--search_pattern",
        "-pat",
        help=(
            "search pattern for L2 file discovery. Example for CryoTEMPO C001: "
            "**/CS_OFFL_SIR_TDP_LI*C001*.nc"
        ),
        required=True,
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
        "--update_year",
        help=(
            "YYYY (int): Overwrite data for only this single year (e.g. 2015)."
            "example --update_year 2015"
        ),
        type=int,
    )

    parser.add_argument(
        "--yyyymm_str_fname_indices",
        "-ys",
        help=(
            "int int : negative indices from end of L2 file name or path which point to the "
            "YYYY start year in the name. For example with CryoTEMPO L2 files, this is: "
            "-48 -44"
        ),
        nargs=2,
        type=int,
        required=True,
    )

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
        "yyyymm_str_fname_indices": args.yyyymm_str_fname_indices,
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
        "partition_by_month": args.partition_by_month,
        "partition_xy_chunking": args.partition_xy_chunking,
    }

    # 4) Run gridding in the chosen mode
    grid_dataset(data_set, regrid=args.regrid, update_year=args.update_year)


if __name__ == "__main__":
    main(sys.argv[1:])

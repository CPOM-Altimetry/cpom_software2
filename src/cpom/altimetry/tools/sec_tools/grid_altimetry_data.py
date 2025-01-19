#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cpom.altimetry.tools.sec_tools.grid_altimetry_data.py

Purpose:
  Grid altimetry data for SEC (model fit) processing,
  storing each measurement in a partitioned Parquet dataset,
  partitioned by (year, x_part, y_part) or (year, month, x_part, y_part)
  if --partition_by_month is used.

  Supports two modes:
    1) Full regrid: remove entire dataset and rewrite from scratch for *all* years
       (use --regrid)
    2) Update/add a single year: only overwrite that year partition in the existing dataset
       (use --update_year YYYY)

Additionally:
  - Maintains a JSON file "partition_index.json" at the top-level Parquet directory,
    which stores the unique (x_part, y_part) pairs (and optionally year/month if desired).
    This helps future reading scripts avoid expensive globbing.

Example usage:
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
         --time time

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


Notes / Changes:
  - We now write "partition_index.json" to the output directory, which tracks the (x_part, y_part)
    pairs encountered so far. This helps future reads avoid a giant glob scan.
  - If --partition_by_month is used, you might also store (year, month, x_part, y_part) 
    in that index for monthly queries. Here, we demonstrate a simpler approach: we only store the 
    spatial chunks in the index (x_part, y_part). If you also want (year, month), you can adapt 
    similarly.

TODO:
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
    """Retrieve a variable from NetCDF file, handling groups if necessary.

    Args:
        nc (Dataset): The opened NetCDF dataset
        nc_var_path (str): Path to the variable (e.g. "data/ku/latitude")
    Returns:
        Variable: The NetCDF variable object
    """
    parts = nc_var_path.split("/")
    var = nc
    for part in parts:
        var = var[part]
    return var


def update_partition_index(
    output_dir: str,
    df: pd.DataFrame,
    # store_month: bool = False,
) -> None:
    """Update or create the partition_index.json file, which records
    all (x_part, y_part) encountered so far.

    If store_month=True, you'd store (year, month, x_part, y_part) as well.
    For simplicity, we only store spatial chunk pairs (x_part, y_part).

    Args:
        output_dir (str): The top-level parquet directory (where grid_meta.json is).
        df (pd.DataFrame): DataFrame with columns: [x_part, y_part, (month)?].
        store_month (bool): If True, you could store year/month too.
    """
    partition_index_path = os.path.join(output_dir, "partition_index.json")

    # Extract the unique (x_part, y_part) from df
    unique_xy = df[["x_part", "y_part"]].drop_duplicates()
    new_pairs = unique_xy.values.tolist()  # list of [x_part, y_part]

    # Load existing partition_index.json (if present)
    if os.path.isfile(partition_index_path):
        with open(partition_index_path, "r", encoding="utf-8") as f_idx:
            partition_data = json.load(f_idx)
    else:
        # If it doesn't exist yet, create fresh
        partition_data = {
            "xypart_list": [],
        }

    existing_pairs = set(tuple(xy) for xy in partition_data["xypart_list"])

    # Add the new pairs
    for xy in new_pairs:
        existing_pairs.add(tuple(xy))

    # Convert back to a sorted list for stable output
    updated_list = sorted(list(existing_pairs))
    partition_data["xypart_list"] = [list(xy) for xy in updated_list]

    # Write updated index back to disk
    with open(partition_index_path, "w", encoding="utf-8") as f_idx:
        json.dump(partition_data, f_idx, indent=2)

    log.info(
        "Updated partition_index.json with %d new (x_part,y_part) chunk pairs (total now %d).",
        len(new_pairs),
        len(updated_list),
    )


def grid_dataset(
    data_set: dict,
    regrid: bool,
    update_year: int | None = None,
    confirm_regrid: bool = True,
) -> None:
    """
    Read NetCDF altimetry data, bin them into (x_bin, y_bin) grid cells,
    and store them in a partitioned Parquet dataset by (year, [month,] x_part, y_part).

    Two modes:
      1) Full regrid => remove entire dataset folder, write all years (or if update_year
         also specified, just that single year).
      2) update_year=YYYY => only overwrite that year partition, keep other years as-is.

    Also maintains a "partition_index.json" listing all (x_part, y_part) chunks
    so subsequent reads can skip a huge glob.

    Args:
        data_set (dict): Dictionary of gridding parameters:

            {
                "dataset": str,              # e.g. 'cryotempo_c001'
                "yyyymm_str_fname_indices": (int,int),  # negative indices from end of filename
                "mission": str,              # e.g. 'cs2', 's3a', ...
                "grid_name": str,            # name recognized by GridArea
                "area_filter": str,          # CPOM area name
                "bin_size": float,           # e.g. 5000
                "l2_directory": str,         # base dir of L2 netcdf
                "search_pattern": str,       # glob pattern
                "latitude_param": str,
                "longitude_param": str,
                "elevation_param": str,
                "power_param": str,
                "time_param": str,
                "max_files": int | None,
                "grid_output_dir": str,
                "partition_by_month": bool,
                "partition_xy_chunking": int,
                # etc...
            }
        regrid (bool): If True, remove any previous dataset and regrid from scratch.
        update_year (int|None): If provided, only that year is updated (mode 2).
        confirm_regrid (bool): If True, ask for user confirmation before removing data.
    """

    start_time = time.time()

    if not regrid and update_year is None:
        log.error("update_year must be provided if regrid is False.")
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
    # Decide how to handle existing data folder
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
                log.error("Invalid output_dir path: %s", output_dir)
                sys.exit(1)
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Mode 2: Update a single year => do *not* remove entire dataset
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
    # Gather all matching L2 NetCDF files and identify unique years
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
    log.info("finding unique years...")

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
        # regrid with a single year
        if update_year not in unique_years_list:
            log.error("Requested update_year=%s but no files exist for that year!", update_year)
            sys.exit(1)
        years_to_process = [update_year]
    elif regrid:
        # regrid => all years
        years_to_process = unique_years_list
    else:
        # update_year => only that year
        if update_year not in unique_years_list:
            log.error("Requested update_year=%s but no files exist for that year!", update_year)
            sys.exit(1)
        years_to_process = [update_year]

    log.info("Years to process: %s", years_to_process)

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

        # Gather files for this year
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

            # parse month
            month = int(file_path[month_start_idx:month_end_idx])
            if month < 1 or month > 12:
                log.warning("Invalid month '%d' in file %s", month, file_path)
                files_rejected += 1
                total_files_rejected += 1
                continue

            log.info(
                "Processing file from %d: %d/%d => %s",
                year,
                n_file + 1,
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

                # time
                try:
                    time_data = get_variable(nc, data_set["time_param"])[:].data
                except (KeyError, IndexError):
                    log.warning("No time param in %s, skipping", file_path)
                    total_files_rejected += 1
                    files_rejected += 1
                    continue

                # Convert time to reference (e.g. 1950 epoch)
                time_data += data_set["time_secs_to_1950"]  # user sets in data_set

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
                x_bin, y_bin = grid.get_col_row_from_x_y(x_valid, y_valid)

                # offsets from cell center
                xoffset, yoffset = grid.get_xy_relative_to_cellcentres(x_valid, y_valid)

                data_dict = {
                    "year": year,
                    "x_bin": x_bin,
                    "y_bin": y_bin,
                    "x_cell_offset": xoffset,
                    "y_cell_offset": yoffset,
                    "elevation": elev_valid,
                    "power": pwr_valid,
                    "ascending": ascending_valid,
                    "time": time_valid,
                }

                # If partition_by_month, also store month column
                if data_set["partition_by_month"]:
                    data_dict["month"] = month

                df_meas = pd.DataFrame(data_dict)

                # define coarse partition indices (x_part, y_part)
                df_meas["x_part"] = df_meas["x_bin"] // data_set["partition_xy_chunking"]
                df_meas["y_part"] = df_meas["y_bin"] // data_set["partition_xy_chunking"]

                year_rows.append(df_meas)

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

        # Convert to Arrow
        arrow_table = pa.Table.from_pandas(year_df)

        # Partition columns
        if data_set["partition_by_month"]:
            partition_cols = ["year", "month", "x_part", "y_part"]
        else:
            partition_cols = ["year", "x_part", "y_part"]

        # Write out to the dataset
        pq.write_to_dataset(
            table=arrow_table,
            root_path=output_dir,
            partition_cols=partition_cols,
            use_dictionary=True,
            compression="snappy",
            existing_data_behavior="delete_matching",
        )

        # ------------------------------------------------------------------
        # NEW: Update the partition_index.json with (x_part, y_part) from this year
        # ------------------------------------------------------------------
        update_partition_index(
            output_dir=output_dir,
            df=year_df,
            # store_month=False,  # or True if you want to also store year/month
        )

        # cleanup
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

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    data_set["gridding_time"] = f"{hours:02}:{minutes:02}:{seconds:02}"

    meta_json_path = os.path.join(output_dir, "grid_meta.json")
    try:
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(data_set, f, indent=2)
        log.info("Wrote data_set metadata to %s", meta_json_path)
    except OSError as exc:
        log.error("Failed to write grid_meta.json: %s", exc)


def main(args):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert altimetry data into partitioned Parquet with ragged layout. "
            "Supports regridding everything or updating a single year. "
            "Also maintains a partition_index.json for faster subsequent reads."
        )
    )

    # Required arguments
    parser.add_argument(
        "--area",
        "-a",
        help=("Area name for data filtering/masking. Example: greenland_is"),
        required=True,
    )

    parser.add_argument(
        "--binsize",
        "-b",
        help="Grid binsize in m (e.g., 5e3 == 5km).",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--dataset",
        "-da",
        help=("String identifying L2 data set: e.g. cryotempo_c001"),
        required=True,
    )

    parser.add_argument(
        "--debug",
        "-d",
        help="Output debug log messages to console.",
        action="store_true",
    )

    parser.add_argument(
        "--elevation_param",
        "-elev",
        help="Name of elevation parameter in L2 netcdf. e.g. data/ku/elevation",
        required=True,
    )

    parser.add_argument(
        "--gridarea",
        "-g",
        help="Output grid area specification name (see cpom.gridding.gridareas). e.g. greenland",
        required=True,
    )

    parser.add_argument(
        "--l2_dir",
        "-dir",
        help="L2 base directory to search for netcdf files, e.g. /cpdata/SATS/RA/CRY/Cryo-TEMPO/",
        required=True,
    )

    parser.add_argument(
        "--lat_param",
        "-lat",
        help="Name of latitude param in L2 netcdf, e.g. data/ku/latitude",
        required=True,
    )

    parser.add_argument(
        "--lon_param",
        "-lon",
        help="Name of longitude param in L2 netcdf, e.g. data/ku/longitude",
        required=True,
    )

    parser.add_argument(
        "--max_files",
        "-mf",
        help="Restrict number of input L2 files (int).",
        type=int,
    )

    parser.add_argument(
        "--mission",
        "-mi",
        help="Mission name: e1, e2, ev, s3a, s3b, cs2, is2 etc.",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        help=(
            "Base path to create or update the partitioned Parquet dataset. "
            "The final directory is <output_dir>/<mission>/<grid_name>_<binsize in km>km_<dataset>"
        ),
        required=True,
        type=str,
    )

    parser.add_argument(
        "--partition_by_month",
        "-pm",
        help=(
            "If set, also partition by month=MM. This creates subdirs "
            "year=YYYY/month=MM/x_part=NN/y_part=MM."
        ),
        action="store_true",
    )

    parser.add_argument(
        "--partition_xy_chunking",
        "-px",
        help=(
            "Chunk factor for x_bin,y_bin grouping. e.g. 20 => x_part=x_bin//20, y_part=y_bin//20."
        ),
        default=20,
        type=int,
    )

    parser.add_argument(
        "--power_param",
        "-power",
        help="Name of the backscatter (power) param in L2 netcdf, e.g. data/ku/backscatter",
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
        help="Glob pattern for L2 file discovery. e.g. **/CS_OFFL_SIR_TDP_LI*C001*.nc",
        required=True,
    )

    parser.add_argument(
        "--time_param",
        "-time",
        help="Name of time param in L2 netcdf, e.g. data/ku/time",
        required=True,
    )

    parser.add_argument(
        "--update_year",
        help="YYYY: Overwrite data for only this single year (e.g. --update_year 2015).",
        type=int,
    )

    parser.add_argument(
        "--yyyymm_str_fname_indices",
        "-ys",
        help=(
            "Two ints: negative indices from end of filename for reading YYYY and MM. "
            "e.g. -48 -44 for CryoTEMPO files."
        ),
        nargs=2,
        type=int,
        required=True,
    )

    # We'll assume user sets time_secs_to_1950 or we default
    # You can also store in your data_set if you want to handle 'since 2000' etc.
    parser.add_argument(
        "--time_secs_to_1950",
        help=(
            "Offset to add to time param so that time=0 => 1950-01-01"
            ". Default=1577923200 if netcdf is 'since 2000'? (example)."
        ),
        type=float,
        default=0.0,
    )

    args = parser.parse_args(args)

    # Check that user provided EXACTLY one of --regrid or --update_year
    if args.regrid and args.update_year:
        log.error("Cannot specify both --regrid and --update_year simultaneously.")
        sys.exit(1)
    if not args.regrid and not args.update_year:
        log.error(
            "You must specify one of --regrid (full rewrite) or --update_year YEAR (update only)."
        )
        sys.exit(1)

    # Setup logging
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

    # Build data_set dict
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
        "time_secs_to_1950": args.time_secs_to_1950,
    }

    # Run gridding
    grid_dataset(data_set, regrid=args.regrid, update_year=args.update_year)


if __name__ == "__main__":
    main(sys.argv[1:])

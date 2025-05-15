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
    which stores the unique (year, month, x_part, y_part, file_path) for each partition file.
    This helps future reading scripts avoid expensive globbing; they can just read
    partition_index.json to discover which files exist.

Example usage:
  1) Full regrid (5km grid):
     python grid_altimetry_data.py --regrid \
        --binsize 5e3 \
        --area greenland_is \
        -o /cpnet/altimetry/landice/gridded_altimetry \
        -g greenland  \
        --mission cs2 \
        --dataset cryotempo_li \
        -dir /cpdata/SATS/RA/CRY/Cryo-TEMPO/BASELINE-C/001/LAND_ICE/GREENL \
        -pat '**/CS_OFFL_SIR_TDP_LI*C001*.nc' \
        --yyyymm_str_fname_indices -48 -42 \
        --lat_param latitude  --lon_param longitude \
        --elevation_param elevation \
        --power_param backscatter \
        --time_param time

  2) Update year 2015 only:

  python grid_altimetry_data.py --update_year 2015 \
        --binsize 5e3 \
        --area greenland_is \
        -o /cpnet/altimetry/landice/gridded_altimetry \
        -g greenland  \
        --mission cs2 \
        --dataset cryotempo_li \
        -dir /cpdata/SATS/RA/CRY/Cryo-TEMPO/BASELINE-C/001/LAND_ICE/GREENL \
        -pat '**/CS_OFFL_SIR_TDP_LI*C001*.nc' \
        --yyyymm_str_fname_indices -48 -42 \
        --lat_param latitude  --lon_param longitude \
        --elevation_param elevation \
        --power_param backscatter \
        --time_param time


Notes / Changes:
  - Instead of using pq.write_to_dataset(..., partition_cols=...), we manually group each
    year's DataFrame by (year, [month,], x_part, y_part) and write exactly one .parquet file
    per partition to a folder like:
         year=YYYY[/month=MM]/x_part=NNN/y_part=MMM/data.parquet
  - We record each partition's file path in partition_index.json, including
    [year, month, x_part, y_part, file_path].
  - If --partition_by_month is used, we store 'month' in each group; otherwise it's None.

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
from typing import Dict, List, Union

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
    """Retrieve variable from NetCDF file, handling groups if necessary.

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


def _ensure_dir(path: str) -> None:
    """Utility function to ensure a directory exists, create if needed."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def update_partition_index_file(
    index_path: str,
    new_entries: List[Dict[str, Union[int, str, None]]],
) -> None:
    """Update or create the 'partition_index.json' file, storing a list of
    partition-file entries.

    Each entry is a dict like:
      {
        "year": 2015,
        "month": 3 or None,
        "x_part": 12,
        "y_part": 34,
        "file_path": "year=2015/month=03/x_part=12/y_part=34/data.parquet"
      }

    Args:
        index_path (str): Full path to partition_index.json in the output dir.
        new_entries (List[Dict[str, Union[int, str, None]]]):
            The new records to add (no duplicates).

    The function merges these entries with any existing ones in the file,
    deduplicates, and writes out the updated JSON.
    """
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    else:
        index_data = {"entries": []}

    existing = set()
    for rec in index_data["entries"]:
        # We'll define a stable dedup key
        # e.g. (year, month, x_part, y_part, file_path)
        y = rec.get("year", None)
        m = rec.get("month", None)
        xp = rec.get("x_part", None)
        yp = rec.get("y_part", None)
        fp = rec.get("file_path", None)
        existing.add((y, m, xp, yp, fp))

    additions = 0
    for rec in new_entries:
        key = (
            rec.get("year"),
            rec.get("month"),
            rec.get("x_part"),
            rec.get("y_part"),
            rec.get("file_path"),
        )
        if key not in existing:
            index_data["entries"].append(rec)
            existing.add(key)
            additions += 1

    if additions > 0:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
        log.info("Appended %d new entries into partition_index.json", additions)
    else:
        log.info("No new entries to add to partition_index.json (all duplicates).")


def grid_dataset(
    data_set: dict,
    regrid: bool,
    update_year: int | None = None,
    confirm_regrid: bool = True,
) -> dict:
    """
    Read NetCDF altimetry data, bin them into (x_bin, y_bin) grid cells,
    and store them in a partitioned Parquet dataset by (year, [month,], x_part, y_part).
    Each partition is written as exactly one data.parquet file, and we record
    that file's path in partition_index.json.

    Two modes:
      1) Full regrid => remove entire dataset folder, write all years (or if update_year
         also specified, just that single year).
      2) update_year=YYYY => only overwrite that year partition, keep other years as-is.

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
                "time_secs_to_1950": float,
                # etc...
            }
        regrid (bool): If True, remove any previous dataset and regrid from scratch.
        update_year (int|None): If provided, only that year is updated (mode 2).
        confirm_regrid (bool): If True, ask for user confirmation before removing data.
    Returns:
        data_set (dict) : updated gridding parameter dictionary
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

    matching_files = glob.glob(os.path.join(search_dir, pattern), recursive=True)
    if not matching_files:
        log.error("No matching L2 files found in %s/%s", search_dir, pattern)
        sys.exit()

    start_idx = int(data_set["yyyymm_str_fname_indices"][0])
    end_idx = int(data_set["yyyymm_str_fname_indices"][1])
    log.info("end_idx %d", end_idx)

    yr_start_idx = start_idx
    yr_end_idx = end_idx - 2
    month_start_idx = start_idx + 4
    month_end_idx = end_idx

    log.info(
        "YYYYMM indices: %d %d %d %d", yr_start_idx, yr_end_idx, month_start_idx, month_end_idx
    )

    unique_years = set()
    log.info("finding unique years....")

    for file_path in matching_files:
        year_val = int(file_path[yr_start_idx:yr_end_idx])
        log.info("file_path %s %d %d", file_path, month_start_idx, month_end_idx)
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
            log.error("Requested update_year=%s but no files exist for that year!", update_year)
            sys.exit(1)
        years_to_process = [update_year]
    elif regrid:
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

    # We'll gather final row counts for metadata
    total_rows_for_all_years = 0

    # We'll store new partition index records here
    partition_index_new_entries = []

    for year_val in years_to_process:
        files_ingested = 0
        files_rejected = 0
        log.info("Processing year: %d", year_val)

        # Gather files for this year
        files_this_year = [
            fp for fp in matching_files if int(fp[yr_start_idx:yr_end_idx]) == year_val
        ]
        if not files_this_year:
            log.info("No files for year=%d, skipping", year_val)
            continue

        n_files_this_year = len(files_this_year)
        years_ingested.append(year_val)
        log.info("Number of files found for year: %d", n_files_this_year)

        big_rows_list = []

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
                year_val,
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
                time_data += data_set["time_secs_to_1950"]

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
                    "year": year_val,
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

                big_rows_list.append(df_meas)

        years_files_ingested.append(files_ingested)
        years_files_rejected.append(files_rejected)

        if not big_rows_list:
            log.info("No data for year=%d after filtering", year_val)
            continue

        df_year = pd.concat(big_rows_list, ignore_index=True)
        log.info(
            "Year=%d => total rows = %d from %d files",
            year_val,
            len(df_year),
            len(files_this_year),
        )

        # Group the year's data by (year, [month,], x_part, y_part)
        group_cols = ["year", "x_part", "y_part"]
        if data_set["partition_by_month"]:
            group_cols.insert(1, "month")  # => [year, month, x_part, y_part]

        grouped = df_year.groupby(group_cols, as_index=False)
        rows_in_this_year = 0

        for keys, chunk_df in grouped:
            # keys might be (year, x_part, y_part) or (year, month, x_part, y_part)
            if data_set["partition_by_month"]:
                # e.g. (year_val, month_val, xp, yp)
                if len(keys) == 4:
                    yv, mv, xp, yp = keys
                else:
                    # fallback in case of mismatch
                    yv, xp, yp = keys
                    mv = None
            else:
                # e.g. (year_val, xp, yp)
                yv, xp, yp = keys
                mv = None

            # Build the subdirectory
            if data_set["partition_by_month"] and mv is not None:
                # year=YYYY/month=MM/x_part=xxx/y_part=yyy
                subdir = os.path.join(
                    output_dir, f"year={yv}", f"month={mv}", f"x_part={xp}", f"y_part={yp}"
                )
            else:
                # year=YYYY/x_part=xxx/y_part=yyy
                subdir = os.path.join(output_dir, f"year={yv}", f"x_part={xp}", f"y_part={yp}")

            _ensure_dir(subdir)
            out_file = os.path.join(subdir, "data.parquet")

            # Convert chunk_df to Arrow and write
            arrow_table = pa.Table.from_pandas(chunk_df)
            pq.write_table(
                arrow_table,
                out_file,
                compression="snappy",
            )

            # We'll store an index entry so future reading scripts can skip globbing
            # Store relative path
            rel_path = os.path.relpath(out_file, output_dir)
            index_entry = {
                "year": int(yv),
                "month": int(mv) if mv is not None else None,
                "x_part": int(xp),
                "y_part": int(yp),
                "file_path": rel_path,
            }
            partition_index_new_entries.append(index_entry)
            rows_in_this_year += len(chunk_df)

        total_rows_for_all_years += len(df_year)

    # Write or update the partition_index.json
    index_path = os.path.join(output_dir, "partition_index.json")
    update_partition_index_file(index_path, partition_index_new_entries)

    log.info("Wrote %d total rows across all processed years.", total_rows_for_all_years)

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
    data_set["total_rows_ingested"] = total_rows_for_all_years

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    data_set["gridding_time"] = f"{hours:02}:{minutes:02}:{seconds:02}"

    meta_json_path = os.path.join(output_dir, "grid_meta.json")
    try:
        with open(meta_json_path, "w", encoding="utf-8") as f_meta:
            json.dump(data_set, f_meta, indent=2)
        log.info("Wrote data_set metadata to %s", meta_json_path)
    except OSError as exc:
        log.error("Failed to write grid_meta.json: %s", exc)

    data_set["grid_path"] = output_dir

    return data_set


def main(args):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert altimetry data into partitioned Parquet with ragged layout, storing each "
            "partition in a single file and recording the path in partition_index.json. "
            "Supports regridding everything or updating a single year."
        )
    )

    # Required arguments
    parser.add_argument(
        "--area",
        "-a",
        help=("Area name for data filtering/masking of input measurements. Example: greenland_is"),
        required=True,
    )

    parser.add_argument(
        "--binsize",
        "-b",
        help="grid binsize in m, default=5e3 == 5km",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--dataset",
        "-da",
        help="string identifying L2 data set: valid values are: cryotempo_c001",
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
            "use / to separate. example data/ku/elevation"
        ),
        required=True,
    )

    parser.add_argument(
        "--gridarea",
        "-g",
        help=(
            "output grid area specification name. Valid names in "
            "cpom.gridding.gridareas.py: GridArea definitions"
        ),
        required=True,
    )

    parser.add_argument(
        "--l2_dir",
        "-dir",
        help="L2 base directory to recursively search for L2 files using the search pattern",
        required=True,
    )

    parser.add_argument(
        "--lat_param",
        "-lat",
        help=(
            "name of latitude parameter to use in L2 netcdf files. If netcdf contains groups use / "
            "to separate. example data/ku/latitude"
        ),
        required=True,
    )

    parser.add_argument(
        "--lon_param",
        "-lon",
        help=(
            "name of longitude parameter to use in L2 netcdf files. If netcdf contains groups use /"
            "to separate. example data/ku/longitude"
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
            "Base path to create or update the gridded data set Parquet format files. "
            "The final parquet directory is formed from: <output_dir>/<mission>/"
            "<grid_name>_<binsize in km>km_<dataset>"
        ),
        required=True,
        type=str,
    )

    parser.add_argument(
        "--partition_by_month",
        "-pm",
        help=(
            "If set to True, the Parquet dataset will be partitioned by both year and month"
            " (in addition to coarser spatial partitions)."
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
            "partitions by dividing them by this factor. e.g. 20 => x_part=x_bin//20, "
            "y_part=y_bin//20."
        ),
        required=False,
        default=20,
    )

    parser.add_argument(
        "--power_param",
        "-power",
        help=(
            "name of power (backscatter) parameter to use in L2 netcdf files. If netcdf contains "
            "groups, use / to separate. example data/ku/backscatter"
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
            "groups, use / to separate. example data/ku/time"
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

    parser.add_argument(
        "--time_secs_to_1950",
        help=(
            "Offset (float) to add to the raw L2 'time' param so that time=0 => 1950-01-01. "
            "Default=0.0, but e.g. ~ -1577923200 if your data are 'since 2000-01-01'."
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
        "partition_xy_chunking": int(args.partition_xy_chunking),
        "time_secs_to_1950": float(args.time_secs_to_1950),
    }

    # Run gridding in the chosen mode
    grid_dataset(data_set, regrid=args.regrid, update_year=args.update_year)


if __name__ == "__main__":
    main(sys.argv[1:])

"""
cpom.altimetry.tools.sec_tools.grid_for_elevation_change_update_year.py

Purpose:
    Update a single year of gridded elevation change data in a Parquet dataset
    produced in cpom.altimetry.tools.sec_tools.grid_for_elev_change_regrid.py.
"""

import argparse
import json
import logging

from cpom.altimetry.datasets.dataset_helper import DatasetHelper
from cpom.altimetry.tools.sec_tools.grid_for_elev_change_regrid import (
    get_data_and_status_multiprocessed,
)
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers

log = logging.getLogger(__name__)


def get_set_up_objects(args):
    """
    Gets the output directory , area and grid objects based on command line parameters,
    and grid metadata JSON file.
    Validates if the output directory exists.
    Constructs a DatasetHelper object based on the metadata JSON file.

     Args:
         args (_type_): _description_
         logger (_type_): _description_

     Returns:
         _type_: _description_
    """
    # Load json metadata
    with open(args.grid_metadata, "r", encoding="utf-8") as f:
        grid_meta = json.load(f)

    args.full_output_dir = grid_meta["full_output_dir"]
    args.partition_columns = grid_meta["partition_columns"]
    args.partition_xy_chunking = grid_meta["partition_xy_chunking"]

    # Construct dataset object
    dataset = DatasetHelper(
        data_dir=grid_meta["base_dir"],
        mission=grid_meta["mission"],
        long_name=grid_meta["long_name"],
        dataset_epoch=grid_meta["dataset_epoch"],
        search_pattern=grid_meta["search_pattern"],
        yyyymm_str_fname_indices=grid_meta["yyyymm_str_fname_indices"],
        latitude_param=grid_meta["latitude_param"],
        longitude_param=grid_meta["longitude_param"],
        elevation_param=grid_meta["elevation_param"],
        time_param=grid_meta["time_param"],
        power_param=grid_meta["power_param"],
        mode_param=grid_meta["mode_param"],
    )

    thisgrid = GridArea(grid_meta["gridarea"], grid_meta["binsize"])
    thisarea = Area(grid_meta["area"])

    return dataset, thisgrid, thisarea, grid_meta


def update_metadata_json(grid_meta, args, status, logger):
    """Update the metadata JSON file with the new status information.
    for the year that has been reprocessed.
    Args:
        grid_meta (dict): The original grid metadata.
        status (dict): The new status information to update.
    """
    idx = grid_meta["years_ingested"].index(args.update_year)

    old_year_file_ingested = grid_meta["years_files_ingested"][idx]
    old_year_file_rejected = grid_meta["years_files_rejected"][idx]
    old_year_rows_ingested = grid_meta["years_rows_ingested"][idx]

    new_year_file_ingested = status["years_files_ingested"][0]
    new_year_file_rejected = status["years_files_rejected"][0]
    new_year_rows_ingested = status["years_rows_ingested"][0]

    grid_meta["years_files_ingested"][idx] = new_year_file_ingested
    grid_meta["years_files_rejected"][idx] = new_year_file_rejected
    grid_meta["years_rows_ingested"][idx] = new_year_rows_ingested

    grid_meta["total_l2_files_ingested"] = grid_meta["total_l2_files_ingested"] + (
        new_year_file_ingested - old_year_file_ingested
    )
    grid_meta["total_l2_files_rejected"] = grid_meta["total_l2_files_rejected"] + (
        new_year_file_rejected - old_year_file_rejected
    )
    grid_meta["total_rows_ingested"] = grid_meta["total_rows_ingested"] + (
        new_year_rows_ingested - old_year_rows_ingested
    )

    try:
        with open(args.grid_metadata, "w", encoding="utf-8") as f_meta:
            json.dump(grid_meta, f_meta, indent=2)
        logger.info("Wrote data_set metadata to %s", args.grid_metadata)
    except OSError as exc:
        logger.error("Failed to write grid_meta.json: %s", exc)


def update_year(args, logger):
    """
    Update a single year of gridded elevation change data.
    1. Get dataset, area and grid objects
    2. Get files/ dates from the filename in the year to update
    3. Get offset from the dataset
    4. Process year of data and write to Parquet files
    5. Update the metadata JSON file with the new year data
    Args:
        args (_type_): _description_
        logger (_type_): _description_
    """
    # --------------------------------------------------------#
    # 1. Get dataset, area and grid objects #
    # --------------------------------------------------------#
    dataset, thisgrid, thisarea, grid_meta = get_set_up_objects(args)
    # -----------------------------------------------------------#
    # 2. Get files/ dates from the filename in the year to update#
    #  3. Get offset from the dataset                            #
    # -----------------------------------------------------------#
    min_date = f"{args.update_year}0101"
    max_date = f"{args.update_year}1231"
    files_and_dates = dataset.get_files_and_dates(min_dt_time=min_date, max_dt_time=max_date)

    print(min_date, max_date)

    offset = dataset.get_unified_time_epoch_offset(
        grid_meta["standard_epoch"], grid_meta["dataset_epoch"]
    )

    print("Offset is ", offset)

    # --------------------------------------------------------#
    # 4. Process  year of data and write to Parquet files #
    # --------------------------------------------------------#
    status = get_data_and_status_multiprocessed(
        params=args,
        dataset=dataset,
        thisgrid=thisgrid,
        thisarea=thisarea,
        offset=offset,
        file_and_dates=files_and_dates,
        logger=logger,
    )
    print("DEBUG: get_data_and_status_multiprocessed returned:", type(status), status)
    # -----------------------------------------------------------------------
    # 5. Update the metadata JSON file with the new year data
    # -----------------------------------------------------------------------
    update_metadata_json(grid_meta, args, status, logger)


def main():
    """
    Main function to parse arguments and initiate the year update process.
    1. Parse command line arguments.
    2. Set up logging.
    3. Call the update_year function with parsed arguments and logger.
    """
    parser = argparse.ArgumentParser(
        description=("Update a single year of gridded elevation change data ")
    )
    parser.add_argument(
        "--griddir",
        type=str,
        required=True,
        help="Path to the directory containing the gridded dataset to update.",
    )

    parser.add_argument(
        "--grid_metadata", type=str, required=True, help="Path to the grid metadata JSON file."
    )
    parser.add_argument(
        "--update_year",
        type=int,
        required=True,
        help="Year to update in the dataset (e.g., 2020)",
    )

    args = parser.parse_args()

    default_log_level = logging.INFO
    logfile = "grid.log"
    set_loggers(
        log_file_info=logfile[:-3] + "info.log",
        log_file_warning=logfile[:-3] + "warning.log",
        log_file_error=logfile[:-3] + "errors.log",
        log_file_debug=logfile[:-3] + "debug.log",
        log_format="%(levelname)s : %(asctime)s %(name)s : %(message)s",
        default_log_level=default_log_level,
    )

    update_year(args, log)


log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()

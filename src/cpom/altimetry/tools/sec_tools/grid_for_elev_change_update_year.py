"""
cpom.altimetry.tools.sec_tools.grid_for_elev_change_update_year

Purpose:
    Update a single year of gridded elevation data in an existing Parquet dataset.

    Reprocesses one year of altimetry data and updates the corresponding partitions
    in a dataset previously created by grid_for_elev_change.py. Useful for adding
    new data or reprocessing specific years without regenerating the entire dataset.

Output:
    - Updated partitions: <grid_dir>/year=YYYY/**/*.parquet
    - Updated metadata: <grid_dir>/grid_meta.json
"""

import argparse
import json
import logging
from pathlib import Path

from cpom.altimetry.datasets.dataset_helper import DatasetHelper
from cpom.altimetry.tools.sec_tools.grid_for_elev_change import (
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
         args (Argparse.Namespace): Parsed command line arguments.
         logger (logging.Logger): Logger object for logging.

     Returns:
         tuple: Contains the dataset, grid, area, mask and grid metadata.
    """
    # Load json metadata
    with open(args.grid_info_json, "r", encoding="utf-8") as f:
        grid_meta = json.load(f)

    args.out_dir = grid_meta["out_dir"]
    args.partition_columns = grid_meta["partition_columns"]
    args.partition_xy_chunking = grid_meta["partition_xy_chunking"]
    args.fill_missing_poca = grid_meta["fill_missing_poca"]

    # Construct dataset object
    dataset = DatasetHelper(
        data_dir=grid_meta["data_input_dir"],
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
    thismask = Area(grid_meta["mask"])

    return dataset, thisgrid, thisarea, thismask, grid_meta


def update_metadata_json(grid_meta, args, status, logger):
    """Update the metadata JSON file with the new status information.
    for the year that has been reprocessed.
    Args:
        grid_meta (dict): The original grid metadata.
        status (dict): The new status information to update.
    """

    new_year_file_ingested = status["years_files_ingested"][0]
    new_year_file_rejected = status["years_files_rejected"][0]
    new_year_rows_ingested = status["years_rows_ingested"][0]

    if args.update_year not in grid_meta["years_ingested"]:
        # Add new year to the metadata
        grid_meta["years_ingested"].append(args.update_year)
        grid_meta["years_files_ingested"].append(new_year_file_ingested)
        grid_meta["years_files_rejected"].append(new_year_file_rejected)
        grid_meta["years_rows_ingested"].append(new_year_rows_ingested)
        grid_meta["total_l2_files_ingested"] = (
            grid_meta["total_l2_files_ingested"] + new_year_file_ingested
        )
        grid_meta["total_l2_files_rejected"] = (
            grid_meta["total_l2_files_rejected"] + new_year_file_rejected
        )
        grid_meta["total_rows_ingested"] = grid_meta["total_rows_ingested"] + new_year_rows_ingested

    else:
        # Update existing year in the metadata
        idx = grid_meta["years_ingested"].index(args.update_year)

        old_year_file_ingested = grid_meta["years_files_ingested"][idx]
        old_year_file_rejected = grid_meta["years_files_rejected"][idx]
        old_year_rows_ingested = grid_meta["years_rows_ingested"][idx]

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
        with open(args.grid_info_json, "w", encoding="utf-8") as f_meta:
            json.dump(grid_meta, f_meta, indent=2)
        logger.info("Wrote data_set metadata to %s", args.grid_info_json)
    except OSError as exc:
        logger.error("Failed to write metadata.json: %s", exc)


def update_year(args):
    """
    Update a single year of gridded elevation change data.
    1. Get dataset, area and grid objects
    2. Get files/ dates from the filename in the year to update
    3. Get offset from the dataset
    4. Process year of data and write to Parquet files
    5. Update the metadata JSON file with the new year data
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    # --------------------------------------------------------#
    # 1. Get dataset, area and grid objects #
    # --------------------------------------------------------#
    dataset, thisgrid, thisarea, thismask, grid_meta = get_set_up_objects(args)
    # -----------------------------------------------------------#
    # 2. Get files/ dates from the filename in the year to update#
    #  3. Get offset from the dataset                            #
    # -----------------------------------------------------------#
    min_date = f"{args.update_year}0101"
    max_date = f"{args.update_year}1231"
    files_and_dates = dataset.get_files_and_dates(min_dt_time=min_date, max_dt_time=max_date)

    logger = set_loggers(
        log_file_info=Path(grid_meta["out_dir"]) / "info.log",
        log_file_error=Path(grid_meta["out_dir"]) / "errors.log",
    )

    if len(files_and_dates) == 0:
        logger.error("No files found for year %s", args.update_year)
        return

    offset = dataset.get_unified_time_epoch_offset(
        grid_meta["standard_epoch"], grid_meta["dataset_epoch"]
    )

    # --------------------------------------------------------#
    # 4. Process  year of data and write to Parquet files #
    # --------------------------------------------------------#
    status = get_data_and_status_multiprocessed(
        params=args,
        dataset=dataset,
        thisgrid=thisgrid,
        thisarea=thisarea,
        thismask=thismask,
        offset=offset,
        file_and_dates=files_and_dates,
        logger=logger,
    )
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
        "--grid_info_json", type=str, required=True, help="Path to the grid metadata JSON file."
    )
    parser.add_argument(
        "--update_year",
        type=int,
        required=True,
        help="Year to update in the dataset (e.g., 2020)",
    )

    args = parser.parse_args()
    update_year(args)


log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()

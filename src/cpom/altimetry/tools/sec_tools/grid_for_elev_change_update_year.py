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
import os
import sys
from functools import partial
from pathlib import Path

from cpom.altimetry.datasets.dataset_helper import DatasetHelper
from cpom.altimetry.tools.sec_tools.grid_for_elev_change import (
    get_data_and_status_multiprocessed,
    process_file,
)
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers
from cpom.masks.masks import Mask

log = logging.getLogger(__name__)


def parse_arguments(args):
    """
    Parse command line arguments for updating an altimetry grid.

    Return:
        parser.parse_args(args)
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
    parser.add_argument(
        "--correction_function",
        type=str,
        default="default_corrections",
        help="Correction function name from grid_for_elev_change_corrections.py.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Maximum worker processes for multiprocessing.",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=60,
        help="Number of files to process before writing to disk",
    )
    parser.add_argument(
        "--add_vars",
        type=json.loads,
        default="{}",
        help="Optional extra variables as JSON dict, "
        'e.g. {"tide_ocean":"land_ice_segments/geophysical/tide_ocean"}',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging and collect per-file processing stats",
    )
    return parser.parse_args(args)


def get_set_up_objects(args):
    """
    Reconstruct command line argumets from grid metadata JSON file.
    Reconstruct DatasetHelper, GridArea, Area and Mask objects.
    Validates if the output directory exists.

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
    thismask = Mask(grid_meta["mask_name"])

    return dataset, thisgrid, thisarea, thismask, grid_meta


def update_metadata_json(grid_meta, args, status, logger):
    """
    Update the metadata JSON file with the new status information.
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


# ---------------#
# Main Function #
# ---------------#
def grid_for_elev_change_update_year(args):
    """
    Main function to parse arguments and initiate the year update process.
        1. Parse command line arguments.
        2. Set up logging.
        3. Get dataset, area and grid objects
        4. Get files/ dates from the filename in the year to update
        5. Get offset from the dataset
        6. Process year of data and write to Parquet files
        7. Update the metadata JSON file with the new year data
    """
    params = parse_arguments(args)

    # --------------------------------------------------------#
    # 1. Get dataset, area and grid objects #
    # --------------------------------------------------------#
    dataset, thisgrid, thisarea, thismask, grid_meta = get_set_up_objects(params)
    # -----------------------------------------------------------#
    # 2. Get files/ dates from the filename in the year to update#
    #  3. Get offset from the dataset                            #
    # -----------------------------------------------------------#
    min_date = f"{params.update_year}0101"
    max_date = f"{params.update_year}1231"
    files_and_dates = dataset.get_files_and_dates(min_dt_time=min_date, max_dt_time=max_date)

    log_path = Path(grid_meta["out_dir"]) / f"update_{params.update_year}_logs"
    log_path.mkdir(parents=True, exist_ok=True)
    logger = set_loggers(
        log_dir=str(log_path),
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )
    logger.info("output_dir=%s", params.out_dir)

    if len(files_and_dates) == 0:
        logger.error("No files found for year %s", params.update_year)
        return

    offset = dataset.get_unified_time_epoch_offset(
        grid_meta["standard_epoch"], grid_meta["dataset_epoch"]
    )
    worker = partial(
        process_file,
        dataset=dataset,
        offset=offset,
        this_grid=thisgrid,
        this_area=thisarea,
        this_mask=thismask,
        params=params,
    )
    # --------------------------------------------------------#
    # 4. Process  year of data and write to Parquet files #
    # --------------------------------------------------------#
    status = get_data_and_status_multiprocessed(
        params=params,
        file_and_dates=files_and_dates,
        worker=worker,
        logger=logger,
    )
    # -----------------------------------------------------------------------
    # 5. Update the metadata JSON file with the new year data
    # -----------------------------------------------------------------------
    update_metadata_json(grid_meta, params, status, logger)


if __name__ == "__main__":
    grid_for_elev_change_update_year(sys.argv[1:])

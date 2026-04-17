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
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    get_metadata_params,
    get_metadata_path,
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
        "--in_step",
        help="Input algorithm step to source metadata from",
        type=str,
        required=False,
        default="grid_for_elev_change",
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        required=False,
        help="Directory containing the existing grid metadata file.",
    )
    parser.add_argument(
        "--in_meta",
        type=str,
        required=False,
        help="Explicit metadata JSON path or directory containing <in_step>_meta.json.",
    )
    parser.add_argument(
        "--update_year",
        type=int,
        required=True,
        help="Year to update in the dataset (e.g., 2020)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Maximum worker processes for multiprocessing.",
    )
    parser.add_argument(
        "--hive_mode",
        action="store_true",
        help="Writes partitions slower, "
        "but lower memory footprint. May be needed for antarctica is2",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging",
    )
    return parser.parse_args(args)


def get_set_up_objects(args):
    """
    Reconstruct command line argumets from grid metadata JSON file.
    Reconstruct DatasetHelper, GridArea, Area and Mask objects.
    Validates if the output directory exists.

    Args:
        args (Argparse.Namespace): Parsed command line arguments.

    Returns:
        tuple: Contains the dataset, grid, area, mask and grid metadata.
    """
    grid_meta = get_metadata_params(args, "all", "grid_for_elev_change")

    for key, value in grid_meta.items():
        args.__setattr__(key, value)

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

    year_label = str(args.update_year)
    existing_year_labels = [str(y) for y in grid_meta["years_ingested"]]

    if year_label not in existing_year_labels:
        # Add new year to the metadata
        grid_meta["years_ingested"].append(year_label)
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
        idx = existing_year_labels.index(year_label)

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
    metadata_path = get_metadata_path(args, basin_name=None, logger=logger)
    if metadata_path is None:
        raise ValueError("Could not resolve metadata path for updating grid metadata.")

    metadata_file = Path(metadata_path)

    try:
        with open(metadata_file, "r", encoding="utf-8") as handle:
            entry_store = json.load(handle)
    except OSError as exc:
        logger.error("Failed to read metadata.json: %s", exc)
        raise

    matching_keys = [key for key in entry_store if key.startswith("grid_for_elev_change_")]
    if not matching_keys:
        raise ValueError(f"No grid_for_elev_change entry found in {metadata_file}")

    latest_key = max(matching_keys)
    entry_store[latest_key] = grid_meta

    try:
        with open(metadata_file, "w", encoding="utf-8") as handle:
            json.dump(entry_store, handle, indent=2)
            handle.write("\n")
        logger.info("Updated grid metadata counts in %s", metadata_file)
    except OSError as exc:
        logger.error("Failed to write metadata.json: %s", exc)
        raise


# ---------------#
# Main Function #
# ---------------#
def grid_for_elev_change_update_year(
    args,
):
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
    args = parse_arguments(args)

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
        log_dir=grid_meta["out_dir"],
        default_log_level=logging.DEBUG if args.debug else logging.INFO,
    )

    logger.info("Found %d files for year %s", len(files_and_dates), args.update_year)
    if len(files_and_dates) == 0:
        logger.error("No files found for year %s", args.update_year)
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
        params=args,
    )
    # --------------------------------------------------------#
    # 4. Process  year of data and write to Parquet files #
    # --------------------------------------------------------#
    status = get_data_and_status_multiprocessed(
        params=args,
        file_and_dates=files_and_dates,
        worker=worker,
        logger=logger,
    )
    # -----------------------------------------------------------------------
    # 5. Update the metadata JSON file with the new year data
    # -----------------------------------------------------------------------
    update_metadata_json(grid_meta, args, status, logger)


log = logging.getLogger(__name__)

if __name__ == "__main__":
    grid_for_elev_change_update_year(sys.argv[1:])

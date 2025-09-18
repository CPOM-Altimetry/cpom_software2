"""
cpom.altimetry.tools.sec_tools.grid_for_elevation_change.py

Purpose:
  Grid altimetry data for SEC (model fit) processing,
  storing each measurement in a partitioned Parquet dataset,
  partitioned by (year, x_part, y_part) or (year, month, x_part, y_part)
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from netCDF4 import Dataset  # pylint: disable=E0611

from cpom.altimetry.datasets.dataset_helper import DatasetHelper
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers

log = logging.getLogger(__name__)


def get_set_up_objects(params, dataset, logger, confirm_regrid=True):
    """
    Gets the output directory , area and grid objects based on command line parameters.
    Validates if the output directory exists and prompts the user for confirmation
    of removal.
    Args:
        params (argparse.Namespace): Command line parameters
        dataset (DatasetHelper): Dataset configuration object
        logger (logging.Logger): Kogger object
        confirm_regrid (bool): Boolean flag to prompt user for regridding.
    Returns:
        Tuple: Configuration dict, grid area, and area objects
    """

    params.full_output_dir = os.path.join(
        params.griddir,
        dataset.mission,
        f"{params.gridarea}_{int(params.binsize/1000)}km_{dataset.mission}",
    )

    # Full regrid => remove entire directory, then create fresh
    if os.path.exists(params.full_output_dir):
        if (
            params.full_output_dir != "/" and dataset.mission in params.full_output_dir
        ):  # safety check
            logger.info("Removing previous grid dir: %s ...", params.full_output_dir)
            if confirm_regrid:
                response = (
                    input("Confirm removal of previous grid archive? (y/n): ").strip().lower()
                )
                if response == "y":
                    shutil.rmtree(params.full_output_dir)
                else:
                    print("Exiting as user requested not to overwrite grid archive")
                    sys.exit(0)
        else:
            logger.error("Invalid output_dir path: %s", params.full_output_dir)
            sys.exit(1)
    os.makedirs(params.full_output_dir, exist_ok=True)

    logger.info("output_dir=%s", params.full_output_dir)

    thisgrid = GridArea(params.gridarea, params.binsize)
    thisarea = Area(params.area)

    return params, thisgrid, thisarea


def fill_missing_poca_with_nadir_fdr4alt(
    dataset, nc, lats: np.ndarray, lons: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill missing POCA lat/lons with nadir values, where the 'expert/ice_sheet_qual_relocation'
    is > 0. This function is specific to the FDR4ALT dataset.

    Args:
        dataset (Dataset): The dataset configuration.
        nc (Dataset or h5py.File): Opened NetCDF or HDF5 file.
        lats (np.ndarray): Array of latitudes.
        lons (np.ndarray): Array of longitudes.

    Returns:
        tuple(np.ndarray): Updated lat/lons filled with Nadir
    """
    # check if expert/ice_sheet_qual_relocation exists
    if "expert/ice_sheet_qual_relocation" not in nc.variables:
        return lats, lons

    latitude_nadir = dataset.get_variable(nc, "latitude_nadir")
    longitude_nadir = dataset.get_variable(nc, "longitude_nadir")
    if lats is None and lons is None:
        lats = dataset.get_variable(nc, "latitude")
        lons = dataset.get_variable(nc, "longitude") % 360.0

    relocation_failure = dataset.get_variable(nc, "expert/ice_sheet_qual_relocation")
    lats = np.where(relocation_failure > 0, latitude_nadir, lats)
    lons = np.where(relocation_failure > 0, longitude_nadir, lons)

    return lats, lons


# pylint: disable=R0917, R0913 , R0914 , R0915, R0912
def process_file(
    file_and_date: dict,
    dataset: argparse.Namespace,
    offset: float,
    this_grid: GridArea,
    this_area: Area,
) -> pl.LazyFrame | None:
    """
    Extract and process data from a single altimetry file.
    To be run in parallel in get_data_and_status_multiprocessed.

    Args:
        file_and_date (Tuple): Tuple containing file path and date information
        data_set (Dict): Dataset configuration dictionary defined in YAML
        this_grid (GridArea): CPOM grid area object.
        this_area (Area): CPOM area object.
        offset (float): Time offset to apply to time data

    Returns:
        pl.LazyFrame: A Polars LazyFrame containing processed data, or None if no valid data
    """

    def _get_variable(dataset, nc, var_name, context_manager=Dataset):
        """Extract a variable from the dataset.
        If the dataset is a NetCDF file,
        check the _FillValue attribute and replace fill values with NaN.

        Args:
            dataset (argparse.Namespace): The dataset configuration.
            nc (Dataset or h5py.File): The opened NetCDF or HDF5 file.
            var_name (str): The name of the variable to extract.
            context_manager (type, optional): The context manager to use.
                                                Defaults to context_manager.

        Returns:
            np.ndarray: Data array for the variable, or None if not found.
        """

        data = dataset.get_variable(nc, var_name)
        if context_manager == Dataset:
            try:
                fill_value = nc[var_name]._FillValue  # pylint: disable=W0212
                data[data == fill_value] = np.nan
            except (KeyError, AttributeError):
                pass
        return data

    # Deconstruct file and date tuple
    file_path = file_and_date["path"]
    year_val = file_and_date["year"]
    month_val = file_and_date["month"]

    # Get filetype from search pattern
    context_manager = Dataset
    if dataset.search_pattern.endswith(".h5"):
        context_manager = h5py.File
    with context_manager(file_path) as nc:

        #  Get lat/lons and mask to area.
        if dataset.beams != []:
            lats, beams = dataset.get_variable(nc, dataset.latitude_param, return_beams=True)
        else:
            lats = dataset.get_variable(nc, dataset.latitude_param)
        lons = dataset.get_variable(nc, dataset.longitude_param) % 360.0

        # lats, lons = fill_missing_poca_with_nadir_fdr4alt(dataset, nc, lats, lons)
        area_mask, n_inside = this_area.inside_area(lats, lons)
        if n_inside < 2:  # Number of points needed to get the heading
            print(f"No points in area for file {file_path}, skipping...")
            return None

        elevs = _get_variable(dataset, nc, dataset.elevation_param)
        bool_mask = area_mask & np.isfinite(elevs)

        # Get quality mask
        if (
            dataset.beams != []
            and dataset.quality_param == "land_ice_segments/atl06_quality_summary"
        ):
            bool_mask &= dataset.get_variable(nc, dataset.quality_param) == 0
        if dataset.quality_param is not None:
            bool_mask &= dataset.get_variable(nc, dataset.quality_param) == 0

        if dataset.power_param is not None:
            pwr = _get_variable(dataset, nc, dataset.power_param)
            bool_mask &= np.isfinite(pwr)
        if sum(bool_mask) < 2:
            print(f"No valid points in area for file {file_path}, skipping...")
            return None

        if dataset.uncertainty_param is not None:
            uncert = _get_variable(dataset, nc, dataset.uncertainty_param)

        # Get time variable and apply offset
        t = _get_variable(dataset, nc, dataset.time_param) + offset
        bool_mask &= np.isfinite(t)

        ascending_points = dataset.get_file_orbital_direction(
            latitude=(
                dataset.get_variable(nc, dataset.latitude_nadir_param)
                if dataset.latitude_nadir_param is not None
                else lats
            ),
            nc=nc,
        )[bool_mask]

        # Get grid cell and offsets for each point
        x_coords, y_coords = this_area.latlon_to_xy(lats[bool_mask], lons[bool_mask])
        x_bin, y_bin = this_grid.get_col_row_from_x_y(x_coords, y_coords)
        xoffset, yoffset = this_grid.get_xy_relative_to_cellcentres(x_coords, y_coords)

        data_dict = {
            "time": t[bool_mask],
            "year": year_val,
            "month": month_val,
            "x_bin": x_bin,
            "y_bin": y_bin,
            "x_cell_offset": xoffset,
            "y_cell_offset": yoffset,
            "x": x_coords,
            "y": y_coords,
            "elevation": elevs[bool_mask],
            "ascending": ascending_points,
        }

        if dataset.power_param is not None:
            data_dict["power"] = pwr[bool_mask]

        if dataset.mode_param is not None:
            mode = dataset.get_variable(nc, dataset.mode_param)
            data_dict["mode"] = mode[bool_mask]

        if dataset.beams != []:
            data_dict["beam"] = beams[bool_mask]

        if dataset.uncertainty_param is not None:
            data_dict["uncertainty"] = uncert[bool_mask]

        df_meas = (
            pl.LazyFrame(data_dict)
            .cast({"elevation": pl.Float64})
            .filter((pl.col("elevation") < 6000) & (pl.col("elevation") > -500))
        )

        return df_meas


def get_data_and_status_multiprocessed(
    params, dataset, thisgrid, thisarea, offset, file_and_dates, logger
):
    """
    Extract and process data from altimetry netcdf or hdf5 files
    in parallel and write the results to disk as a partitioned parquet file.

    Uses multiprocessing and writes on year of data at a time.
    Calls :
        process_file: Function to process a single file.
        write_partitions_to_disk: Function to write processed data to disk.

    Args:
        dataset (Dataset): CPOM Dataset Object
        thisgrid (GridArea): GridArea Object
        thisarea (Area): Area Object
        offset (float): Offset in seconds between the datasets epoch and the epoch used for the grid
        logger (Logger): Logger Object
        file_and_dates (np.ndarray): Array of file paths and dates
        params (argparse.Namespace): Command line parameters

    Returns:
        dict: Processing status dictionary, containing information about the files processed.
    """
    status = {
        "years_ingested": [],
        "years_files_ingested": [],
        "years_files_rejected": [],
        "years_rows_ingested": [],
        "total_l2_files_rejected": 0,
        "total_l2_files_ingested": 0,
        "total_rows_ingested": 0,
    }

    worker = partial(
        process_file,
        dataset=dataset,
        offset=offset,
        this_grid=thisgrid,
        this_area=thisarea,
    )
    # ----------------------------------------------------------------------
    # Process each year
    # ----------------------------------------------------------------------

    # Get list of years from file_and_date from a tuple of (file_path, date)
    # Extract unique years from the file_and_dates array
    years = sorted(set(file_and_dates["year"]))

    try:
        with ProcessPoolExecutor() as executor:
            for year in years:
                log.info("Processing year: %s", year)
                big_rows_list = []
                file_and_date_year = file_and_dates[file_and_dates["year"] == year]
                chunksize = max(15, len(file_and_date_year) // (10 * 4))
                results = executor.map(worker, file_and_date_year, chunksize=chunksize)
                big_rows_list = [r for r in results if r is not None]
                print(f"Collected {len(big_rows_list)} valid results for year {year}")

                if len(big_rows_list) == 0:
                    print(f"No valid data for year {year}, skipping...")
                    status["years_files_rejected"].append(len(file_and_date_year))
                    status["total_l2_files_rejected"] += len(file_and_date_year)
                    continue

                final = pl.concat(big_rows_list).collect()
                print(f"Final DataFrame shape: {final.shape}")

                total_rows = write_partitions_to_disk(
                    final_df=final,
                    partition_columns=params.partition_columns,
                    partition_xy_chunking=params.partition_xy_chunking,
                    full_output_dir=params.full_output_dir,
                )

                status["years_ingested"].append(int(year))
                status["years_files_ingested"].append(len(big_rows_list))
                status["years_files_rejected"].append(len(file_and_date_year) - len(big_rows_list))
                status["years_rows_ingested"].append(total_rows)
                status["total_l2_files_ingested"] += len(big_rows_list)
                status["total_l2_files_rejected"] += len(file_and_date_year) - len(big_rows_list)
                status["total_rows_ingested"] += total_rows
    except (OSError, ValueError) as e:
        logger.error("Multiprocessing failed, exiting... %s", e)
        sys.exit(1)

    return status


def write_partitions_to_disk(final_df, partition_columns, partition_xy_chunking, full_output_dir):
    """
    Write  DataFrame of altimetry data to disk.
    Partitioned by ('year', 'month', 'x_part', 'y_part') or
    ('year', 'x_part', 'y_part')

    Args:
        final_df (DataFrame): The final DataFrame to write.
        partition_columns (list): List of columns to partition by.
        partition_xy_chunking (int): Chunking factor for spatial partitioning.
        full_output_dir (str): Full path to the output directory.

    Returns:
        int: Total number of rows written to disk.
    """
    if "x_part" in partition_columns and "y_part" in partition_columns:
        final_df = final_df.with_columns(
            (pl.col("x_bin") / partition_xy_chunking).cast(pl.Int64).alias("x_part"),
            (pl.col("y_bin") / partition_xy_chunking).cast(pl.Int64).alias("y_part"),
        )

    partitions = final_df.partition_by(partition_columns, as_dict=True)

    total_rows = 0
    for key, group in partitions.items():
        subdir = os.path.join(
            full_output_dir,
            *[f"{group}={str(i)}" for group, i in zip(partition_columns, key)],
        )
        if not os.path.isdir(subdir):
            os.makedirs(subdir, exist_ok=True)

        outfile = os.path.join(subdir, "data.parquet")
        pq.write_table(
            pa.Table.from_pandas(group.to_pandas()),
            outfile,
            compression="zstd",
        )
        # log.info("Written: %s", outfile)
        total_rows += len(group)

    return total_rows


def get_metadata_json(params, dataset, status, thisgrid, start_time):
    """Writes a metadata JSON file with gridding details to the output directory.
    Metadata includes:
        - command line parameters
        - dataset
        - processing status
        - grid area details
        - processing time

    Args:
        params (argparse.Namespace): Command line parameters
        dataset (dict): Dataset configuration
        config (dict): Gridding configuration
        status (dict): Processing status dictionary
        thisgrid (GridArea): CPOM GridArea object
        start_time (float): Processing start time
    """
    # get dict from command line parameters
    ds_dict = vars(dataset)
    params_dict = vars(params)

    if ".yml" in params.dataset or ".yaml" in params.dataset:
        output = {
            **{k: v for k, v in params_dict.items() if k not in ds_dict.keys()},
            **ds_dict,
            **status,
        }
    else:
        output = {**params_dict, **status}

    output["grid_xmin"] = thisgrid.minxm
    output["grid_ymin"] = thisgrid.minym
    output["grid_crs"] = thisgrid.coordinate_reference_system
    output["grid_x_size"] = thisgrid.grid_x_size
    output["grid_y_size"] = thisgrid.grid_y_size

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    output["gridding_time"] = f"{hours:02}:{minutes:02}:{seconds:02}"

    # hours, remainder = divmod(int(compaction_time), 3600)
    # minutes, seconds = divmod(remainder, 60)
    # output["compaction_time"] = f"{hours:02}:{minutes:02}:{seconds:02}"

    meta_json_path = os.path.join(params.full_output_dir, "grid_meta.json")
    try:
        with open(meta_json_path, "w", encoding="utf-8") as f_meta:
            json.dump(output, f_meta, indent=2)
        log.info("Wrote data_set metadata to %s", meta_json_path)
    except OSError as exc:
        log.error("Failed to write grid_meta.json: %s", exc)


def main(args):
    """Main function to grid altimetry data for elevation change processing.
    1. Load command line arguments
    2. Load dataset object
    3. Get grid and area objects
    4. Get files/ dates and offset from dataset
    5. Process each year of data and write to Parquet files
    6. Write metadata JSON file with gridding details
    """
    # ------------------------------#
    # 1.Load command line arguments#
    # ------------------------------#
    parser = argparse.ArgumentParser(
        description="""Convert altimetry data into partitioned parquet
        with ragged layout, storing each partition in a single file"""
    )
    parser.add_argument("--base_dir", type=str, required=True, help="Directory containing L2 files")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset definition YAML file"
        " or inline dataset config as JSON string defined in the rcf configuration yml",
    )
    parser.add_argument("--griddir", type=str, required=True, help="Output directory")
    parser.add_argument("--area", type=str, required=True, help="CPOM area object")
    parser.add_argument("--gridarea", type=str, required=True, help="CPOM grid area object")
    parser.add_argument("--binsize", type=int, default=5000, help="Grid bin size in meters")
    parser.add_argument(
        "--standard_epoch",
        type=str,
        default="1991-01-01T00:00:00",
        help="Standard epoch for time conversion (e.g., '1991-01-01T00:00:00')",
    )
    parser.add_argument(
        "--partition_columns",
        type=str,
        nargs="+",
        default=["year", "x_part", "y_part"],
        help="Columns to partition data by, must include 'year', 'x_part', 'y_part'",
    )
    parser.add_argument(
        "--partition_xy_chunking",
        help=(
            "This parameter sets the chunking factor for spatial partitioning. "
            "Individual grid cells (identified by x_bin and y_bin) are grouped into larger "
            "partitions by dividing them by this factor. e.g. 20 => x_part=x_bin//20, "
            "y_part=y_bin//20."
        ),
        type=int,
        default=20,
    )

    args = parser.parse_args(args)
    start_time = time.time()
    # ------------------------------#
    # 2. Load Dataset Object
    # ------------------------------#
    dataset_path = Path(args.dataset)
    if dataset_path.exists() and dataset_path.suffix in [".yml", ".yaml"]:
        # Method 1: Load from YAML file
        dataset = DatasetHelper(data_dir=args.base_dir, dataset_yaml=args.dataset)
    else:
        # Method 2: Treat as JSON string (inline config)
        try:
            dataset_config = json.loads(args.dataset)
            dataset = DatasetHelper(data_dir=args.base_dir, **dataset_config)
        except json.JSONDecodeError:
            parser.error(
                f"Invalid dataset configuration: {args.dataset}. "
                f"Must be either a path to a YAML file or a valid JSON string"
            )

    # Setup logging
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
    # -----------------------------#
    # 3. Get grid and area objects #
    # -----------------------------#
    args, thisgrid, thisarea = get_set_up_objects(args, dataset, log, confirm_regrid=True)
    # --------------------------------------------#
    # 4.Get files/ dates and offset from dataset#
    # -------------------------------------------#
    files_and_dates = dataset.get_files_and_dates(hemisphere=thisarea.hemisphere)
    # Get the number of seconds between the epoch to be used in the grid and the dataset epoch
    offset = dataset.get_unified_time_epoch_offset(args.standard_epoch, dataset.dataset_epoch)
    # --------------------------------------------------------#
    # 5. Process each year of data and write to Parquet files #
    # --------------------------------------------------------#
    status = get_data_and_status_multiprocessed(
        params=args,
        dataset=dataset,
        thisgrid=thisgrid,
        thisarea=thisarea,
        offset=offset,
        file_and_dates=files_and_dates,
        logger=log,
    )

    # --------------------------------------------------#
    # 6. Write metadata JSON file with gridding details #
    # --------------------------------------------------#
    get_metadata_json(args, dataset, status, thisgrid, start_time)


if __name__ == "__main__":
    # Parse command line arguments
    main(sys.argv[1:])

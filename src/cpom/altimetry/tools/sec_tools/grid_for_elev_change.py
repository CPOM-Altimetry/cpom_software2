"""
cpom.altimetry.tools.sec_tools.grid_for_elevation_change.py

Purpose:
    Grid altimetry data for SEC (model fit) processing.
    Stores data in a ragged layout,  partitioned by (year, x_part, y_part)
    or (year, month, x_part, y_part).
    Each x_part/y_part is a group of grid cells. The size defined by
    partition_xy_chunking parameter.

Outputs:
    - Partitioned Parquet files in out_dir, with ragged layout.
    - Each partition stored in a single Parquet file:
        <out_dir>/year=YYYY/x_part=XX/y_part=YY/data.parquet
        or
        <out_dir>/year=YYYY/month=MM/x_part=XX/y_part=YY/data.parquet
    - Metadata: <out_dir>/metadata.json

Parameters:
    --data_input_dir: Directory containing L2 files
    --dataset: Path to dataset definition YAML file
                or inline dataset config as JSON string.
    --out_dir: Output directory
    --area: CPOM area object
    --gridarea: CPOM grid area object
    --binsize: Grid bin size in meters
    --standard_epoch: Standard epoch for time conversion (e.g., '1991-01-01T00:00:00')
    --partition_columns: Columns to partition data by, must include 'year', 'x_part', 'y_part'
    --partition_xy_chunking: Chunking factor for spatial partitioning
    --fill_missing_poca: Fill missing POCA lat/lons with nadir values.
    Set True for FDR4ALT datasets.
"""

import argparse
import json
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
from cpom.masks.masks import Mask

# --------------------------------------#
# Set up Functions                     #
# --------------------------------------#


def parse_arguments(args):
    """
    Parse command line arguments for gridding altimetry data.

    Return:
        parser.parse_args(args)
    """
    parser = argparse.ArgumentParser(
        description="""Convert altimetry data into partitioned parquet
        with ragged layout, storing each partition in a single file"""
    )
    parser.add_argument(
        "--data_input_dir", type=str, required=True, help="Directory containing L2 files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset definition YAML file"
        " or inline dataset config as JSON string defined in the rcf configuration yml",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--area", type=str, required=True, help="CPOM area object")
    parser.add_argument(
        "--mask_name",
        type=str,
        required=False,
        help="Optional Mask Name to apply instead of Area Mask",
    )
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
    parser.add_argument(
        "--fill_missing_poca",
        action="store_true",
        help="Fill missing POCA lat/lons with nadir values, "
        "where the 'expert/ice_sheet_qual_relocation' is > 0. "
        "This is specific to the FDR4ALT dataset.",
    )

    return parser.parse_args(args)


def clean_directory(params: argparse.Namespace, dataset: DatasetHelper, confirm_regrid=False):
    """
    Validate and clear the output directory if it exists. Prompt user for confirmation
    if confirm_regrid is True.

    Args:
        params (argparse.Namespace): Command line parameters
        dataset (DatasetHelper): Dataset configuration object
        confirm_regrid (bool): Boolean flag to prompt user for regridding.

    """
    # Full regrid => remove entire directory, then create fresh
    if os.path.exists(params.out_dir):
        if params.out_dir != "/" and dataset.mission in params.out_dir:  # safety check
            if confirm_regrid is True:
                response = (
                    input("Confirm removal of previous grid archive? (y/n): ").strip().lower()
                )
                if response == "y":
                    shutil.rmtree(params.out_dir)
                else:
                    print("Exiting as user requested not to overwrite grid archive")
                    sys.exit(0)
            else:
                shutil.rmtree(params.out_dir)

        else:
            sys.exit(1)
    os.makedirs(params.out_dir, exist_ok=True)


def get_processing_objects(params, dataset):
    """
    Get processing objects: grid area, area, mask, logger, and construct
    worker partial function for multiprocessing.

    Get file_and_dates from the dataset with a list of all netcdf files and their associated dates.

    Args:
        params (argparse.Namespace): Command line parameters
        dataset (DatasetHelper): Dataset configuration object
    Returns:
        tuple: file_and_dates (dict),
        worker (partial function),
        thisgrid (GridArea),
        logger
    """

    logger = set_loggers(
        log_file_info=Path(params.out_dir) / "info.log",
        log_file_error=Path(params.out_dir) / "errors.log",
        log_file_warning=Path(params.out_dir) / "warnings.log",
    )

    logger.info("output_dir=%s", params.out_dir)

    thisgrid = GridArea(params.gridarea, params.binsize)
    thisarea = Area(params.area)
    if params.mask_name:
        thismask = Mask(params.mask_name)
    else:
        thismask = None

    file_and_dates = dataset.get_files_and_dates(hemisphere=thisarea.hemisphere)

    # Get the number of seconds between the epoch to be used in the grid and the dataset epoch
    offset = dataset.get_unified_time_epoch_offset(params.standard_epoch, dataset.dataset_epoch)

    worker = partial(
        process_file,
        dataset=dataset,
        offset=offset,
        this_grid=thisgrid,
        this_area=thisarea,
        this_mask=thismask,
        fill_missing_poca=params.fill_missing_poca,
    )

    return file_and_dates, worker, thisgrid, logger


# --------------------------------------#
# Data Loading and Processing Functions #
# --------------------------------------#


def _fill_missing_poca_with_nadir_fdr4alt(
    dataset: DatasetHelper, nc, lats: np.ndarray, lons: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    FDR4ALT specific function to:
        Fill missing POCA lat/lons with nadir values, where the 'expert/ice_sheet_qual_relocation'
    is > 0.

    Args:
        dataset (DatasetHelper): The dataset configuration.
        nc (netcdf Dataset or h5py.File): Opened NetCDF or HDF5 file.
        lats (np.ndarray): Array of latitudes.
        lons (np.ndarray): Array of longitudes.

    Returns:
        tuple(np.ndarray): Filled latitudes and longitudes arrays.
    """
    try:
        # Check if nadir parameters are configured
        if dataset.latitude_nadir_param is None or dataset.longitude_nadir_param is None:
            return lats, lons

        # Try to get the relocation failure variable
        relocation_failure = dataset.get_variable(
            nc, "expert/ice_sheet_qual_relocation", replace_fill=False
        )

        # Get nadir coordinates
        latitude_nadir = dataset.get_variable(nc, dataset.latitude_nadir_param)
        longitude_nadir = dataset.get_variable(nc, dataset.longitude_nadir_param)

        # Fill missing POCA locations with nadir values
        lats = np.where(relocation_failure > 0, latitude_nadir, lats)
        lons = np.where(relocation_failure > 0, longitude_nadir, lons)

        return lats, lons

    except (KeyError, ValueError, AttributeError) as e:
        # Variable doesn't exist or can't be accessed - skip POCA filling
        print(f"POCA filling skipped: {e}")
        return lats, lons


def get_coordinates_from_file(dataset: DatasetHelper, nc, fill_missing_poca: bool = False) -> dict:
    """
    Get latitude and longitude arrays as arrays from a netCDF or HDF5 file.
    Optionally fills missing POCA lat/lons with nadir values for FDR4ALT datasets.
    Optionally returns an array of beam numbers if dataset.beams is not empty.

    Args:
        dataset (DatasetHelper): DatasetHelper object.
        nc (netcdf Dataset or h5py.File): Opened NetCDF or HDF5 file.
        fill_missing_poca (bool): Flag to fill POCA with Nadir (Set True for FDR4ALT data)

    Returns:
        dict: A dictionary containing arrays of "latitude", "longitude", and optionally "beams".
    """
    if dataset.latitude_param is None:
        raise ValueError("dataset.latitude_param is required but None was provided")
    latitude_param = dataset.latitude_param

    if dataset.beams != []:
        lats = dataset.get_variable(nc, latitude_param)
        beams = dataset.get_variable(nc, latitude_param, return_beams=True)
    else:
        lats = dataset.get_variable(nc, latitude_param)
        beams = None

    if dataset.longitude_param is None:
        raise ValueError("dataset.longitude_param is required but None was provided")
    longitude_param = dataset.longitude_param
    lons = dataset.get_variable(nc, longitude_param) % 360.0

    if fill_missing_poca:
        lats, lons = _fill_missing_poca_with_nadir_fdr4alt(dataset, nc, lats, lons)

    variable_dict = {"latitude": lats, "longitude": lons, "beams": beams}
    return variable_dict


def get_spatial_filter(
    variable_dict: dict, this_area: Area, this_mask: Mask
) -> tuple[dict, int, np.ndarray]:
    """
    Get indicies of latitude and longitude arrays that are inside the CPOM Mask (if specified)
    or Area.
    Convert latlon to x/y coordinates and create a boolean mask for points inside the area/mask.

    Args:
        variable_dict (dict): Dictionary containing arrays of latitudes and longitudes.
        this_area (Area): CPOM Area object.
        this_mask (Mask): CPOM Mask object.

    Returns:
        tuple: Updated variable dictionary to include "x" and "y", the
            number of points inside area/mask, and the bounded mask object.

    """

    bounded_lat, bounded_lon, bounded_mask, _ = this_area.inside_latlon_bounds(
        variable_dict["latitude"], variable_dict["longitude"]
    )

    if this_mask is not None:
        area_mask_valid, n_inside = this_mask.points_inside(bounded_lat, bounded_lon)
        x_coords, y_coords = this_mask.latlon_to_xy(
            variable_dict["latitude"], variable_dict["longitude"]
        )
    else:
        area_mask_valid, n_inside = this_area.inside_area(bounded_lat, bounded_lon)
        x_coords, y_coords = this_area.latlon_to_xy(
            variable_dict["latitude"], variable_dict["longitude"]
        )

    # Reconstruct full-size mask
    area_mask = np.zeros_like(variable_dict["latitude"], dtype=bool)
    area_mask[bounded_mask] = area_mask_valid

    variable_dict["x"] = x_coords
    variable_dict["y"] = y_coords

    return variable_dict, n_inside, area_mask


def get_variables_and_mask(
    dataset: DatasetHelper, nc, variable_dict: dict, offset: float, area_mask: np.ndarray
):
    """
    Get elevation, time, and optional power/quality an open netCDF or HDF5 file.
    Constructs a combined boolean mask.

    Args:
        dataset (DatasetHelper): CPOM DatasetHelper object.
        nc: Opened NetCDF/HDF5 handle for the current file.
        variable_dict (dict): Must already contain latitude/longitude and derived x/y.
        offset (float): Seconds to add to the raw time variable.
        area_mask (np.ndarray): Boolean mask marking points inside the area/mask.

    Returns:
        tuple[dict, np.ndarray] | tuple[None, None]:
            (updated variable_dict, combined_mask) when at least two valid points remain;
            (None, None) when fewer than two points survive masking.
    """
    masks = [area_mask]

    # Construct the final mask and load variables

    if dataset.elevation_param is None:
        raise ValueError("dataset.elevation_param is required but None was provided")
    elevation_param = dataset.elevation_param
    variable_dict["elevation"] = dataset.get_variable(nc, elevation_param)
    masks.append(np.isfinite(variable_dict["elevation"]))

    if dataset.time_param is None:
        raise ValueError("dataset.time_param is required but None was provided")
    time_param = dataset.time_param
    variable_dict["time"] = dataset.get_variable(nc, time_param) + offset
    masks.append(np.isfinite(variable_dict["time"]))
    if dataset.power_param is not None:
        variable_dict["power"] = dataset.get_variable(nc, dataset.power_param)
        masks.append(np.isfinite(variable_dict["power"]))

    # Fdr4alt quality mask
    if dataset.quality_param is not None:
        masks.append(
            dataset.get_variable(nc, dataset.quality_param, replace_fill=False) == 0
        )  # 0 = good, bad = 1

    bool_mask = np.logical_and.reduce(masks)

    if bool_mask.sum() < 2:
        return None, None

    variable_dict["ascending"] = dataset.get_file_orbital_direction(
        latitude=(
            dataset.get_variable(nc, dataset.latitude_nadir_param)
            if dataset.latitude_nadir_param is not None
            else variable_dict["latitude"]
        ),
        nc=nc,
    )
    if dataset.uncertainty_param is not None:
        variable_dict["uncertainty"] = dataset.get_variable(nc, dataset.uncertainty_param)
    if dataset.mode_param is not None:
        variable_dict["mode"] = dataset.get_variable(nc, dataset.mode_param)

    return variable_dict, bool_mask


def get_grid_cells(variable_dict: dict, this_grid: GridArea) -> dict:
    """
    Get grid cell indices and offsets for each point.

    Args:
        variable_dict (dict): Dictionary containing arrays of x and y coordinates.
        this_grid (GridArea): CPOM GridArea object.
    Returns:
        dict: Updated variable dictionary with
        "x_bin", "y_bin", "x_cell_offset", and "y_cell_offset".
    """
    variable_dict["x_bin"], variable_dict["y_bin"] = this_grid.get_col_row_from_x_y(
        variable_dict["x"], variable_dict["y"]
    )
    variable_dict["x_cell_offset"], variable_dict["y_cell_offset"] = (
        this_grid.get_xy_relative_to_cellcentre(
            variable_dict["x"], variable_dict["y"], variable_dict["x_bin"], variable_dict["y_bin"]
        )
    )
    return variable_dict


# pylint: disable=R0913, R0917
def process_file(
    file_and_date: dict,
    dataset: DatasetHelper,
    offset: float,
    this_grid: GridArea,
    this_area: Area,
    this_mask: Mask,
    fill_missing_poca: bool = False,
) -> pl.LazyFrame | None:
    """
    Extract and process data from a single altimetry file.
    To be run in parallel in get_data_and_status_multiprocessed.

    Steps:
        1. Load coordinates (lat/lon, optional beams) from file
        2. Apply spatial filtering (area/mask bounds, convert to x/y coordinates)
        3. Load variables (elevation, time, power, quality) and build combined quality mask
        4. Apply mask to all variables
        5. Calculate grid cell positions and offsets
        6. Cast variables to appropriate data types
        7. Create Polars LazyFrame and apply elevation range filter

    Args:
        file_and_date (dict): Dictionary with 'path', 'year', 'month' keys
        dataset (DatasetHelper): CPOM dataset object
        offset (float): Time offset in seconds to apply to time data
        this_grid (GridArea): CPOM grid area object
        this_area (Area): CPOM area object
        this_mask (Mask): Optional CPOM mask object (None uses area mask)
        fill_missing_poca (bool): Fill missing POCA lat/lons with nadir (for FDR4ALT)

    Returns:
        pl.LazyFrame: Processed data with elevation range filter applied,
                        or None if insufficient valid data (< 2 points)
    """

    # Get filetype from search pattern
    context_manager = Dataset
    if dataset.search_pattern is not None and dataset.search_pattern.endswith(".h5"):
        context_manager = h5py.File

    variable_dict = {}
    with context_manager(file_and_date["path"]) as nc:
        # 1. Get coordinates
        variable_dict = get_coordinates_from_file(dataset, nc, fill_missing_poca=fill_missing_poca)
        # 2. Get spatial filter
        variable_dict, n_inside, bounded_mask = get_spatial_filter(
            variable_dict, this_area, this_mask
        )

        if n_inside < 2:  # Number of points needed to get the heading
            return None

        # 3. Get variables and combined mask
        variable_dict, bool_mask = get_variables_and_mask(
            dataset, nc, variable_dict, offset, bounded_mask
        )
        if bool_mask is None:
            return None

        for variable in variable_dict:
            if variable_dict[variable] is None:
                continue
            variable_dict[variable] = variable_dict[variable][bool_mask]

        variable_dict = get_grid_cells(variable_dict, this_grid)

        # 5. Cast to correct types
        for var in [
            "time",
            "elevation",
            "power",
            "uncertainty",
            "x",
            "y",
            "x_cell_offset",
            "y_cell_offset",
        ]:
            if var in variable_dict:
                variable_dict[var] = variable_dict[var].astype(np.float32)

        return (
            pl.LazyFrame(variable_dict)
            .with_columns(
                [
                    pl.lit(file_and_date["year"]).alias("year"),
                    pl.lit(file_and_date["month"]).alias("month"),
                ]
            )
            .cast({"elevation": pl.Float32})
            .filter((pl.col("elevation") < 6000) & (pl.col("elevation") > -500))
        )


# --------------------------------------#
# Multiprocessing Data Extraction  #
# --------------------------------------#


def get_data_and_status_multiprocessed(params, file_and_dates: np.ndarray, worker, logger):
    """
    Extract and process data from altimetry netcdf or hdf5 files
    in parallel and write the results to disk as a partitioned parquet file.

    Uses multiprocessing and writes on year of data at a time.
    Calls :
        process_file: Function to process a single file.
        write_partitions_to_disk: Function to write processed data to disk.

    Args:
        params (argparse.Namespace): Command line parameters
        file_and_dates (np.ndarray): Structured numpy array of file paths and date information.
        worker (partial): Partial function for processing files.
        logger (Logger): Logger object.

    Returns:
        dict: Processing status dictionary, containing information about the files processed.
    """

    status: dict = {
        "years_ingested": [],
        "years_files_ingested": [],
        "years_files_rejected": [],
        "years_rows_ingested": [],
        "total_l2_files_rejected": 0,
        "total_l2_files_ingested": 0,
        "total_rows_ingested": 0,
    }

    # ----------------------------------------------------------------------
    # Process each year
    # ----------------------------------------------------------------------
    # Get list of years from file_and_date from a tuple of (file_path, date)
    # Extract unique years from the file_and_dates array
    years = sorted(set(file_and_dates["year"]))

    try:
        with ProcessPoolExecutor() as executor:
            for year in years:
                logger.info("Processing year: %s", year)
                big_rows_list = []
                file_and_date_year = file_and_dates[file_and_dates["year"] == year]
                logger.info(
                    f"Number of files to process for year {year}: {len(file_and_date_year)}"
                )
                results = executor.map(
                    worker,
                    file_and_date_year,
                    chunksize=max(15, len(file_and_date_year) // (10 * 4)),
                )
                try:
                    big_rows_list = [r for r in results if r is not None]
                except (ValueError, OSError, RuntimeError) as e:
                    logger.error(f"Error collecting results for year {year}: {e}", exc_info=True)
                    big_rows_list = []

                logger.info(f"Collected {len(big_rows_list)} valid results for year {year}")

                if len(big_rows_list) == 0:
                    logger.info(f"No valid data for year {year}, skipping...")
                    status["years_files_rejected"].append(len(file_and_date_year))
                    status["total_l2_files_rejected"] += len(file_and_date_year)
                    continue

                final = pl.concat(big_rows_list).collect()
                logger.info(f"Final DataFrame shape: {final.shape}")

                total_rows = write_partitions_to_disk(
                    final_df=final,
                    partition_columns=params.partition_columns,
                    partition_xy_chunking=params.partition_xy_chunking,
                    out_dir=params.out_dir,
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


# --------------------------------------#
# Data output functions                 #
# --------------------------------------#


def write_partitions_to_disk(final_df, partition_columns, partition_xy_chunking, out_dir):
    """
    Write  DataFrame of altimetry data to disk.
    Partitioned by ('year', 'month', 'x_part', 'y_part') or
    ('year', 'x_part', 'y_part')

    Args:
        final_df (DataFrame): The final DataFrame to write.
        partition_columns (list): List of columns to partition by.
        partition_xy_chunking (int): Chunking factor for spatial partitioning.
        out_dir (str): Full path to the output directory.

    Returns:
        int: Total number of rows written to disk.
    """
    if "x_part" in partition_columns and "y_part" in partition_columns:
        final_df = final_df.filter(
            pl.col("x_bin").is_not_nan() & pl.col("y_bin").is_not_nan()
        ).with_columns(
            (pl.col("x_bin") / partition_xy_chunking).cast(pl.Int64).alias("x_part"),
            (pl.col("y_bin") / partition_xy_chunking).cast(pl.Int64).alias("y_part"),
        )

    partitions = final_df.partition_by(partition_columns, as_dict=True)

    total_rows = 0
    for key, group in partitions.items():
        subdir = os.path.join(
            out_dir,
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
        total_rows += len(group)

    return total_rows


def get_metadata_json(params, dataset, status, thisgrid, start_time):
    """
    Create and save metadata.json to the output directory.
    Metadata includes:
        - Command line parameters
        - Dataset parameters
        - Processing status
        - Grid details
        - Execution time

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
    output["gridding_time"] = f"{elapsed_time//3600}h {(elapsed_time%3600)//60}m {elapsed_time%60}s"

    meta_json_path = os.path.join(params.out_dir, "metadata.json")
    try:
        with open(meta_json_path, "w", encoding="utf-8") as f_meta:
            json.dump(output, f_meta, indent=2)
    except OSError:
        pass


def main(args):
    """
    Main function to grid altimetry data for elevation change processing.

    1. Load command line arguments
    2. Load DatasetHelper object - From a dataset YAML file or inline JSON string
    3. Clean output directory if it is populated
    4. Set up processing objects
    5. Process each year of data and write to Parquet files
    6. Write metadata JSON file with gridding details
    """
    # ------------------------------#
    # 1.Load command line arguments#
    # ------------------------------#
    args = parse_arguments(args)

    start_time = time.time()
    # ------------------------------#
    # 2. Load Dataset Object
    # ------------------------------#
    dataset_path = Path(args.dataset)
    if dataset_path.exists() and dataset_path.suffix in [".yml", ".yaml"]:
        # Method 1: Load from YAML file
        dataset = DatasetHelper(data_dir=args.data_input_dir, dataset_yaml=args.dataset)
    else:
        # Method 2: Treat as JSON string (inline config)
        try:
            dataset_config = json.loads(args.dataset)
            dataset = DatasetHelper(data_dir=args.data_input_dir, **dataset_config)
        except json.JSONDecodeError:
            sys.exit(
                f"Invalid dataset configuration: {args.dataset}. "
                f"Must be either a path to a YAML file or a valid JSON string"
            )

    clean_directory(args, dataset, confirm_regrid=True)

    file_and_dates, worker, thisgrid, logger = get_processing_objects(args, dataset)

    # --------------------------------------------------------#
    # 5. Process each year of data and write to Parquet files #
    # --------------------------------------------------------#
    status = get_data_and_status_multiprocessed(
        params=args,
        file_and_dates=file_and_dates,
        worker=worker,
        logger=logger,
    )

    # --------------------------------------------------#
    # 6. Write metadata JSON file with gridding details #
    # --------------------------------------------------#
    get_metadata_json(args, dataset, status, thisgrid, start_time)


if __name__ == "__main__":
    # Parse command line arguments
    main(sys.argv[1:])

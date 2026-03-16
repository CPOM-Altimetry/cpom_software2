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
import pyarrow.parquet as pq
from netCDF4 import Dataset  # pylint: disable=E0611

from cpom.altimetry.datasets.dataset_helper import DatasetHelper
from cpom.altimetry.tools.sec_tools.grid_for_elev_change_corrections import (
    apply_corrections,
)
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
    parser = argparse.ArgumentParser(description="""Convert altimetry data into partitioned parquet
        with ragged layout, storing each partition in a single file""")
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
    parser.add_argument(
        "--mask_values",
        default=None,
        required=False,
        nargs="+",
        help="Optional list of values to filter the mask by, e.g. --mask_values 1 2 3. ",
    )
    parser.add_argument(
        "--correction_function",
        type=str,
        default="default_corrections",
        help=("Correction function name from grid_for_elev_change_corrections.py."),
    )
    parser.add_argument("--gridarea", type=str, required=True, help="CPOM grid area object")
    parser.add_argument("--binsize", type=int, default=5000, help="Grid bin size in meters")
    parser.add_argument(
        "--standard_epoch",
        type=str,
        default="1991-01-01T00:00:00",
        help="Standard epoch for time conversion (e.g., '1991-01-01T00:00:00')",
    )
    parser.add_argument("--min_date", type=str, help="Minimum date to include (e.g., '2000.01.01')")
    parser.add_argument("--max_date", type=str, help="Maximum date to include (e.g., '2020.12.31')")
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

    parser.add_argument(
        "--force_regrid",
        action="store_true",
        help=(
            "Force regridding even if output directory exists. "
            "WARNING: This will delete the existing output directory and all its contents."
        ),
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
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )
    logger.info("output_dir=%s", params.out_dir)

    thisgrid = GridArea(params.gridarea, params.binsize)
    thisarea = Area(params.area)
    if params.mask_name:
        thismask = Mask(params.mask_name, params.mask_values)
    else:
        thismask = None

    file_and_dates = dataset.get_files_and_dates(
        hemisphere=thisarea.hemisphere, min_dt_time=params.min_date, max_dt_time=params.max_date
    )

    # Get the number of seconds between the epoch to be used in the grid and the dataset epoch
    offset = dataset.get_unified_time_epoch_offset(params.standard_epoch, dataset.dataset_epoch)

    worker = partial(
        process_file,
        dataset=dataset,
        offset=offset,
        this_grid=thisgrid,
        this_area=thisarea,
        this_mask=thismask,
        params=params,
    )

    return file_and_dates, worker, thisgrid, logger


# --------------------------------------#
# Data Loading and Processing Functions #
# --------------------------------------#


def _fill_missing_poca_with_nadir_fdr4alt(
    dataset: DatasetHelper, nc, lats: np.ndarray, lons: np.ndarray, strict_missing: bool = False
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
        latitude_nadir_param = dataset.latitude_nadir_param
        longitude_nadir_param = dataset.longitude_nadir_param
        if latitude_nadir_param is None or longitude_nadir_param is None:
            return lats, lons

        # Try to get the relocation failure variable
        relocation_failure = dataset.get_variable(
            nc,
            "expert/ice_sheet_qual_relocation",
            replace_fill=False,
            raise_if_missing=strict_missing,
        )

        # Get nadir coordinates
        latitude_nadir = dataset.get_variable(
            nc, latitude_nadir_param, raise_if_missing=strict_missing
        )
        longitude_nadir = dataset.get_variable(
            nc, longitude_nadir_param, raise_if_missing=strict_missing
        )

        # Fill missing POCA locations with nadir values
        lats = np.where(relocation_failure > 0, latitude_nadir, lats)
        lons = np.where(relocation_failure > 0, longitude_nadir, lons)

        return lats, lons

    except (KeyError, ValueError, AttributeError) as e:
        # Variable doesn't exist or can't be accessed - skip POCA filling
        print(f"POCA filling skipped: {e}")
        return lats, lons


def _log_file_stats(logger, file_stats):
    """Extracted helper to keep the hot loop clean."""
    logger.debug(
        "file=%s | accepted=%d/%d | rejected(area/mask=%d, quality/time/nan=%d, elev_range=%d)%s%s",
        file_stats.get("path"),
        int(file_stats.get("accepted", 0)),
        int(file_stats.get("total", 0)),
        int(file_stats.get("outside_area_or_mask", 0)),
        int(file_stats.get("outside_quality_time_elev_or_nan", 0)),
        int(file_stats.get("outside_elevation_range", 0)),
        (
            f" | reason={file_stats.get('file_reject_reason')}"
            if file_stats.get("file_reject_reason")
            else ""
        ),
        (
            f" | detail={file_stats.get('file_reject_detail')}"
            if file_stats.get("file_reject_detail")
            else ""
        ),
    )


def get_coordinates_from_file(
    dataset: DatasetHelper,
    nc,
    fill_missing_poca: bool = False,
    strict_missing: bool = False,
) -> dict:
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
        lats = dataset.get_variable(nc, latitude_param, raise_if_missing=strict_missing)
        beams = dataset.get_variable(
            nc,
            latitude_param,
            return_beams=True,
            raise_if_missing=strict_missing,
        )
    else:
        lats = dataset.get_variable(nc, latitude_param, raise_if_missing=strict_missing)
        beams = None

    if dataset.longitude_param is None:
        raise ValueError("dataset.longitude_param is required but None was provided")
    longitude_param = dataset.longitude_param
    lons = dataset.get_variable(nc, longitude_param, raise_if_missing=strict_missing) % 360.0

    if fill_missing_poca:
        lats, lons = _fill_missing_poca_with_nadir_fdr4alt(dataset, nc, lats, lons)

    return {"latitude": lats, "longitude": lons, "beams": beams}


def get_spatial_filter(
    variable_dict: dict, this_area: Area, this_mask: Mask
) -> tuple[dict, int, np.ndarray]:
    """
    Get indicies of latitude and longitude arrays that are inside the CPOM Mask (if specified)
    or Area.

    Args:
        variable_dict (dict): Dictionary containing arrays of latitudes and longitudes.
        this_area (Area): CPOM Area object.
        this_mask (Mask): CPOM Mask object.

    Returns:
        tuple: Updated variable dictionary to include "x" and "y", the
            number of points inside area/mask, and the bounded mask object.

    """
    bounded_lat, bounded_lon, bounded_indices, n_bounded = this_area.inside_latlon_bounds(
        variable_dict["latitude"], variable_dict["longitude"]
    )
    if n_bounded == 0:
        return variable_dict, 0, np.array([], dtype=int)

    if this_mask is not None:
        area_mask_valid, n_inside = this_mask.points_inside(bounded_lat, bounded_lon)
        x, y = this_mask.latlon_to_xy(bounded_lat, bounded_lon)
    else:
        area_mask_valid, n_inside = this_area.inside_area(bounded_lat, bounded_lon)
        x, y = this_area.latlon_to_xy(bounded_lat, bounded_lon)

    valid_original_indices = bounded_indices[area_mask_valid]

    for var in variable_dict:
        if variable_dict[var] is not None:
            variable_dict[var] = variable_dict[var][valid_original_indices]

    variable_dict["x"] = x[area_mask_valid]
    variable_dict["y"] = y[area_mask_valid]

    return variable_dict, n_inside, valid_original_indices


# pylint: disable= R0913, R0917
def get_variables_and_mask(
    dataset: DatasetHelper,
    nc,
    variable_dict: dict,
    params: argparse.Namespace,
    offset: float,
    area_mask: np.ndarray,
):
    """
    Get elevation, time, optional power/quality and additional variables from
    an open netCDF or HDF5 file.
    Returns a masked variable dictionary and combined area and finite (elevation/time/power) mask.
    Args:
        dataset (DatasetHelper): CPOM DatasetHelper object.
        nc: Opened NetCDF/HDF5 handle for the current file.
        variable_dict (dict): Must already contain latitude/longitude and derived x/y.
        offset (float): Seconds to add to the raw time variable.
        area_mask (np.ndarray): Boolean mask marking points inside the area/mask.
        strict_missing (bool): Whether to raise an error if a required variable is missing.
    Returns:
        tuple[dict, np.ndarray] | tuple[None, None]:
            (updated variable_dict, combined_mask) when at least two valid points remain;
            (None, None) when fewer than two points survive masking.
    """
    elevation_param = dataset.elevation_param
    if elevation_param is None:
        raise ValueError("dataset.elevation_param is required but None was provided")

    time_param = dataset.time_param
    if time_param is None:
        raise ValueError("dataset.time_param is required but None was provided")

    for key, param in [("elevation", elevation_param), ("time", time_param)]:
        variable_dict[key] = dataset.get_variable(nc, param, raise_if_missing=params.debug)[
            area_mask
        ]

    if dataset.power_param is not None:
        variable_dict["power"] = dataset.get_variable(
            nc, dataset.power_param, raise_if_missing=params.debug
        )[area_mask]

    variable_dict["time"] += offset

    # Build boolean mask
    bool_mask = np.isfinite(variable_dict["elevation"]) & np.isfinite(variable_dict["time"])
    if "power" in variable_dict:
        bool_mask &= np.isfinite(variable_dict["power"])

    if bool_mask.sum() < 2:
        return None, None

    variable_dict = {k: v[bool_mask] for k, v in variable_dict.items() if v is not None}

    variable_dict["ascending"] = dataset.get_file_orbital_direction(
        latitude=(
            dataset.get_variable(
                nc,
                dataset.latitude_nadir_param,
                raise_if_missing=params.debug,
            )[
                area_mask
            ][bool_mask]
            if dataset.latitude_nadir_param is not None
            else variable_dict["latitude"]
        ),
        nc=nc,
    )
    for var, param in {
        "uncertainty": dataset.uncertainty_param,
        "mode": dataset.mode_param,
        **(params.add_vars if params.add_vars is not None else {}),
    }.items():
        if param is not None:
            variable_dict[var] = dataset.get_variable(nc, param, raise_if_missing=params.debug)[
                area_mask
            ][bool_mask]

    return variable_dict, area_mask[bool_mask]


def get_grid_cells(variable_dict: dict, this_grid: GridArea) -> dict:
    """
    Convert latlon to x/y coordinates.

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


# pylint: disable= R0911, R0912, R0913, R0914, R0915,R0917
def process_file(
    file_and_date: dict,
    dataset: DatasetHelper,
    offset: float,
    this_grid: GridArea,
    this_area: Area,
    this_mask: Mask,
    params: argparse.Namespace,
) -> pl.LazyFrame | tuple[pl.LazyFrame | None, dict | None] | None:
    """
    Extract and process data from a single altimetry file.
    To be run in parallel called in get_data_and_status_multiprocessed.

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
        params (argparse.Namespace): Command Line Parameters

    Returns:
        pl.LazyFrame | tuple[pl.LazyFrame | None, dict | None] | None:
            In debug mode, returns `(frame_or_none, stats)`; otherwise returns `frame` or `None`.
    """
    # Get filetype from search pattern
    context_manager = Dataset
    if dataset.search_pattern is not None and dataset.search_pattern.endswith(".h5"):
        context_manager = h5py.File

    stats: dict | None = None
    if params.debug:
        stats = {
            "path": file_and_date["path"],
            "accepted": 0,
            "outside_area_or_mask": 0,
            "outside_quality_time_elev_or_nan": 0,
            "file_reject_reason": None,
            "file_reject_detail": None,
            "total": 0,
        }

    variable_dict = {}
    try:
        with context_manager(file_and_date["path"]) as nc:
            # 1. Get coordinates
            variable_dict = get_coordinates_from_file(
                dataset,
                nc,
                fill_missing_poca=params.fill_missing_poca,
                strict_missing=params.debug,
            )
            if params.debug and stats is not None:
                total = int(len(variable_dict["latitude"]))
                stats["total"] = total

            # 2. Get spatial filter
            variable_dict, n_inside, area_mask = get_spatial_filter(
                variable_dict, this_area, this_mask
            )
            if params.debug and stats is not None:
                stats["outside_area_or_mask"] = max(0, total - int(n_inside))

            if n_inside < 2:  # Number of points needed to get the heading
                if params.debug:
                    if stats is not None:
                        stats["file_reject_reason"] = "too_few_points_after_spatial_filter"
                    return None, stats
                return None

            # 3. Get variables and combined finite mask
            # 4. Apply mask to variables
            variable_dict, mask = get_variables_and_mask(
                dataset, nc, variable_dict, params, offset, area_mask
            )
            if variable_dict is None:
                if params.debug:
                    if stats is not None:
                        stats["file_reject_reason"] = "too_few_points_after_finite_mask"
                    return None, stats
                return None

            if (
                dataset.default_elev_correction is not None
                or dataset.default_qual_correction is not None
                or params.correction_function != "default_corrections"
            ):
                try:
                    variable_dict = apply_corrections(
                        dataset=dataset,
                        nc=nc,
                        input_mask=mask,
                        variable_dict=variable_dict,
                        params=params,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to apply correction "
                        f"'{params.correction_function}' for mission '{dataset.mission}': "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

                if variable_dict is None:
                    if params.debug:
                        if stats is not None:
                            stats["file_reject_reason"] = "too_few_points_after_quality_mask"
                        return None, stats
                    return None

            if params.debug:
                n_quality = int(len(variable_dict["time"]))
                if stats is not None:
                    stats["outside_quality_time_elev_or_nan"] = max(0, n_inside - n_quality)

            # 5. Calculate grid cell positions and offsets
            variable_dict = get_grid_cells(variable_dict, this_grid)

            # 6. Cast variables to appropriate data types
            for variable, arr in variable_dict.items():
                if arr is None:
                    continue
                variable_dict[variable] = (
                    arr.astype(np.float32)
                    if variable
                    in [
                        "time",
                        "elevation",
                        "power",
                        "uncertainty",
                        "x",
                        "y",
                        "x_cell_offset",
                        "y_cell_offset",
                    ]
                    else arr
                )

            frame = pl.LazyFrame(variable_dict).with_columns(
                pl.lit(file_and_date["year"]).alias("year"),
                pl.lit(file_and_date["month"]).alias("month"),
            )

            if params.debug:
                n_before_elev = frame.select(pl.len()).collect().item()

            frame = frame.filter((pl.col("elevation") < 6000) & (pl.col("elevation") > -500))

            if params.debug:
                n_after_elev = frame.select(pl.len()).collect().item()
                if stats is not None:
                    stats["outside_elevation_range"] = max(0, n_before_elev - n_after_elev)
                    stats["accepted"] = n_after_elev
                if n_after_elev == 0:
                    if stats is not None:
                        stats["file_reject_reason"] = "all_points_outside_elevation_range"
                    return None, stats
            if params.debug:
                return frame, stats
            return frame

    except KeyError as e:
        if params.debug:
            if stats is not None:
                stats["file_reject_reason"] = "missing_variable"
                stats["file_reject_detail"] = str(e)
            return None, stats
        return None
    except (ValueError, OSError, RuntimeError, TypeError, AttributeError) as e:
        if params.debug:
            if stats is not None:
                stats["file_reject_reason"] = "processing_error"
                stats["file_reject_detail"] = f"{type(e).__name__}: {e}"
            return None, stats
        return None


# --------------------------------------#
# File Writing Helpers   #
# --------------------------------------#


def _group_periods(file_and_dates, use_month):
    """
    Group files by processing period (year/month or just year).

    Args:
        file_and_dates (np.ndarray): Structured array with year/month data
        use_month (bool): If True, group by year/month. If False, group by year only.

    Yields:
        tuple: (period_label, files_in_period)
    """
    if use_month:
        # Group by year/month pairs
        keys = np.unique(
            np.stack([file_and_dates["year"], file_and_dates["month"]], axis=1),
            axis=0,
        )
        for year, month in keys:
            mask = (file_and_dates["year"] == year) & (file_and_dates["month"] == month)
            period_label = f"{int(year)}-{int(month):02d}"
            yield period_label, file_and_dates[mask]
    else:
        # Group by year only
        years = sorted(set(file_and_dates["year"]))
        for year in years:
            mask = file_and_dates["year"] == year
            yield str(int(year)), file_and_dates[mask]


def write_partitions_to_disk(
    final_df,
    partition_columns,
    partition_xy_chunking,
    out_dir,
    writer_cache=None,
):
    """
    Write altimetry data to disk.
    Partitioned by ('year', 'month', 'x_part', 'y_part') or ('year', 'x_part', 'y_part')
    Args:
        final_df (DataFrame | LazyFrame): The data to write.
        partition_columns (list): List of columns to partition by.
        partition_xy_chunking (int): Chunking factor for spatial partitioning.
        out_dir (str): Full path to the output directory.
        writer_cache (dict | None): Cache of open ParquetWriters keyed by output path.

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

    if writer_cache is None:
        writer_cache = {}

    total_rows = 0
    for key, group in final_df.partition_by(
        partition_columns, as_dict=True, maintain_order=False
    ).items():
        subdir = os.path.join(
            out_dir,
            *[f"{col}={val}" for col, val in zip(partition_columns, key)],
        )
        outfile = os.path.join(subdir, "data.parquet")

        if outfile not in writer_cache:
            os.makedirs(subdir, exist_ok=True)
            writer_cache[outfile] = pq.ParquetWriter(
                outfile, group.to_arrow().schema, compression="zstd"
            )

        writer_cache[outfile].write_table(group.to_arrow())
        total_rows += len(group)

    return total_rows


# -----------------------------#
# Multi-Processing Function    #
# ------------------------------#


def get_data_and_status_multiprocessed(
    params, file_and_dates: np.ndarray, worker, logger
):  # pylint: disable=too-many-arguments
    """
    Extract and process data from altimetry netcdf or hdf5 files
    in parallel and write the results to disk as a partitioned parquet file.

    Uses multiprocessing and writes on year of data at a time.

    Calls:
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
        "empty_periods": [],
        "years_ingested": [],
        "years_files_ingested": [],
        "years_files_rejected": [],
        "years_rows_ingested": [],
        "total_l2_files_rejected": 0,
        "total_l2_files_ingested": 0,
        "total_rows_ingested": 0,
    }
    use_month = "month" in params.partition_columns

    try:
        with ProcessPoolExecutor(max_workers=params.max_workers) as executor:
            for period_label, files_in_period in _group_periods(file_and_dates, use_month):
                logger.info("Processing period %s (%d files)", period_label, len(files_in_period))
                if len(files_in_period) == 0:
                    continue

                n_valid = n_rejected = total_rows = 0
                writer_cache: dict[str, pq.ParquetWriter] = {}
                buffer: list[pl.LazyFrame] = []
                chunksize = max(1, len(files_in_period) // max(1, params.max_workers * 4))

                try:
                    for result in executor.map(worker, files_in_period, chunksize=chunksize):
                        frame, file_stats = (
                            result
                            if isinstance(result, tuple) and len(result) == 2
                            else (result, None)
                        )

                        if params.debug and file_stats and isinstance(file_stats, dict):
                            _log_file_stats(logger, file_stats)

                        if frame is not None:
                            buffer.append(frame)
                            n_valid += 1

                            if len(buffer) >= params.flush_every:
                                total_rows += write_partitions_to_disk(
                                    pl.concat(buffer, rechunk=False).collect(),
                                    params.partition_columns,
                                    params.partition_xy_chunking,
                                    params.out_dir,
                                    writer_cache,
                                )
                                buffer.clear()
                        else:
                            n_rejected += 1

                    if buffer:  # Handle last batch if < flush_every
                        total_rows += write_partitions_to_disk(
                            pl.concat(buffer, rechunk=False).collect(),
                            params.partition_columns,
                            params.partition_xy_chunking,
                            params.out_dir,
                            writer_cache,
                        )
                        buffer.clear()

                except (ValueError, OSError, RuntimeError) as e:
                    logger.error(
                        "Error collecting results for %s: %s", period_label, e, exc_info=True
                    )
                finally:
                    for w in writer_cache.values():
                        w.close()

                logger.info(
                    "Period %s: %d valid, %d rejected, %d rows",
                    period_label,
                    n_valid,
                    n_rejected,
                    total_rows,
                )

                if n_valid == 0:
                    status["empty_periods"].append(period_label)
                    continue

                status["years_ingested"].append(period_label)
                status["years_files_ingested"].append(n_valid)
                status["years_files_rejected"].append(n_rejected)
                status["years_rows_ingested"].append(total_rows)
                status["total_l2_files_ingested"] += n_valid
                status["total_l2_files_rejected"] += n_rejected
                status["total_rows_ingested"] += total_rows

    except (OSError, ValueError) as e:
        logger.error("Multiprocessing failed: %s", e)
        sys.exit(1)

    return status


# --------------------------------------#
# Metadata Output Function
# --------------------------------------#
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
    output["gridding_time"] = (
        f"{elapsed_time // 3600}h {(elapsed_time % 3600) // 60}m {elapsed_time % 60}s"
    )

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

    clean_directory(args, dataset, confirm_regrid=not args.force_regrid)

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

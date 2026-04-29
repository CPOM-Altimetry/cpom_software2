"""
cpom.altimetry.tools.sec_tools.grid_for_elevation_change.py

Purpose:
    Grid altimetry data for SEC (model fit) processing.
    Stores data in a ragged layout,  partitioned by (year, x_part, y_part)
    or (year, month, x_part, y_part).
    Each x_part/y_part is a group of grid cells. The size defined by
    partition_xy_chunking parameter.

Output:
    - Partitioned Parquet files in out_dir, with ragged layout.
    - Each partition stored in a single Parquet file:
        <out_dir>/year=YYYY/x_part=XX/y_part=YY/data.parquet
        or
        <out_dir>/year=YYYY/month=MM/x_part=XX/y_part=YY/data.parquet
    - Metadata: <out_dir>/metadata.json
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
from typing import Any, Generator

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from cpom.altimetry.datasets.dataset_helper import DatasetHelper
from cpom.altimetry.tools.sec_tools.grid_for_elev_change_corrections import (
    apply_corrections,
)
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    elapsed,
    get_algo_name,
    write_metadata,
)
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers
from cpom.masks.masks import Mask

# pylint: disable=too-many-lines,too-many-locals,too-many-statements, too-many-branches

# --------------------------------------#
# Set up Functions                     #
# --------------------------------------#


def parse_arguments(args):
    """Parse command line arguments for gridding altimetry data."""
    parser = argparse.ArgumentParser(description="""Convert altimetry data into partitioned parquet
        with ragged layout, storing each partition in a single file""")
    # I/O parameters
    parser.add_argument(
        "--data_input_dir",
        type=str,
        required=True,
        help="Root directory containing L2 input files to grid.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a dataset YAML file or an inline JSON string with dataset config.",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory Path")
    parser.add_argument(
        "--correction_function",
        type=str,
        default="default_corrections",
        help="Correction function name from grid_for_elev_change_corrections.py.",
    )
    parser.add_argument(
        "--force_regrid",
        action="store_true",
        help=(
            "Force regridding even if output directory exists. "
            "WARNING: This will delete the existing output directory and all its contents."
        ),
    )
    # Data processing function parameters
    parser.add_argument("--area", type=str, required=True, help="CPOM Area object")
    parser.add_argument(
        "--mask_name",
        type=str,
        help="Optional CPOM Mask to apply, instead of the default --area mask.",
    )
    parser.add_argument(
        "--mask_values",
        default=None,
        nargs="+",
        help="Mask pixel values to accept as valid, e.g. --mask_values 1, 2, 3.",
        type=int,
    )
    parser.add_argument("--gridarea", type=str, required=True, help="CPOM GridArea object")
    parser.add_argument("--binsize", type=int, default=5000, help="Grid binsize in meters.")
    parser.add_argument(
        "--fill_missing_poca",
        action="store_true",
        help="Replace failed POCA relocations with nadir lat/lon. FDR4ALT datasets only.",
    )
    # Time filtering parameters
    parser.add_argument(
        "--standard_epoch",
        type=str,
        default="1991-01-01T00:00:00",
        help="Reference epoch for time values. Default: '1991-01-01T00:00:00'",
    )
    parser.add_argument("--min_date", type=str, help="Minimum date to include (e.g., '2000.01.01')")
    parser.add_argument("--max_date", type=str, help="Maximum date to include (e.g., '2020.12.31')")

    # Multiprocessing parameters
    parser.add_argument(
        "--max_workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of parallel worker processes. Default: number of CPU cores.",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=60,
        help="Number of files to buffer before flushing to disk. Default: 60.",
    )
    parser.add_argument(
        "--add_vars",
        type=json.loads,
        default="{}",
        help="Extra variables to extract as a JSON dict mapping output name to NetCDF path,"
        ' e.g. \'{"tide_ocean": "land_ice_segments/geophysical/tide_ocean"}\'.',
    )
    # Output partitioning parameters
    parser.add_argument(
        "--partition_columns",
        type=str,
        nargs="+",
        default=["year", "x_part", "y_part"],
        help="Partition columns. Must include 'year', 'x_part', 'y_part'",
    )
    parser.add_argument(
        "--partition_xy_chunking",
        type=int,
        default=20,
        help=(
            "Number of grid cells per spatial partition. "
            "x_part = x_bin // chunk, y_part = y_bin // chunk. Default = 20"
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging")

    return parser.parse_args(args)


def _robust_rmtree(path: str) -> None:
    """Remove a directory tree robustly, handling NFS silly-renamed files.

    On NFS mounts (e.g. /cpnet/) two failure modes can occur:

    1. Python 3.12's shutil.rmtree uses an fd-based deletion path that calls
       os.rmdir() on subdirectories, which can raise ``OSError: [Errno 39]
       Directory not empty`` due to stale dentry-cache entries.

    2. When a previous run was interrupted while files were still open, the NFS
       client renames them to hidden ``.nfsXXXXXX`` "silly files" that cannot be
       deleted until every process holding a file-descriptor to them has closed
       it.  Neither shutil.rmtree nor ``rm -rf`` can remove these files while
       they are still in use.

    Strategy: try a plain delete first; if that fails, **rename** the directory
    out of the way (to a timestamped sibling) and let the caller create a fresh
    output directory.  The OS will garbage-collect the renamed directory once all
    file-handles are released.
    """
    import subprocess  # pylint: disable=import-outside-toplevel

    # --- attempt 1: shutil.rmtree ---
    try:
        shutil.rmtree(path)
        return  # success
    except OSError:
        pass

    # --- attempt 2: subprocess rm -rf (bypasses Python's fd-safe path) ---
    subprocess.run(["rm", "-rf", path], check=False)
    if not os.path.exists(path):
        return  # success

    # --- attempt 3: NFS silly-file fallback — rename the directory aside ---
    # .nfsXXXXXX files cannot be removed while held open by another process.
    # Moving the whole directory out of the way lets us create a clean output
    # directory immediately; the OS removes the renamed tree once all handles close.
    timestamp = int(time.time())
    parent = os.path.dirname(path.rstrip("/"))
    basename = os.path.basename(path.rstrip("/"))
    aside_path = os.path.join(parent, f".old_{basename}_{timestamp}")
    try:
        os.rename(path, aside_path)
        print(
            f"WARNING: Could not fully delete '{path}' (NFS silly files are still held open "
            f"by another process). Directory has been renamed to '{aside_path}' and will be "
            f"cleaned up automatically once all file handles are closed. "
            f"You can also remove it manually later with: rm -rf '{aside_path}'"
        )
        return  # caller will now mkdir the original path fresh
    except OSError as exc:
        raise OSError(
            f"Failed to remove or move aside '{path}'.  "
            f"NFS silly files (.nfsXXXXXX) may still be held open by another process — "
            f"check with: lsof +D '{path}'"
        ) from exc


def clean_directory(params: argparse.Namespace, dataset: DatasetHelper, confirm_regrid=False):
    """
    Clear the output directory if it exists, then recreate it.
    Optionally prompt the user for confirmation before deleting if confirm_regrid is True.
    Exits if the directory fails the safety check, (must contain the mission name and not be root).

    Args:
        params (argparse.Namespace): Command line arguments.
        dataset (DatasetHelper): CPOM DatasetHelper object.
        confirm_regrid (bool): If True, prompt the user to confirm deletion.

    """
    # Full regrid => remove entire directory, then create fresh
    if os.path.exists(params.out_dir):
        if params.out_dir != "/" and dataset.mission in params.out_dir:  # safety check
            if confirm_regrid is True:
                response = (
                    input("Confirm removal of previous grid archive? (y/n): ").strip().lower()
                )
                if response == "y":
                    _robust_rmtree(params.out_dir)
                else:
                    print("Exiting as user requested not to overwrite grid archive")
                    sys.exit(0)
            else:
                _robust_rmtree(params.out_dir)

        else:
            sys.exit(1)
    os.makedirs(params.out_dir, exist_ok=True)


def get_processing_objects(
    params: argparse.Namespace, dataset: DatasetHelper
) -> tuple[dict, Any, GridArea, logging.Logger]:
    """
    Set up objects needed to process altimetry files.

    Initalises logger, GridArea, Area, and Mask objects based on command line parameters,
    computes the time-epoch offset and builds the worker partial function for multiprocessing.

    Args:
        params (argparse.Namespace): Command line parameters
        dataset (DatasetHelper): CPOM DatasetHelper object.

    Returns:
        tuple:
            - file_and_dates (np.ndarray): Structured array of file paths and date information.
            - worker (partial function): Partial function to be used by process_file.
            - thisgrid (GridArea): GridArea object representing the grid to be used for processing
            - logger (logging.Logger): Logger object.
    """

    logger = set_loggers(
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )
    logger.info("Output_dir=%s", params.out_dir)

    thisgrid = GridArea(params.gridarea, params.binsize)
    thisarea = Area(params.area)
    if params.mask_name:
        thismask = Mask(params.mask_name, params.mask_values)
    else:
        thismask = None

    logger.info("Finding files and dates")

    file_and_dates = dataset.get_files_and_dates(
        hemisphere=thisarea.hemisphere, min_dt_time=params.min_date, max_dt_time=params.max_date
    )

    logger.info("Found %s files", len(file_and_dates))

    # For multi-beam missions (IS2), expand to one task per (file, beam)
    if dataset.beams:
        n, nb = len(file_and_dates), len(dataset.beams)
        new_dtype = np.dtype(file_and_dates.dtype.descr + [("beam", "U10")])
        expanded = np.empty(n * nb, dtype=new_dtype)

        for name in file_and_dates.dtype.names:
            expanded[name] = np.repeat(file_and_dates[name], nb)
        expanded["beam"] = np.tile(dataset.beams, n)

        file_and_dates = expanded
        logger.info("Expanded to %s per-beam tasks (%s beams)", len(file_and_dates), nb)

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
    Replace missing POCA lat/lons with nadir values for FDR4ALT datasets.
    Points where 'expert/ice_sheet_qual_relocation' > 0 are filled with nadir lat/lon values.

    Args:
        dataset (DatasetHelper): CPOM DatasetHelper object (dataset configuration).
        nc (netcdf Dataset or h5py.File): Opened file handle.
        lats (np.ndarray): Array of POCA latitudes.
        lons (np.ndarray): Array of POCA longitudes.
        strict_missing (bool): If True, raise an error if expected variables are absent.

    Returns:
        tuple[np.ndarray, np.ndarray]: Filled latitude and longitude arrays.
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
    """
    Log per-file processing statistics at DEBUG level.

    Args:
        logger (logging.Logger): Logger object.
        file_stats (dict): Per-file counts and rejection reason from process_file.
    """
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
    Get latitude , longitude and optionally beam arrays (for is2) from an open netCDF or HDF5 file.
    Longitudes are normalised to (0, 360) range.
    For FDR4ALT datasets,optionally fill POCA lat/lon values with nadir lat/lon.

    Args:
        dataset (DatasetHelper): CPOM DatasetHelper object (dataset configuration).
        nc (netcdf Dataset or h5py.File): Opened file handle.
        fill_missing_poca (bool): If True, replace failed POCA relocations with nadir values.
        strict_missing (bool): If True, raise an error if expected variables are absent.

    Returns:
        dict[str, np.ndarray]: Dictionary of "latitude", "longitude", and optionally "beams" arrays.
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
    variable_dict: dict, this_area: Area, this_mask: Mask | None
) -> tuple[dict, int, np.ndarray]:
    """
    Filter arrays to points inside CPOM Area bounds or Mask (if specified),
    and convert lat/lon to x/y coordinates.

    First clips to areas lat/lon bounds, then applies mask if specified or area mask if not.

    Args:
        variable_dict (dict): Loaded arrays (must include: latitude, longitude).
        this_area (Area): CPOM Area object for bounding-box and polygon filtering.
        this_mask (Mask): CPOM Mask object for mask-based filtering. If None, the
            area polygon is used instead.

    Returns:
        tuple:
            - variable_dict (dict[str, np.ndarray]): Updated variable_dict arrays filtered to mask.
            - n_inside (int): Number of points inside the area/mask.
            - valid_original_indices (np.ndarray): Surviving indices.
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
) -> tuple[dict, np.ndarray] | tuple[None, None]:
    """
    Reads elevation, time, optionally power, mode, uncertainty and any additional variables from
    an open file. Derives an ascending/descending flag.
    Applies the area_mask, adds the time offset to correct time. Builds a combined finite mask of
    elevation, time and power and subsets all arrays.

    Args:
        dataset (DatasetHelper): CPOM DatasetHelper object (dataset configuration).
        nc (netcdf Dataset or h5py.File): Opened file handle.
        variable_dict (dict): Loaded arrays (must include: latitude, longitude, x, y).
        offset (float): Seconds to add to raw time values to convert to standard epoch.
        area_mask (np.ndarray): Boolean index array from get_spatial_filter.
    Returns:
        - variable_dict (dict[str, np.ndarray]): Updated variable_dict arrays.
        - combined_mask (np.ndarray): Combined boolean mask of area_mask and
            finite elevation/time/power or (None, None) if fewer than two valid points remain.
    """
    elevation_param = dataset.elevation_param
    if elevation_param is None:
        raise ValueError("dataset.elevation_param is required but None was provided")

    time_param = dataset.time_param
    if time_param is None:
        raise ValueError("dataset.time_param is required but None was provided")

    for key, param in [("elevation", elevation_param), ("time", time_param)]:
        var_data = dataset.get_variable(nc, param, raise_if_missing=params.debug)
        if var_data.size == 0:
            return None, None
        variable_dict[key] = var_data[area_mask]

    if dataset.power_param is not None:
        var_data = dataset.get_variable(nc, dataset.power_param, raise_if_missing=params.debug)
        if var_data.size == 0:
            return None, None
        variable_dict["power"] = var_data[area_mask]

    variable_dict["time"] += offset

    # Build boolean mask
    bool_mask = np.isfinite(variable_dict["elevation"]) & np.isfinite(variable_dict["time"])
    if "power" in variable_dict:
        bool_mask &= np.isfinite(variable_dict["power"])

    if bool_mask.sum() < 2:
        return None, None

    variable_dict = {k: v[bool_mask] for k, v in variable_dict.items() if v is not None}

    lat_nadir_data = None
    if dataset.latitude_nadir_param is not None:
        lat_nadir_data = dataset.get_variable(
            nc, dataset.latitude_nadir_param, raise_if_missing=params.debug
        )
        if lat_nadir_data.size == 0:
            return None, None
        lat_nadir_data = lat_nadir_data[area_mask][bool_mask]
    else:
        lat_nadir_data = variable_dict["latitude"]

    variable_dict["ascending"] = dataset.get_file_orbital_direction(
        latitude=lat_nadir_data,
        nc=nc,
    )
    for var, param in {
        "uncertainty": dataset.uncertainty_param,
        "mode": dataset.mode_param,
        **(params.add_vars if params.add_vars is not None else {}),
    }.items():
        if param is not None:
            var_data = dataset.get_variable(nc, param, raise_if_missing=params.debug)
            if var_data.size > 0:
                variable_dict[var] = var_data[area_mask][bool_mask]

    return variable_dict, area_mask[bool_mask]


def get_grid_cells(variable_dict: dict, this_grid: GridArea) -> dict:
    """
    Compute grid cell indices and cell offsets to the grid cell centre for each point.

    Args:
        variable_dict (dict): Loaded arrays (must include: x, y).
        this_grid (GridArea): CPOM GridArea object.
    Returns:
        dict: Updated variable_dict arrays (adds: x_bin, y_bin, x_cell_offset, y_cell_offset).
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
    file_and_date: dict | np.void,
    dataset: DatasetHelper,
    offset: float,
    this_grid: GridArea,
    this_area: Area,
    this_mask: Mask | None,
    params: argparse.Namespace,
) -> pl.LazyFrame | tuple[pl.LazyFrame | None, dict | None] | None:
    """
    Extract and grid data from a single altimetry file.

    Runs a full per file pipeline:
        1. Load coordinates
        2. Spatial filter to area/mask and convert to x/y
        3. Load variables, apply corrections and mask.
        4. Compute grid cell positions.
        5. Cast dtypes
        6. Create Polars LazyFrame and apply elevation range filter.

    Args:
        file_and_date (dict): File metadata (must include: 'path', 'year', 'month' keys)
        dataset (DatasetHelper): CPOM DatasetHelper object (dataset configuration).
        offset (float): Time offset in seconds to convert to standard epoch.
        this_grid (GridArea): CPOM GridArea object
        this_area (Area): CPOM Area object
        this_mask (Mask):CPOM mask object, or None to use the area polygon.
        params (argparse.Namespace): Command line arguments

    Returns:
        pl.LazyFrame | [pl.LazyFrame | None, dict | None] | None: Processed data frame,
        or None if the file is rejected.
        In debug mode returns a tuple of (frame or None, stats dict).
    """
    # Get filetype from search pattern

    # For multi-beam missions expanded to per-beam tasks, restrict dataset to the
    # single beam for this task.
    beam_value: str | None = None
    if isinstance(file_and_date, np.void):
        field_names = file_and_date.dtype.names or ()
        if "beam" in field_names and file_and_date["beam"]:
            beam_value = str(file_and_date["beam"])
    elif isinstance(file_and_date, dict):
        beam = file_and_date.get("beam")
        if beam:
            beam_value = str(beam)

    if beam_value is not None:
        dataset.beams = [beam_value]

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
        with dataset.get_file_handle(file_and_date["path"]) as nc:
            # 1. Get coordinates
            variable_dict = get_coordinates_from_file(
                dataset,
                nc,
                fill_missing_poca=params.fill_missing_poca,
                strict_missing=params.debug,
            )
            if params.debug:
                logging.debug(
                    "File: %s, Latitudes: %s", file_and_date["path"], variable_dict["latitude"]
                )
                logging.debug(
                    "File: %s, Longitudes: %s", file_and_date["path"], variable_dict["longitude"]
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
            tmp_variable_dict, mask = get_variables_and_mask(
                dataset, nc, variable_dict, params, offset, area_mask
            )
            if tmp_variable_dict is None or mask is None:
                if params.debug:
                    if stats is not None:
                        stats["file_reject_reason"] = "too_few_points_after_finite_mask"
                    return None, stats
                return None
            variable_dict = tmp_variable_dict

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
    except (ValueError, OSError, RuntimeError, TypeError, AttributeError, IndexError) as e:
        if params.debug:
            if stats is not None:
                stats["file_reject_reason"] = "processing_error"
                stats["file_reject_detail"] = f"{type(e).__name__}: {e}"
            return None, stats
        return None


# --------------------------------------#
# File Writing Helpers                  #
# --------------------------------------#


def _group_periods(
    file_and_dates: np.ndarray, use_month: bool
) -> Generator[tuple[str, np.ndarray], None, None]:
    """
    Group files by processing period and yield each period's label and file subset.

    Args:
        file_and_dates (np.ndarray): Structured array of file paths and date information.
        use_month (bool): If True, group by year/month. If False, group by year only.

    Yields:
        tuple[str, np.ndarray]: Period label (e.g. "2020" or "2020-03") and
            the subset of file_and_dates for that period.
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
    final_df: pl.DataFrame,
    partition_columns: list[str],
    partition_xy_chunking: int,
    out_dir: str,
    writer_cache: dict[str, pq.ParquetWriter] | None = None,
):
    """
    Write a DataFrame to disk as hive-partitioned Parquet files.

    Computes x_part and y_part from x_bin and y_bin. Reuses open ParquetWriters to
    avoid opening and closing files repeatedly during streaming writes.

    Args:
        final_df (DataFrame | LazyFrame): The data to write.
        partition_columns (list): List of columns to partition by.
        partition_xy_chunking (int): Divisor applied to x_bin/y_bin to get x_part/y_part.
        out_dir (str): Output directory path.
        writer_cache (dict | None): Open ParquetWriters.

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
        key_tuple = key if isinstance(key, tuple) else (key,)
        subdir = os.path.join(
            out_dir,
            *[f"{col}={val}" for col, val in zip(partition_columns, key_tuple)],
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
    params: argparse.Namespace, file_and_dates: np.ndarray, worker: partial, logger: logging.Logger
) -> dict[str, Any]:  # pylint: disable=too-many-arguments
    """
    Process altimetry files in parallel and write results to partitioned Parquet files.

    Iterates over periods (years or year/months), submits files to a process pool,
    collects LazyFrames into a buffer, and flushes to disk every flush_every files.
    Returns a status dictionary with per-period and total ingestion counts.

    Args:
        params (argparse.Namespace): Command line parameters.
        file_and_dates (np.ndarray): Structured array of file paths and date information.
        worker (partial): Partial function for processing files.
        logger (logging.Logger): Logger object.

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
def get_metadata_json(
    params: argparse.Namespace,
    dataset: DatasetHelper,
    status: dict,
    thisgrid: GridArea,
    start_time: float,
) -> None:
    """
    Write processing metadata to a JSON file.

    Merges command line parameters, dataset attributes, processing status and grid
    details into a single dict, appends the elapsed processing time, and writes it
    via write_metadata.

    Args:
        params (argparse.Namespace): Command line parameters.
        dataset (DatasetHelper): CPOM DatasetHelper object.
        status (dict): Processing status dictionary.
        thisgrid (GridArea): CPOM GridArea object.
        start_time (float): Processing start time.
    """
    # get dict from command line parameters
    ds_dict = vars(dataset)
    params_dict = vars(params)

    output: dict[str, Any] = {}
    if ".yml" in params.dataset or ".yaml" in params.dataset:
        output.update(
            {
                **{k: v for k, v in params_dict.items() if k not in ds_dict.keys()},
                **ds_dict,
                **status,
            }
        )
    else:
        output.update({**params_dict, **status})

    output["grid_xmin"] = thisgrid.minxm
    output["grid_ymin"] = thisgrid.minym
    output["grid_crs"] = thisgrid.coordinate_reference_system
    output["grid_x_size"] = thisgrid.grid_x_size
    output["grid_y_size"] = thisgrid.grid_y_size
    output["gridding_time"] = elapsed(start_time)

    # replace these lines by calling function
    write_metadata(
        params,
        get_algo_name(__file__),
        Path(params.out_dir),
        output,
    )


# --------------------------------------#
# Main Function #
# --------------------------------------#


def grid_for_elev_change(args: list[str]) -> None:
    """
    Main entry point to grid altimetry data for elevation change processing.

    Loads the dataset from a YAML file or inline JSON string, clears the output
    directory, processes all files in parallel, and writes metadata on completion.

    Args:
        args (list[str]): Command line arguments, e.g. sys.argv[1:].
    """
    # ------------------------------#
    # 1.Load command line arguments#
    # ------------------------------#
    parsed_args = parse_arguments(args)

    start_time = time.time()
    # ------------------------------#
    # 2. Load Dataset Object
    # ------------------------------#
    dataset_path = Path(parsed_args.dataset)
    if dataset_path.exists() and dataset_path.suffix in [".yml", ".yaml"]:
        # Method 1: Load from YAML file
        dataset = DatasetHelper(
            data_dir=parsed_args.data_input_dir, dataset_yaml=parsed_args.dataset
        )
    else:
        # Method 2: Treat as JSON string (inline config)
        try:
            dataset_config = json.loads(parsed_args.dataset)
            dataset = DatasetHelper(data_dir=parsed_args.data_input_dir, **dataset_config)
        except json.JSONDecodeError:
            sys.exit(
                f"Invalid dataset configuration: {parsed_args.dataset}. "
                f"Must be either a path to a YAML file or a valid JSON string"
            )

    clean_directory(parsed_args, dataset, confirm_regrid=not parsed_args.force_regrid)

    file_and_dates, worker, thisgrid, logger = get_processing_objects(parsed_args, dataset)

    # --------------------------------------------------------#
    # 5. Process each year of data and write to Parquet files #
    # --------------------------------------------------------#
    status = get_data_and_status_multiprocessed(
        params=parsed_args,
        file_and_dates=file_and_dates,
        worker=worker,
        logger=logger,
    )

    # --------------------------------------------------#
    # 6. Write metadata JSON file with gridding details #
    # --------------------------------------------------#
    get_metadata_json(parsed_args, dataset, status, thisgrid, start_time)


if __name__ == "__main__":
    # Parse command line arguments
    grid_for_elev_change(sys.argv[1:])

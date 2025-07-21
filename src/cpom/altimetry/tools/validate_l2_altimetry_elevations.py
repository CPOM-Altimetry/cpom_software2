"""cpom.altimetry.tools.validate_l2_altimetry_elevations.py

This tool compares a selected date-range or month of radar altimetry against laser elevations.
It also allows comparing a laser mission to it's self.

### **Supported Missions**

- **Altimetry Missions**: CS2 (Native L2 and CryoTempo), S3A, S3B, ENVISAT, ERS1, ERS2
- **Reference Missions**: ICESat-2 (ATL06), IceBridge (ILATM2, ILUTP2),
                        Pre-IceBridge (BRMCR2, BLATM2), ICESat1 (GLAH12)

### **Notes to user**
Use 'get_default_variables' to add your dataset to this script, when testing with new datasets.

- When running for Cryotempo, set --cryotempo_modes .
- When comparing to self, set --compare_to_self.
- When comparing to icesat1 , icebridge, pre-icebridge, set --compare_to_historical_reference.

When comparing to historical references, the default radius increases from 20m to 10000m,
to capture enough points for validation. For the best performance alter the radius and
max_diff argument for your usecase, or provide a DEM to correct reference locations

### **Command Line Options**
**Required**
- `reference_dir`: Path to reference dataset.
- `altim_dir`: Path to altimetry dataset.
- `area`: CPOM area.
- `outdir`: Output directory.
- `year`/ `month` or `start_date`/`end_date`

**Optional**
- `date_delta`: +/- days between altimetry and validation points (default: 0).
- `dem`: Provide a DEM to correct location when comparing with a large radius,
        used in 'correct_elevation_using_slope'
- `radius`: Search radius (in meters) for validation points.
- `max_diff`: Maximum elevation difference between matched points.
- `beams`: Space-separated list of IS2 beams to use (e.g., gt1l gt1r).
- `add_vars`: Additional variables to include in output (e.g., uncertainty).
- `cryotempo_modes`: CryoTempo modes to include (required if using CryoTempo data).
- `max_workers`: Number of parallel processing threads.
- `compare_to_historical_reference`: (default: `False`) Set `True` when comparing to :
                IceBridge/PreIcebridge or ICESAT-1. Increases the max radius to 10000m.
                It is advised to use DEM location correction in this mode.
- `compare_to_self`: Compare reference data points to nearby reference points (default: `False`).
- `bins`: Number of bins for output data plot.

For a full list of the command line descriptions run:
validate_l2_altimetry_elevations -h

### **Example Useage**
Run for cs2 data sin and lrm mode, with 2 beams. Parallelised across 20 workers:
for m in {1..12}
do
./validate_against_is2.py
--altim_dir /media/luna/archive/SATS/RA/CRY/L2I/SIN /media/luna/archive/SATS/RA/CRY/L2I/LRM
--reference_dir /media/luna/archive/SATS/LASER/ICESAT-2/ATL-06/versions/006 --year 2022
--month $m --area antarctica_is --outdir /tmp--beams gt1l gt1r
--add_vars uncertainty_variable_name --max_workers 20 &
done

"""

# pylint: disable=C0302

import argparse
import calendar
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import median_absolute_deviation, sigma_clipped_stats
from netCDF4 import Dataset  # pylint: disable=E0611
from pyproj import CRS, Transformer
from scipy.spatial import KDTree

from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area, list_all_area_definition_names_only
from cpom.dems.dems import Dem


def setup_logging(
    log_name="",
    log_file_info="file.log",
    log_file_error="file.err",
):
    """Setup logger"""
    log = logging.getLogger(log_name)
    log.setLevel(logging.INFO)

    log_format = logging.Formatter(
        "%(levelname)s : %(asctime)s : %(message)s", datefmt="%d/%m/%Y %H:%M:%S"
    )

    for log_file, level in [
        (log_file_info, logging.INFO),
        (log_file_error, logging.ERROR),
    ]:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(log_format)
        file_handler.setLevel(level)
        log.addHandler(file_handler)

    return log


def parse_args() -> argparse.Namespace:
    """Argument Parser to load command line arguments.
    Returns:
        argparse.ArgumentParser: Parsed input arguments
    """
    parser = argparse.ArgumentParser(description="Retrieve command line arguments")

    parser.add_argument(
        "--reference_dir",
        help="The input directory containing the validation data.",
        required=True,
    )
    parser.add_argument(
        "--altim_dir",
        nargs="+",
        help="Altimetry data directory('s) \
        Supports multiple directories by providing a space seperated list of path1 path2",
    )
    parser.add_argument("--year", type=int, help="Year (YYYY)", required=False)
    parser.add_argument(
        "--month", type=int, choices=range(1, 13), help="Month (1-12)", required=False
    )
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date for filtering files (format: YYYYMMDD)",
        required=False,
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date for filtering files (format: YYYYMMDD)",
        required=False,
    )
    parser.add_argument(
        "--date_delta",
        type=int,
        help="+/-days between altimetry and validation points",
        required=False,
        default=0,
    )
    parser.add_argument("--area", help="cpom area name", required=True)
    parser.add_argument("--outdir", help="output directory path for results", required=True)
    parser.add_argument(
        "--beams",
        choices={"gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"},
        nargs="+",
        help="[optional] IS2 beams to use. Space separated list: gt1l gt1r gt2l gt2r gt3l gt3r",
    )
    parser.add_argument(
        "--dem",
        type=str,
        help="[optional] DEM, used in 'correct_elevation_using_slope'",
    )
    parser.add_argument("--radius", type=float, default=20.0, help="[optional] search radius in m")
    parser.add_argument(
        "--maxdiff",
        type=float,
        default=100.0,
        help="[optional] maximum allowed difference in m between reference and altimeter points. \
            Differences > are not saved.",
    )
    parser.add_argument(
        "--add_vars",
        nargs="+",
        help="[optional] additional variables in the altimetry file to include in the output."
        "Space-seperated list of : var1 var2 .",
        default=[],
    )
    parser.add_argument(
        "--cryotempo_modes",
        default=None,
        choices={"lrm", "sin", "sar", "all"},
        nargs="+",
        help="[optional] CryoTempo modes to use. Space-separated list of: lrm sin"
        "For non-CryoTempo L2 CS2 data, specify multiple --altim_dir paths instead.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="[optional, default=10] number of worker processes to use",
    )
    parser.add_argument(  # Default is False, becomes True if provided
        "--compare_to_historical_reference",
        action="store_true",
        help="Use historical reference datasets (IceBridge, Pre-IceBridge, ICESat-1) "
        "with expanded search radius",
    )
    parser.add_argument(
        "--compare_to_self",  # Default is False, becomes True if provided
        action="store_true",
        help="[optional] When set, compare reference to self",
    )
    # Plotting arguments #
    parser.add_argument("--bins", default=100, help="[Optional] Output histogram bin number")

    args = parser.parse_args()

    if not args.compare_to_self and args.altim_dir is None:
        parser.error("--altim_dir is required unless --compare_to_self is set")

    if args.area not in list_all_area_definition_names_only():
        parser.error(f"{args.area} not a valid cpom area name")

    if args.compare_to_historical_reference:
        if args.radius == 20:  # Change default if no value passed.
            args.radius = 1e4
    if args.start_date and args.end_date:
        if args.year or args.month:
            parser.error("Cannot provide both 'start_date'/'end_date' and 'year'/'month'")
        if len(args.start_date) != 8 or len(args.end_date) != 8:
            parser.error("Start and end dates must be in YYYYMMDD format")
    return args


def get_variable(nc: Dataset, nc_var_path: str) -> np.ndarray:
    """Retrieve variable from NetCDF file, handling groups if necessary.

    Args:
        nc (Dataset): The dataset object
        nc_var_path (str): The path to the variable within the file,
                        with groups separated by '/'.

    Raises:
        KeyError: If the variable or group is not found in the file.

    Returns:
        np.array: The retrieved variable as an array.
    """
    try:
        parts = nc_var_path.split("/")
        var = nc
        for part in parts:
            var = var[part]
            if var is None:
                raise IndexError(f"NetCDF parameter '{nc_var_path}' not found.")
        return var[:]
    except IndexError as err:
        raise IndexError(f"NetCDF parameter or group {err} not found") from err


def get_default_variables(file: Path) -> dict:
    """
    Return default variable names based on file naming patterns.

    Args:
        file (Path): Path to the input file.

    Returns:
        dict: Dictionary of default variable names
    """
    basename = os.path.basename(file)
    variable_map = {  # CS2 Products
        "SIR_LRMI2": {
            "lat_nadir": "lat_20_ku",
            "lon_nadir": "lon_20_ku",
            "lat": "lat_poca_20_ku",
            "lon": "lon_poca_20_ku",
            "elev": "height_3_20_ku",
        },
        "SIR_SINI2": {
            "lat_nadir": "lat_20_ku",
            "lon_nadir": "lon_20_ku",
            "lat": "lat_poca_20_ku",
            "lon": "lon_poca_20_ku",
            "elev": "height_1_20_ku",
        },
        "S3A_": {  # Sentinel data
            "lat_nadir": "lat_20_ku",
            "lon_nadir": "lon_20_ku",
            "lat": "lat_cor_20_ku",
            "lon": "lon_cor_20_ku",
            "elev": "elevation_ocog_20_ku_filt",
        },
        "S3B_": {  # Sentinel data
            "lat_nadir": "lat_20_ku",
            "lon_nadir": "lon_20_ku",
            "lat": "lat_cor_20_ku",
            "lon": "lon_cor_20_ku",
            "elev": "elevation_ocog_20_ku_filt",
        },
        "CS_OFFL_SIR_TDP": {  # Cryotempo
            "lat": "latitude",
            "lon": "longitude",
            "elev": "elevation",
        },
        # FDR4ALT Products
        "ER1": {
            "lat_nadir": "expert/ice_sheet_lat_poca",
            "lon_nadir": "expert/ice_sheet_lon_poca",
            "lat": "expert/latitude",
            "lon": "expert/longitude",
            "elev": "expert/ice_sheet_elevation_ice1_roemer",
        },
        "ER2": {
            "lat_nadir": "expert/ice_sheet_lat_poca",
            "lon_nadir": "expert/ice_sheet_lon_poca",
            "lat": "expert/latitude",
            "lon": "expert/longitude",
            "elev": "expert/ice_sheet_elevation_ice1_roemer",
        },
        "EN1": {
            "lat_nadir": "expert/ice_sheet_lat_poca",
            "lon_nadir": "expert/ice_sheet_lon_poca",
            "lat": "expert/latitude",
            "lon": "expert/longitude",
            "elev": "expert/ice_sheet_elevation_ice1_roemer",
        },
        "ATL06": {  # Laser Altimetry #
            "lat": "{beam}/land_ice_segments/latitude",
            "lon": "{beam}/land_ice_segments/longitude",
            "elev": "{beam}/land_ice_segments/h_li",
            "date_pattern": "YYYYMMDDHHMMSS_",
        },
        # iceSAT1
        "GLAH12": {
            "lat": "Data_40HZ/Geolocation/d_lat",
            "lon": "Data_40HZ/Geolocation/d_lon",
            "elev": "Data_40HZ/Elevation_Surfaces/d_elev",
            "date_pattern": "",
        },
        # Icebridge
        "ILATM2": {"lat": "lat", "lon": "lon", "elev": "hgt"},
        "ILUTP2": {"lat": "lat", "lon": "lon", "elev": "hgt"},
        "BLATM2": {"lat": "lat", "lon": "lon", "elev": "height"},
        "BRMCR2": {"lat": "lat", "lon": "lon", "elev": "elevation"},
        "ENVISAT_functional_TDS": {
            "lat": "latitude",
            "lon": "longitude",
            "elev": "elevation",
        },
    }

    for key, value in variable_map.items():  # Check for a matching pattern
        if key in basename:
            return value
    return {}


def get_date_from_filename(file: Path, log) -> Optional[datetime]:
    """Extract date from filename based on the regex patterns.
    Args:
        file (Path): Path to the input file.
    Returns:
        datetime: Extracted date from the filename.
    """
    match = re.compile(r"(?:\d{8}T|\d{8}_|\d{14}_)").findall(str(file))
    date_candidates = []

    for _, date_str in enumerate(match):
        try:
            if len(date_str) == 9 or len(date_str) == 15:
                dt = datetime.strptime(date_str[:8], "%Y%m%d")
                date_candidates.append(dt)
        except ValueError:
            continue

    if len(date_candidates) == 1:
        return date_candidates[0]
    if len(date_candidates) > 1:
        # Get difference between dates
        date_diffs = [
            abs((date_candidates[i] - date_candidates[j]).days)
            for i in range(len(date_candidates))
            for j in range(i + 1, len(date_candidates))
        ]
        # If the difference is less than 2 days, return the first date
        if all(diff < 2 for diff in date_diffs):
            return date_candidates[0]
        log.error(f"Multiple possible dates found in file {file}: {date_candidates}")
        return None

    if len(match) == 0:
        # Check for match to ICESat1 dateformat /YYYY.MM.DD/
        dot_match = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})").findall(str(file))
        if len(dot_match) == 1:
            try:
                return datetime.strptime(".".join(dot_match[0]), "%Y.%m.%d")
            except ValueError:
                log.error(f"Invalid date {dot_match[0]} in file {file}")

    log.error(f"No valid date found in file {file}")
    return None


def get_files_in_dir(
    directory: Path,
    args: argparse.Namespace,
    log: logging.Logger,
    filetype: str = "nc",
    date_delta: int = 0,
) -> dict[Path, str]:
    """
    Find .nc or.h5 files in the specified directory.
    This function searches for files in the given directory and its subdirectories,
    filtering them based on the specified year, month, or optional date range.

    Args:
        directory (Path): The directory to search for files.
        args (argparse.Namespace): Parsed command line arguments containing
        year, month, start_date, end_date, and date_delta.
        filetype (str) : nc or h5
    Returns:
        List[Path]: A list of found .nc or .NC files with their full paths.
    """

    extensions = {"nc": "*.nc", "h5": "*.h5", "H5": "*.H5"}
    all_files: list[Path] = []
    all_files.extend(Path(directory).rglob(extensions[filetype]))

    if args.start_date:
        first_day = datetime.strptime(args.start_date, "%Y%m%d") - timedelta(days=date_delta)
        last_day = datetime.strptime(args.end_date, "%Y%m%d") + timedelta(days=date_delta)
    else:
        year = str(args.year)
        month = str(args.month).zfill(2)
        first_day = datetime.strptime(f"{year}{int(month):02d}01", "%Y%m%d") - timedelta(
            days=date_delta
        )
        last_day = datetime.strptime(
            f"{year}{int(month):02d}{calendar.monthrange(int(year), int(month))[1]:02d}",
            "%Y%m%d",
        ) + timedelta(days=date_delta)

    valid_files = {}
    for file in all_files:
        if "." in file.stem:
            continue
        try:
            date_obj = get_date_from_filename(file, log)
            if date_obj is not None and first_day <= date_obj <= last_day:
                valid_files[file] = date_obj.strftime("%Y%m%d")
        except TypeError:
            pass

    return valid_files


class ProcessData:
    """Class to process laser and altimetry data files by extracting and filtering
    elevation data. To be run by a multiproccessor.

    Methods:
        get_altimetry_data_array: Extracts and processes altimetry data from NetCDF files.
        get_is2_data_array : Extracts and processes is1 elevation data from HDF5 files.
        get_is2_data_array : Extracts and processes is2 elevation data from HDF5 files.
        get_icebridge_data_array : Extracts and processes icebridge data from NetCDF files.
        fill_empty_latlon_with_nadir: Fills missing lat/lon values using nadir coordinates.
        get_cryotempo_filters : Filters cryotempo data to valid modes.
    """

    def __init__(self, args, area, valid_files, log):
        self.args = args
        self.area = area
        self.valid_files = valid_files
        self.log = log
        self.dtype_with_date = [
            ("x", "float64"),
            ("y", "float64"),
            ("h", "float64"),
            ("date", "U8"),
        ]
        self.dtype_no_date = [("x", "float64"), ("y", "float64"), ("h", "float64")]

    def get_cryotempo_filters(self, nc, args):
        """Check Cryotempo data is in a valid mode."""
        if "all" in args.cryotempo_modes:
            return None
        mode_map = {"lrm": 1, "sar": 2, "sin": 3}
        valid_modes = {mode_map[mode] for mode in args.cryotempo_modes if mode in mode_map}
        if not valid_modes:
            return None
        valid_mask = np.isin(get_variable(nc, "instrument_mode"), list(valid_modes))
        return valid_mask if valid_mask.any() else None

    def fill_empty_latlon_with_nadir(
        self, nc: str, lat: np.ndarray, lon: np.ndarray, config: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Populate empty poca lat/lon with nadir lat/lon
        Args:
            nc (str): NetCDF filename
            lat (np.array): Array of latitudes
            lon (np.array): Array of longitudes
            config (dict): Dictionary of variable names
        Returns:
            (np.array, np.arrray): Filled lat/lon arrays
        """
        bad_indices = np.isnan(lon)  # np.flatnonzero(lon.mask)  # Find empty longitude values
        if bad_indices.size > 0:
            lat_nadir = get_variable(nc, config["lat_nadir"])
            lon_nadir = get_variable(nc, config["lon_nadir"])
            lat[bad_indices] = lat_nadir[bad_indices]
            lon[bad_indices] = np.mod(lon_nadir[bad_indices], 360)
        return lat, lon

    def get_altimetry_data_array(self, filename: Path) -> Optional[np.ndarray]:
        """Extract and filter data from an altimetry data file.
        Args:
            filename (Path): Altimetry data filename (*.nc)
        Raises:
            ValueError: If variable lengths do not match.
        Returns:
            np.ndarray: Structured nd.array containing x,y,h fields
        """
        try:
            with Dataset(filename) as nc:
                config = get_default_variables(filename)
                if not config:
                    self.log.error("Unsupported file basename %s for file", filename)
                    return None

                lats, lons, elev = (
                    get_variable(nc, config["lat"]),
                    np.mod(get_variable(nc, config["lon"]), 360),
                    get_variable(nc, config["elev"]),
                )
                additional_data = {
                    var.rsplit("/", 1)[-1]: get_variable(nc, var)
                    for var in self.args.add_vars
                    if var in nc.variables
                }
                if list(additional_data.keys()) != self.args.add_vars:
                    self.log.info(
                        "Variable(s) %s missing from netcdf",
                        set(self.args.add_vars) - set(additional_data.keys()),
                    )

                if "lat_nadir" in config and "lon_nadir" in config:
                    lats, lons = self.fill_empty_latlon_with_nadir(nc, lats, lons, config)

                if self.args.cryotempo_modes:
                    mask = self.get_cryotempo_filters(nc, self.args)
                    if mask is not None:
                        lats, lons, elev = lats[mask], lons[mask], elev[mask]
                        for k in additional_data:
                            additional_data[k] = additional_data[k][mask]

                lats, lons, _, _ = self.area.inside_latlon_bounds(lats, lons)
                x, y = self.area.latlon_to_xy(lats, lons)

                idx, _ = self.area.inside_mask(x, y)
                if not idx.size:
                    return None
                try:
                    x, y, elev = x[idx], y[idx], elev[idx]
                    for k in additional_data:
                        additional_data[k] = additional_data[k][idx]
                except IndexError:
                    self.log.error(f"Mismatch in variable lengths for {filename}")
                    return None

            # Structured NumPy array containing x, y, elevation, and any additional variables
            if self.args.date_delta != 0:
                date_arr = np.full(len(x), self.valid_files[filename], dtype="U8")

                if len(date_arr) != len(x) or len(date_arr) != len(y) or len(date_arr) != len(elev):
                    self.log.error(f"Mismatch in variable lengths for {filename} with date array")
                    raise ValueError(f"Mismatch in variable lengths for {filename} with date array")
                return np.array(
                    list(
                        zip(
                            x,
                            y,
                            elev,
                            date_arr,
                            *(additional_data[var] for var in additional_data),
                        )
                    ),
                    dtype=self.dtype_with_date
                    + [(var, str(data.dtype)) for var, data in additional_data.items()],
                )
            return np.array(
                list(zip(x, y, elev, *(additional_data[var] for var in additional_data))),
                dtype=self.dtype_no_date
                + [(var, str(data.dtype)) for var, data in additional_data.items()],
            )
        except OSError as err:
            self.log.error(f"Error loading altimetry data file {filename}, failed with : {err}")
            return None

    def get_is2_data_array(self, filename: Path) -> Optional[np.ndarray]:
        """Extract and filter data from an is2 data file.
        Args:
            filename (Path): IS2 data filename (.h5)
        Raises:
            ValueError: If variable lengths do not match
        Returns:
            np.ndarray: Structured ndarray with fields x, y, h (and optionally date)
        """
        points = []
        try:
            with h5py.File(filename, "r") as nc:
                for beam in self.args.beams:
                    config = {
                        key: path.format(beam=beam)
                        for key, path in get_default_variables(filename).items()
                    }

                    hemisphere = get_variable(nc, config["lat"])[0]
                    if (hemisphere < 0.0 and self.area.hemisphere == "north") or (
                        hemisphere > 0.0 and self.area.hemisphere == "south"
                    ):
                        continue

                    elevation = get_variable(nc, config["elev"])
                    ok = np.flatnonzero(
                        (get_variable(nc, f"{beam}/land_ice_segments/atl06_quality_summary") == 0)
                        & (elevation <= 10e3)
                    )
                    if len(ok) == 0:
                        continue

                    lats, lons, elevation = (
                        get_variable(nc, config["lat"])[ok],
                        np.mod(get_variable(nc, config["lon"]), 360)[ok],
                        elevation[ok],
                    )

                    # Filter on area's bounding box (not mask, for speed)
                    lats, lons, bounded_indices, _ = self.area.inside_latlon_bounds(lats, lons)

                    if len(lats) < 1:
                        continue

                    # Transform lat, lon to x,y in appropriate polar stereo projections
                    elevation = elevation[bounded_indices]
                    x, y = self.area.latlon_to_xy(lats, lons)

                    if not len(x) == len(y) == len(elevation):
                        continue
                    # Structured NumPy array containing x, y, elevation
                    if self.args.date_delta != 0:
                        points.append(
                            np.array(
                                list(
                                    zip(
                                        x,
                                        y,
                                        elevation,
                                        np.full(
                                            len(x),
                                            self.valid_files[filename],
                                            dtype="U8",
                                        ),
                                    )
                                ),
                                dtype=self.dtype_with_date,
                            )
                        )
                    else:
                        points.append(
                            np.array(
                                list(zip(x, y, elevation)),
                                dtype=self.dtype_no_date,
                            )
                        )
        except OSError as err:
            self.log.error(f"Error loading atl06 data file {filename} failed with : {err}")
            return np.array(
                [],
                dtype=(self.dtype_no_date if self.args.date_delta == 0 else self.dtype_with_date),
            )

        return (
            np.concatenate(points)
            if points
            else np.array(
                [],
                dtype=(self.dtype_no_date if self.args.date_delta == 0 else self.dtype_with_date),
            )
        )  # Concat point arrays into one, or return an empty structured array if none exist

    def get_icebridge_data_array(self, filename: Path) -> Optional[np.ndarray]:
        """Extract and filter icebridge and pre-icebridge data.
        Args:
            filename (Path): Filename
        Raises:
            ValueError: If arrays not equal in length
        Returns:
            np.ndarray: Structured nd.array containing x,y,h fields
        """
        with Dataset(filename) as nc:
            config = get_default_variables(filename)
            lats, lons = get_variable(nc, config["lat"]), np.mod(
                get_variable(nc, config["lon"]), 360
            )
            lats, lons, bounded_indices, _ = self.area.inside_latlon_bounds(lats, lons)
            if len(lats) == 0:
                return None

            x, y = self.area.latlon_to_xy(lats, lons)
            elevation = get_variable(nc, config["elev"])[bounded_indices]
            if not len(x) == len(y) == len(elevation):
                self.log.error(f"Mismatch in variable lengths for x,y,h in file {filename}")
                return None

            if self.args.date_delta != 0:
                date_arr = np.full(len(x), self.valid_files[filename], dtype="U8")
                return np.array(
                    list(zip(x, y, elevation, date_arr)),
                    dtype=self.dtype_with_date,
                )
            return np.array(
                list(zip(x, y, elevation)),
                dtype=self.dtype_no_date,
            )

    def get_is1_data_array(self, filename: Path) -> Optional[np.ndarray]:  # pylint: disable=R0914
        """Extract and filter IS1 data from a file.
        Follows method from Smith et al., Science(2020)
        Args:
            filename (Path): IS1 data filename
        Raises:
            ValueError: If variable lengths do not match
        Returns:
            np.ndarray: Structured nd.array containing x,y,h fields
        """

        def _get_crs_transformers():
            """Get the CRS and transformers for converting coordinates from TOPEX to WGS84."""
            # Convert geodetic coordinates to cartesian coordinates
            topex_crs = CRS.from_proj4("+proj=latlong +a=6378136.300 +rf=298.257 +no_defs")
            ecef_crs = CRS.from_proj4("+proj=geocent  +a=6378136.300 +rf=298.257 +no_defs")
            return (
                Transformer.from_crs(topex_crs, ecef_crs, always_xy=True),
                Transformer.from_crs(ecef_crs, CRS.from_epsg(4979), always_xy=True),
            )

        with h5py.File(filename, "r") as file:
            config = get_default_variables(filename)
            # Get the hemisphere of the file
            hemisphere = get_variable(file, config["lat"])[0]
            if (hemisphere < 0.0 and self.area.hemisphere == "north") or (
                hemisphere > 0.0 and self.area.hemisphere == "south"
            ):
                return None

            elevation_tpx = get_variable(file, config["elev"])
            valid_mask = (
                (file["Data_40HZ"]["Waveform"]["i_numPk"][:] == 1)  # N. peaks in returned echo
                & (file["Data_40HZ"]["Quality"]["elev_use_flg"][:] == 0)  # Flag to use elevation
                & (file["Data_40HZ"]["Quality"]["sigma_att_flg"][:] == 0)  # Attitude quality flag
                & (file["Data_40HZ"]["Quality"]["sat_corr_flg"][:] < 3)  # Saturation corr flag
                & (elevation_tpx != file[config["elev"]].attrs["_FillValue"])
            )

            if not valid_mask.any():
                self.log.error("No valid data found in file %s", filename)
                return None

            lats_tpx, lons_tpx, elevation_tpx = (
                get_variable(file, config["lat"])[valid_mask],
                get_variable(file, config["lon"])[valid_mask],
                elevation_tpx[valid_mask],
            )
            sat_corr = file["Data_40HZ"]["Elevation_Corrections"]["d_satElevCorr"][:][valid_mask]

            topex_to_ecef, ecef_to_wgs = _get_crs_transformers()
            # Convert geodetic coordinates to cartesian coordinates
            x_ecef, y_ecef, h_ecef = topex_to_ecef.transform(lons_tpx, lats_tpx, elevation_tpx)
            # Convert tpx cartesian coordinates to wgs84
            lons, lats, elevation = ecef_to_wgs.transform(x_ecef, y_ecef, h_ecef)
            # Transform lat, lon to x,y in appropriate polar stereo projections
            lons = np.mod(lons, 360)
            lats, lons, bounded_indices, _ = self.area.inside_latlon_bounds(lats, lons)
            x, y = self.area.latlon_to_xy(lats, lons)

            # Apply saturation correction
            elevation = elevation[bounded_indices] + sat_corr[bounded_indices]

            if not len(x) == len(y) == len(elevation):
                self.log.error(f"Mismatch in variable lengths for x,y,h in file {filename}")
                return None

            if self.args.date_delta != 0:
                date_arr = np.full(len(x), self.valid_files[filename], dtype="U8")
                return np.array(
                    list(zip(x, y, elevation, date_arr)),
                    dtype=self.dtype_with_date,
                )
            return np.array(
                list(zip(x, y, elevation)),
                dtype=self.dtype_no_date,
            )

    def process_reference_file(self, file_path: Path):
        """Process reference files,
        determines which function to call based on file type.
        nc : Icebridge / pre-icebridge
        h5 : IS2
        H5 : IS1
        """
        try:
            if self.args.compare_to_historical_reference and file_path.suffix == ".nc":
                return self.get_icebridge_data_array(file_path)
            if self.args.compare_to_historical_reference and file_path.suffix == ".H5":
                return self.get_is1_data_array(file_path)
            return self.get_is2_data_array(file_path)
        except ValueError as err:
            self.log.error("Error processing file %s: %s", file_path, err)
            return None


# pylint: disable=R0914
def get_elev_differences(
    args: argparse.Namespace,
    laser_points: np.ndarray,
    altimeter_points: np.ndarray,
    prefix: str = "",
    nearest_only=False,
) -> dict:
    """
    Calculate the elevation differences between IS2 points and altimeter points
    within a specified search radius and maximum elevation difference.
    Args:
        args (argparse.Namespace): Configuration parameters
        laser_points (np.ndarray): Is2 array containing x,y,h
        altimeter_points (np.ndarray): Altimetry array containing x,y,h
        prefix(str): Reference dataset prefix.
        nearest_only (Boolean): Compare altimetry to only the nearest reference point
    Returns:
        dict: Dh between altimetry and is2 elevations.
    """

    is2_tree = KDTree(np.c_[laser_points["x"], laser_points["y"]])
    if args.date_delta > 0:
        add_vars = [
            var for var in altimeter_points.dtype.names if var not in {"x", "y", "h", "date"}
        ]
    else:
        add_vars = [var for var in altimeter_points.dtype.names if var not in {"x", "y", "h"}]

    results: dict = {
        key: []
        for key in [
            "dh",
            "sep_dist",
            f"{prefix}x",
            f"{prefix}y",
            f"{prefix}h",
            "reference_x",
            "reference_y",
            "reference_h",
        ]
        + add_vars
        + ([f"{prefix}date", "reference_date"] if args.date_delta > 0 else [])
    }
    for altimeter_point in altimeter_points:
        if nearest_only:
            dist, idx = is2_tree.query((altimeter_point["x"], altimeter_point["y"]))
            indices = [idx] if dist <= args.radius else []
        else:
            indices = is2_tree.query_ball_point(  # Find reference points within search_radius
                (altimeter_point["x"], altimeter_point["y"]), args.radius
            )

        if args.date_delta > 0:
            valid_indices = []
            for i in indices:
                date_diff = abs(
                    (
                        datetime.strptime(altimeter_point["date"], "%Y%m%d")
                        - datetime.strptime(laser_points[i]["date"], "%Y%m%d")
                    ).days
                )
                if date_diff <= args.date_delta:  # Maximum date difference
                    valid_indices.append(i)
            indices = valid_indices

        if args.compare_to_self:  # Remove comparision to self
            indices = [
                i
                for i in indices
                if (laser_points[i]["x"], laser_points[i]["y"])
                != (altimeter_point["x"], altimeter_point["y"])
            ]

        for idx in indices:
            ref_point = laser_points[idx]
            dh = altimeter_point["h"] - ref_point["h"]

            if np.abs(dh) <= args.maxdiff:
                results["dh"].append(dh)
                results["sep_dist"].append(
                    np.sqrt(
                        (altimeter_point["x"] - ref_point["x"]) ** 2
                        + (altimeter_point["y"] - ref_point["y"]) ** 2
                    )
                )
                results[f"{prefix}x"].append(altimeter_point["x"])
                results[f"{prefix}y"].append(altimeter_point["y"])
                results[f"{prefix}h"].append(altimeter_point["h"])
                results["reference_x"].append(ref_point["x"])
                results["reference_y"].append(ref_point["y"])
                results["reference_h"].append(ref_point["h"])

                for var in add_vars:
                    results[var].append(altimeter_point[var])

                if args.date_delta > 0:
                    results[f"{prefix}date"].append(altimeter_point["date"])
                    results["reference_date"].append(ref_point["date"])
    return results


def correct_elevation_using_slope(data, args, log, prefix) -> dict:
    """Correct for differences in the measurement location
    of the reference and altimetry data point.
    Conceptually shift reference location, to be located at altimetry datapoint,
    using DEM slope.
    Args:
        data (nd.array): Output of "get_elev_differences"
        args (argparse.Namespace): Configuration parameters
        log (Logger): Logger
        prefix (str): Reference data prefix
    Returns
       dict: Slope corrected Dh between altimetry and is2 elevations.
    """
    log.info("Performing slope correction")
    dem = Dem("arcticdem_100m_greenland_v4.1")

    dh_dem_elev = (dem.interp_dem(data[f"{prefix}x"], data[f"{prefix}y"])) - (
        dem.interp_dem(data["reference_x"], data["reference_y"])
    )
    data["reference_h"] = data["reference_h"] + dh_dem_elev
    data["dh"] = data["dh"] - dh_dem_elev

    data = {key: np.array(data[key]) for key in data}
    valid_mask = (~np.isnan(data["dh"])) & (np.abs(data["dh"]) <= args.maxdiff)

    return {key: val[valid_mask] for key, val in data.items()}


######## Plotting and statistics ########
def elev_dh_histograms(dh, outdir, bins=100):
    """dh frequency histograms"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # Create 1 row, 2 columns
    ax[0].hist(dh, bins, color="tab:blue", edgecolor="black", label=f"dh:{bins}-bins")
    ax[0].set_xlabel("Elevation Difference (m)", size=10)
    ax[0].set_ylabel("Frequency")
    ax[0].set_title("Distribution of Elevation Differences", fontsize=10)
    ax[0].grid(linestyle=":", color="tab:grey", alpha=0.5)
    ax[0].legend()

    ax[1].hist(dh, bins, color="tab:purple", edgecolor="black", label=f"dh:{bins}-bins")
    ax[1].set_xlabel("Elevation Difference (m)", size=10)
    ax[1].set_ylabel("Frequency")
    ax[1].set_title("Distribution of Elevation Differences", fontsize=10)
    ax[1].grid(linestyle=":", color="tab:grey", alpha=0.5)
    ax[1].set_yscale("log")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(outdir, bbox_inches="tight", dpi=300)
    plt.close(fig)


def elevation_dh_cumulative_dist(dh, out_dir):
    """Cumulative distribution of dh"""
    sorted_diff = np.sort(dh)
    cdf = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sorted_diff, cdf, linestyle="-", color="tab:purple", label="dh")
    ax.set_xlabel("Elevation Difference (m)", size=10)
    ax.set_ylabel("Cumulative Probability", size=10)
    ax.set_title("Cumulative Distribution of Elevation Differences", size=10)
    ax.grid(linestyle=":", color="tab:grey", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir, bbox_inches="tight", dpi=300)
    plt.close(fig)


def compute_elevation_stats(output, prefix, output_file="elevation_stats.txt"):
    """
    Compute statistics for elevation differences.
    Args:
        output (dict): Dictionary containing summary statistics.
    Writes:
        CSV file containing summary statistics.
    """
    num_points = len(output["dh"])

    # Compute R² value for height correlation
    p2p_r2_hgt = np.corrcoef(output[f"{prefix}h"], output["reference_h"])[0, 1] ** 2
    mean_sepdist, stddev_sepdist = np.mean(output["seperation_distance"]), np.std(
        output["seperation_distance"]
    )  # Compute separation distance metrics

    def compute_stats(data):
        return [
            np.mean(data),
            np.median(data),
            np.std(data),
            median_absolute_deviation(data),
            np.sqrt(np.mean(np.square(data))),
            *sigma_clipped_stats(data, sigma=2)[:3:2],
        ]

    def format_stats(data):
        return [num_points] + compute_stats(data) + [p2p_r2_hgt, mean_sepdist, stddev_sepdist]

    # Compute stats for both signed and absolute differences
    pd.DataFrame(
        [format_stats(output["dh"]), format_stats(np.abs(output["dh"]))],
        index=["Signed Differences", "Absolute Differences"],
        columns=[
            "num_points",
            "mean_dh",
            "median_dh",
            "std_dh",
            "MAD_dh",
            "RMS_dh",
            "2_sigma_mean_dh",
            "2_sigma_std_dh",
            "r2_hgt",
            "mean_dist_(m)",
            "std_dist_(m)",
        ],
    ).to_csv(output_file)


if __name__ == "__main__":
    params = parse_args()
    AREA_OBJ = Area(params.area)

    if params.start_date and params.end_date:
        output_dir = (
            Path(params.outdir)
            / f"{datetime.strptime(params.start_date, '%Y%m%d').strftime('%Y%m%d')}\
            _{datetime.strptime(params.end_date, '%Y%m%d').strftime('%Y%m%d')}"
        )
    else:
        DATE_YEAR = str(params.year)
        DATE_MONTH = str(params.month).zfill(2)
        output_dir = Path(params.outdir) / f"{DATE_YEAR}{DATE_MONTH}"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(
            log_file_info=output_dir / "info.log",
            log_file_error=output_dir / "errors.log",
        )
        with open(output_dir / "command_line_args.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(sys.argv) + "\n")
    except OSError as e:
        sys.exit(f"Failed to create directory {output_dir}: {e}")

    logger.info("Start processing with arguments %s", params)

    if params.date_delta != 0 or params.start_date:
        reference_dir = Path(params.reference_dir)
    else:
        reference_path = Path(params.reference_dir) / f"{DATE_YEAR}/{DATE_MONTH}"
        reference_dir = reference_path if reference_path.is_dir() else Path(params.reference_dir)

    if params.compare_to_historical_reference:  # icebridge/pre-icebridge
        reference_file_date = get_files_in_dir(
            reference_dir, params, logger, "nc", date_delta=params.date_delta
        )
        if not reference_file_date:
            reference_file_date = get_files_in_dir(
                reference_dir, params, logger, "H5", date_delta=params.date_delta
            )  # is1
    else:
        reference_file_date = get_files_in_dir(
            reference_dir, params, logger, "h5", date_delta=params.date_delta
        )  # is2

    logger.info("Loaded %d reference data files", len(reference_file_date))

    processor = ProcessData(params, AREA_OBJ, reference_file_date, logger)
    reference_files = list(reference_file_date.keys())
    chunksize = max(1, len(reference_files) // (params.max_workers * 4))

    logger.info("Chunksize %d", chunksize)
    with ProcessPoolExecutor(max_workers=params.max_workers) as executor:
        ref_results = list(
            executor.map(processor.process_reference_file, reference_files, chunksize=chunksize)
        )
    reference_points = np.concatenate([result for result in ref_results if result is not None])
    unique_indices = np.unique(reference_points[["x", "y", "h"]], return_index=True)[1]
    reference_points = reference_points[unique_indices]
    logger.info("Loaded reference data points, len : %d", len(reference_points))

    # Load altimetry data #
    if params.compare_to_self:
        logger.info("Performing reference self-comparison")
        PREFIX = "neighbour_"
        altimetry_points = reference_points  # Compare is2 to itself
    else:
        logger.info("Comparing reference to altimetry")
        PREFIX = ""
        altimetry_files = {}
        for basepath in params.altim_dir:  # Can have multiple altimetry directories e.g. LRM/SIN
            if params.start_date:
                altimetry_dir = Path(basepath)
            else:
                alt_path = Path(basepath) / f"{DATE_YEAR}/{DATE_MONTH}"
                altimetry_dir = alt_path if alt_path.is_dir() else Path(basepath)

            altimetry_files.update(
                get_files_in_dir(altimetry_dir, params, logger, "nc", date_delta=0)
            )

        logger.info("Loaded %d altimetry data files", len(altimetry_files))

        chunksize = max(1, len(altimetry_files) // (params.max_workers * 4))
        logger.info("Chunksize %d", chunksize)
        processor.valid_files = altimetry_files  # Set valid files for processor
        with ProcessPoolExecutor(max_workers=params.max_workers) as executor:
            altim_results = list(
                executor.map(
                    processor.get_altimetry_data_array,
                    altimetry_files,
                    chunksize=chunksize,
                )
            )

        altimetry_points = np.concatenate(
            [result for result in altim_results if result is not None]
        )
        unique_indices = np.unique(altimetry_points[["x", "y", "h"]], return_index=True)[1]
        altimetry_points = altimetry_points[unique_indices]
        logger.info("Loaded altimetry data points, len : %d", len(altimetry_points))

    outfile = output_dir / f"p2p_diffs_{params.area}"

    elev_differences = get_elev_differences(params, reference_points, altimetry_points, PREFIX)

    logger.info("Found %d elevation differences", len(elev_differences["dh"]))
    if params.dem:
        elev_differences = correct_elevation_using_slope(elev_differences, params, logger, PREFIX)

    # Convert to lat/lon #
    alt_lats, alt_lons = AREA_OBJ.xy_to_latlon(
        elev_differences[f"{PREFIX}x"], elev_differences[f"{PREFIX}y"]
    )
    reference_lats, reference_lons = AREA_OBJ.xy_to_latlon(
        elev_differences["reference_x"], elev_differences["reference_y"]
    )

    # Output #
    logger.info("Saving month data to %s.npz", outfile)
    save_data = {
        "dh": elev_differences["dh"],
        "seperation_distance": elev_differences["sep_dist"],
        f"{PREFIX}lons": alt_lons,
        f"{PREFIX}lats": alt_lats,
        f"{PREFIX}x": elev_differences[f"{PREFIX}x"],
        f"{PREFIX}y": elev_differences[f"{PREFIX}y"],
        f"{PREFIX}h": elev_differences[f"{PREFIX}h"],
        "reference_lons": reference_lons,
        "reference_lats": reference_lats,
        "reference_x": elev_differences["reference_x"],
        "reference_y": elev_differences["reference_y"],
        "reference_h": elev_differences["reference_h"],
    }

    # Include any additional variables
    for variable in elev_differences:
        if variable not in save_data:
            save_data[variable] = elev_differences[variable]

    np.savez(f"{outfile}.npz", **save_data)

    compute_elevation_stats(save_data, prefix=PREFIX, output_file=f"{outfile}_elevation_stats.csv")

    Polarplot(params.area).plot_points(
        {
            "name": "Difference_in_height_(dh)",
            "lats": save_data[f"{PREFIX}lats"],
            "lons": save_data[f"{PREFIX}lons"],
            "vals": save_data["dh"],
        },
        output_dir=str(output_dir),
    )
    elev_dh_histograms(save_data["dh"], f"{outfile}_dh_histogram.png", bins=params.bins)
    elevation_dh_cumulative_dist(save_data["dh"], f"{outfile}_dh_cumulative_distribution.png")

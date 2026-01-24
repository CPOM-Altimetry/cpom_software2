"""cpom.altimetry.tools.validate_l2_altimetry_elevations.py

This tool compares elevations between a Altimetry and Laser dataset over a selected date-range.
Also supports validating data against nearby measurements.

### **Supported Missions**

- **Altimetry Missions**: CS2 (Native L2 and CryoTempo), S3A, S3B, ENVISAT, ERS1, ERS2
- **Reference Missions**: ICESat-2 (ATL06), IceBridge (ILATM2, ILUTP2),
                        Pre-IceBridge (BRMCR2, BLATM2), ICESat1 (GLAH12)

### **Notes to user**
Use 'get_default_variables' to add your dataset to this script, when testing with new datasets.

- When running for Cryotempo, set --cryotempo_modes .
- When comparing to self, set --compare_to_self.
- When comparing to icesat1 , icebridge, pre-icebridge, set --compare_to_historical_reference.

When comparing to historical references, the default search distance increases from 20m to 10000m,
to capture enough points for validation. For the best performance alter the min_dist, max_dist and
max_diff argument for your usecase, or provide a DEM to correct reference locations

### **Command Line Options**
**Required**
- `reference_dir`: Path to reference dataset.
- `altim_dir`: Path to altimetry dataset.
- `area`: CPOM area.
- `outdir`: Output directory.
- `year`/ `month` or `start_date`/`end_date`

For a full list of the command line options and descriptions run:
validate_l2_altimetry_elevations -h

### **Example Useage**
Run for cs2 data sin and lrm mode, with 2 beams. Parallelised across 20 workers:
./validate_against_is2.py
--altim_dir /media/luna/archive/SATS/RA/CRY/L2I/SIN /media/luna/archive/SATS/RA/CRY/L2I/LRM
--reference_dir /media/luna/archive/SATS/LASER/ICESAT-2/ATL-06/versions/006 --year 2022
--month 01 --area antarctica_is --outdir /tmp--beams gt1l gt1r
--add_vars uncertainty_variable_name --max_workers 20 &
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
from typing import Callable, Iterable, Optional

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
    log_name: str = "",
    log_file_info: str | Path = "file.log",
    log_file_error: str | Path = "file.err",
) -> logging.Logger:
    """Create a logger with separate info/error handlers.

    Args:
        log_name: Logger name (empty for root-style logger).
        log_file_info: Path for info-level messages.
        log_file_error: Path for error-level messages.

    Returns:
        logging.Logger: Configured logger instance.
    """
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
    """Parse CLI arguments for altimetry validation.

    Supports two date-selection modes (year/month or start/end range), optional
    date padding via `date_delta`, and switches for self-comparison versus
    altimetry-to-reference comparisons.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate altimetry elevations against reference laser datasets. "
            "Supports year/month or start/end date selection, optional date padding, "
            "and self-comparison modes."
        )
    )

    parser.add_argument(
        "--reference_dir",
        help=(
            "Directory containing reference (laser) files used for validation. "
            "Searched recursively."
        ),
        required=True,
    )
    parser.add_argument(
        "--reference_filetype",
        choices={"nc", "h5", "H5"},
        default="h5",
        help="Filetype of reference data files (default: h5)",
    )
    parser.add_argument(
        "--altim_dir",
        nargs="+",
        help=("Altimetry data directories (space separated). Each is searched recursively. "),
    )
    parser.add_argument(
        "--altimetry_filetype",
        choices={"nc", "h5", "H5"},
        default="nc",
        help="Filetype of altimetry data files (default: nc)",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Year (YYYY) used with --month to filter filenames.",
        required=False,
    )
    parser.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        help="Month (1-12) used with --year to filter filenames.",
        required=False,
    )
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date (YYYYMMDD) for filtering files",
        required=False,
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date (YYYYMMDD) for filtering files",
        required=False,
    )
    parser.add_argument(
        "--date_delta",
        type=int,
        help="+/- days allowed between altimetry and reference timestamps when matching points.",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--area",
        help=(
            "CPOM area name; sets spatial mask and projection for filtering and coordinate transforms."
        ),
        required=True,
    )
    parser.add_argument(
        "--outdir",
        help="Output directory for logs, csv/npz, and plots.",
        required=True,
    )
    parser.add_argument(
        "--beams",
        choices={"gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"},
        nargs="+",
        help="Optional ICESat-2 beams to use (space separated).",
    )
    parser.add_argument(
        "--dem",
        type=str,
        help=(
            "Optional DEM file used to slope-correct elevations when when matching points are "
            "far apart. Particularly useful when comparing to historical reference datasets."
        ),
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.0,
        help="Optional minimum horizontal separation (m) to consider a match.",
    )
    parser.add_argument(
        "--max_dist",
        type=float,
        default=20.0,
        help="Optional maximum horizontal separation (m) for matches (default 20 m).",
    )
    parser.add_argument(
        "--max_diff",
        type=float,
        default=200.0,
        help=(
            "Optional maximum absolute elevation difference (m) to keep between matched points. "
            "Pairs exceeding this threshold are discarded."
        ),
    )
    parser.add_argument(
        "--add_vars",
        nargs="+",
        help=(
            "Optional extra altimetry variables to include in outputs. "
            "Provide a space-separated list of variable names."
        ),
        default=[],
    )
    parser.add_argument(
        "--cryotempo_modes",
        default=None,
        choices={"lrm", "sin", "sar", "all"},
        nargs="+",
        help=(
            "Optional CryoTempo modes to include (lrm, sin, sar, all). "
            "For non-CryoTempo CS2 L2 data, supply multiple --altim_dir paths instead."
        ),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Worker process count for parallel file processing (default 10).",
    )
    parser.add_argument(  # Default is False, becomes True if provided
        "--compare_to_historical_reference",
        action="store_true",
        help=(
            "Use historical reference datasets (IceBridge, Pre-IceBridge, ICESat-1); "
            "auto-expands default max_dist to 10000 m."
        ),
    )
    parser.add_argument(
        "--compare_to_self",  # Default is False, becomes True if provided
        action="store_true",
        help="Compare reference data against itself (no altimetry input needed).",
    )
    parser.add_argument(
        "--nearest_only",  # Default is True, becomes False if provided
        action="store_false",
        help="Use all matches within search radius instead of only the nearest neighbor.",
    )
    # Plotting arguments #
    parser.add_argument(
        "--bins",
        default=100,
        type=int,
        help="Number of bins for elevation difference histograms (default 100).",
    )

    args = parser.parse_args()

    if not args.compare_to_self and args.altim_dir is None:
        parser.error("--altim_dir is required unless --compare_to_self is set")

    if args.area not in list_all_area_definition_names_only():
        parser.error(f"{args.area} not a valid cpom area name")

    if args.compare_to_historical_reference:
        if args.max_dist == 20.0:  # Change default if no value passed.
            args.max_dist = 1e4
    if args.start_date and args.end_date:
        if args.year or args.month:
            parser.error("Cannot provide both 'start_date'/'end_date' and 'year'/'month'")
        if len(args.start_date) != 8 or len(args.end_date) != 8:
            parser.error("Start and end dates must be in YYYYMMDD format")
    return args


def get_variable(nc: Dataset, nc_var_path: str) -> np.ndarray:
    """Retrieve a NetCDF variable, traversing nested groups.

    Args:
        nc (Dataset): The dataset object
        nc_var_path (str): The path to the variable within the file,
                        with groups separated by '/'.

    Raises:
        KeyError: If the variable or group is not found in the file.

    Returns:
        np.ndarray: Requested variable values.
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
    """Return default variable names inferred from filename patterns.

    Args:
        file: Data file path.

    Returns:
        dict: Mapping of logical names to dataset variable paths.
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


def get_date_from_filename(file: Path, log: logging.Logger) -> Optional[datetime]:
    """Extract a date from filename using known regex patterns.

    Supports multiple instrument conventions and logs an error when ambiguous
    or missing dates are encountered so the calling code can skip the file.

    Args:
        file: Data file path.
        log: Logger Object

    Returns:
        datetime | None: Parsed date if found, else None.
    """

    date_candidates = []
    for date_str in re.findall(r"(\d{8}T|\d{8}_|\d{14}_)", str(file)):
        try:
            # Check if date_str is a valid date
            # Check if it is a month
            if len(date_str) == 9 or len(date_str) == 15:
                dt = datetime.strptime(date_str[:8], "%Y%m%d")
                if not 1 <= dt.month <= 12:
                    continue
                if 1960 <= dt.year <= 2100:
                    date_candidates.append(dt)
                else:
                    continue
            else:
                continue
        except ValueError:
            continue
    if len(date_candidates) == 1:
        return date_candidates[0]
    if len(date_candidates) > 1:
        # If multiple dates are very close (within 1 day), pick the first
        if all(
            abs((d1 - d2).days) < 2
            for i, d1 in enumerate(date_candidates)
            for d2 in date_candidates[i + 1 :]
        ):
            return date_candidates[0]
        log.error(f"Multiple possible dates found in file {file}: {date_candidates}")
        return None

    # Check for match to ICESat1 dateformat /YYYY.MM.DD/
    dot_match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", str(file))
    if dot_match is not None:
        try:
            return datetime.strptime(".".join(dot_match[0]), "%Y.%m.%d")
        except ValueError:
            log.error(f"Invalid date {dot_match[0]} in file {file}")

    log.error(f"No valid date found in file {file}")
    return None


def get_date_bounds(
    args: argparse.Namespace, use_date_delta: bool = True
) -> tuple[datetime, datetime]:
    """Return start/end date bounds for filtering files.

    Args:
        args: Parsed CLI arguments.
        use_date_delta: Whether to apply date padding.

    Returns:
        tuple[datetime, datetime]: Inclusive start and end datetimes.
    """
    delta = timedelta(days=args.date_delta) if use_date_delta else timedelta(0)
    if args.start_date and args.end_date:
        return (
            datetime.strptime(args.start_date, "%Y%m%d") - delta,
            datetime.strptime(args.end_date, "%Y%m%d") + delta,
        )
    year = str(args.year)
    month = str(args.month).zfill(2)
    return (
        datetime.strptime(f"{year}{int(month):02d}01", "%Y%m%d") - delta,
        datetime.strptime(
            f"{year}{int(month):02d}{calendar.monthrange(int(year), int(month))[1]:02d}",
            "%Y%m%d",
        )
        + delta,
    )


def get_files_in_dir(
    directory: Path,
    min_day: datetime,
    max_day: datetime,
    log: logging.Logger,
    filetype: str = "nc",
) -> dict[Path, str]:
    """
    Find files of given type within date bounds; returns path -> YYYY-MM-DD string.
    This function searches for files in the given directory and its subdirectories,
    filtering them based on the specified year, month, or optional date range.

    Args:
        directory: Root directory to search.
        min_day: Start date for filtering files.
        max_day: End date for filtering files.
        log: Logger Object
        filetype: File extension key: nc, h5, or H5.

    Returns:
        dict[Path, str]: Mapping of file path to YYYY-MM-DD string.
    """
    extensions = {"nc": "*.nc", "h5": "*.h5", "H5": "*.H5"}
    all_files: list[Path] = []
    all_files.extend(Path(directory).rglob(extensions[filetype]))

    valid_files = {}
    for file in all_files:
        if "." in file.stem:
            continue
        try:
            date_obj = get_date_from_filename(file, log)
            if date_obj is not None and min_day <= date_obj <= max_day:
                valid_files[file] = date_obj.strftime("%Y-%m-%d")
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
        self.dtypes = [
            ("x", "float64"),
            ("y", "float64"),
            ("h", "float64"),
            ("date", "datetime64[D]"),
        ]

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
        """Extract and filter altimetry data. Returns structured array or None.
        Args:
            filename: Altimetry NetCDF file.
        Returns:
            np.ndarray | None: Structured array of points or None if empty/unsupported.
                            Includes :
                                x, y, h , date , (+ params.add_vars)
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

            # Structured NumPy array containing x, y, elevation,date and any additional variables
            arr = np.empty(
                len(x),
                dtype=list(self.dtypes)
                + [(var, data.dtype) for var, data in additional_data.items()],
            )
            arr["x"] = x
            arr["y"] = y
            arr["h"] = elev
            arr["date"] = np.full(len(x), np.datetime64(self.valid_files[filename], "D"))
            for var, data in additional_data.items():
                arr[var] = data

            if len(arr) != len(x) or len(arr) != len(y) or len(arr) != len(elev):
                self.log.error(f"Mismatch in variable lengths for {filename} with date array")
                raise ValueError(f"Mismatch in variable lengths for {filename} with date array")

            return arr
        except OSError as err:
            self.log.error(f"Error loading altimetry data file {filename}, failed with : {err}")
            return None

    def get_is2_data_array(self, filename: Path) -> Optional[np.ndarray]:
        """Extract and filter ICESat-2 data. Returns structured array or None.
        Args:
            filename: ICESat-2 HDF5 file.

        Returns:
            np.ndarray | None: Structured array of points or None if no valid data.
                                Includes : x , y , h , date
        """
        points = []
        try:
            with h5py.File(filename, "r") as nc:
                for beam in self.args.beams:
                    config = {
                        key: path.format(beam=beam)
                        for key, path in get_default_variables(filename).items()
                    }

                    # Check if beam exists in file
                    try:
                        hemisphere = get_variable(nc, config["lat"])[0]
                    except (KeyError, IndexError) as e:
                        self.log.debug(f"Beam {beam} not found in {filename}: {e}")
                        continue
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

                    # Structured NumPy array containing x, y, elevation , date
                    arr = np.empty(len(x), dtype=self.dtypes)
                    arr["x"] = x
                    arr["y"] = y
                    arr["h"] = elevation
                    arr["date"] = np.full(len(x), np.datetime64(self.valid_files[filename], "D"))

                    points.append(arr)

        except OSError as err:
            self.log.error(f"Error loading atl06 data file {filename} failed with : {err}")
            return np.array(
                [],
                dtype=self.dtypes,
            )

        return (
            np.concatenate(points) if points else np.array([], dtype=self.dtypes)
        )  # Concat point arrays into one, or return an empty structured array if none exist

    def get_icebridge_data_array(self, filename: Path) -> Optional[np.ndarray]:
        """Extract and filter IceBridge / Pre-IceBridge data. Returns array or None.

        Args:
            filename: IceBridge NetCDF file.

        Returns:
            np.ndarray | None: Structured array of points or None if empty.
                            Includes: x , y , h , date
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

            arr = np.empty(len(x), dtype=self.dtypes)
            arr["x"] = x
            arr["y"] = y
            arr["h"] = elevation
            arr["date"] = np.full(len(x), np.datetime64(self.valid_files[filename], "D"))

            return arr

    def get_is1_data_array(self, filename: Path) -> Optional[np.ndarray]:  # pylint: disable=R0914
        """Extract and filter IS1 data from a fill. Returns array or None.
        Follows method from Smith et al., Science(2020)
        Args:
            filename: ICESat-1 HDF5 file.

        Returns:
            np.ndarray | None: Structured array of points or None if empty/invalid.
                                Includes x, y , h , date
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

            arr = np.empty(len(x), dtype=self.dtypes)
            arr["x"] = x
            arr["y"] = y
            arr["h"] = elevation
            arr["date"] = np.full(len(x), np.datetime64(self.valid_files[filename], "D"))

            return arr

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


def get_elev_differences(
    args: argparse.Namespace,
    laser_points: np.ndarray,
    altimeter_points: np.ndarray,
    prefix: str,
    log: logging.Logger,
) -> dict:
    """
    Calculate elevation differences between reference (laser_points) and altimetry
    points.
    Filters to a search radius (min/max dist), date tolerance,
    optional self-match exclusion, maximum dh.
    Optionally keep only the nearest neighbor.

    Args:
        args: Command Line Arguments
        laser_points: Reference array containing x, y, h, date.
        altimetry_points: Altimetry array containing x, y, h, date (and extras).
        prefix: Prefix for altimetry fields in outputs.
        log: logging.Logger

    Returns:
        dict: Dh values and associated metadata lists.
    """

    is2_tree = KDTree(np.c_[laser_points["x"], laser_points["y"]])
    log.info("Constructed KDTree")

    add_vars = [var for var in altimeter_points.dtype.names if var not in {"x", "y", "h", "date"}]
    indices_list = is2_tree.query_ball_point(
        np.column_stack((altimeter_points["x"], altimeter_points["y"])), r=args.max_dist
    )
    laser_idx = np.concatenate(indices_list)
    log.info("Found %d potential matches within %d m", len(laser_idx), args.max_dist)

    alt_idx = np.repeat(
        np.arange(len(indices_list)),
        [len(i) for i in indices_list],
    )

    ref = laser_points[laser_idx]
    alt = altimeter_points[alt_idx]

    # Get the distances
    dx = alt["x"] - ref["x"]
    dy = alt["y"] - ref["y"]
    dist = np.sqrt(dx * dx + dy * dy)

    # Filter Filters
    log.info("Applying filters to matches")
    mask = dist >= args.min_dist

    if args.date_delta > 0:
        mask &= np.abs(ref["date"] - alt["date"]) <= np.timedelta64(args.date_delta, "D")

    if args.compare_to_self:
        mask &= ~((ref["x"] == alt["x"]) & (ref["y"] == alt["y"]) & (ref["h"] == alt["h"]))

    ref, alt, dist, alt_idx = ref[mask], alt[mask], dist[mask], alt_idx[mask]
    log.info(" %d matches remain after filtering", len(ref))

    if args.nearest_only:
        # Sort by (alt_idx, distance)
        order = np.lexsort((dist, alt_idx))
        ref, alt, dist, alt_idx = ref[order], alt[order], dist[order], alt_idx[order]

        # Keep first occurrence per altimeter point
        _, unique_pos = np.unique(alt_idx, return_index=True)
        ref, alt, dist = ref[unique_pos], alt[unique_pos], dist[unique_pos]
        log.info(" %d matches remain after keeping nearest neighbor", len(ref))

    # Filter to max dh
    dh = alt["h"] - ref["h"]
    keep = np.abs(dh) <= args.max_diff
    ref, alt, dist, dh = ref[keep], alt[keep], dist[keep], dh[keep]
    log.info(" %d matches remain after applying max dh of %f m", len(ref), args.max_diff)

    results = {
        "dh": dh.tolist(),
        "sep_dist": dist.tolist(),
        f"{prefix}x": alt["x"].tolist(),
        f"{prefix}y": alt["y"].tolist(),
        f"{prefix}h": alt["h"].tolist(),
        f"{prefix}date": alt["date"].tolist(),
        "reference_x": ref["x"].tolist(),
        "reference_y": ref["y"].tolist(),
        "reference_h": ref["h"].tolist(),
        "reference_date": ref["date"].tolist(),
    }

    for var in add_vars:
        results[var] = alt[var].tolist()

    return results


def correct_elevation_using_slope(
    data: dict, args: argparse.Namespace, log: logging.Logger, prefix: str
) -> dict:
    """Slope-correct reference elevations by shifting to altimetry footprint using a DEM.

    Adjusts both dh and reference_h in-place to account for horizontal separation
    using the specified DEM.

    Args :
        data (np.ndarray): Output of "get_elev_differences"
        args (argparse.Namespace) : Command Line Parameters
        log (Logger): Logger Object
        prefix : Reference data prefix

    Returns:
        dict : Slope corrected elevation difference dictionary
    """
    log.info("Performing slope correction")
    dem = Dem(args.dem)

    dh_dem_elev = (dem.interp_dem(data[f"{prefix}x"], data[f"{prefix}y"])) - (
        dem.interp_dem(data["reference_x"], data["reference_y"])
    )
    data["reference_h"] = data["reference_h"] + dh_dem_elev
    data["dh"] = data["dh"] - dh_dem_elev

    data = {key: np.array(data[key]) for key in data}
    valid_mask = (~np.isnan(data["dh"])) & (np.abs(data["dh"]) <= args.max_diff)

    return {key: val[valid_mask] for key, val in data.items()}


######## Plotting and statistics ########
def elev_dh_histograms(dh: np.ndarray, outdir: str | Path, bins: int = 100) -> None:
    """Plot linear/log histograms of dh and save to file."""
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


def elevation_dh_cumulative_dist(dh: np.ndarray, out_dir: str | Path) -> None:
    """Plot cumulative distribution of dh and save to file."""
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


def compute_elevation_stats(
    output: dict,
    prefix: str,
    output_file: str | Path = "elevation_stats.txt",
) -> None:
    """Compute summary statistics for dh and write a CSV."""
    num_points = len(output["dh"])

    # Compute R² value for height correlation
    p2p_r2_hgt = np.corrcoef(output[f"{prefix}h"], output["reference_h"])[0, 1] ** 2
    mean_sepdist, stddev_sepdist = np.mean(output["sep_dist"]), np.std(
        output["sep_dist"]
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

    columns = [
        "metric",
        "num_points",
        "mean_dh",
        "median_dh",
        "std_dh",
        "MAD_dh",
        "RMS_dh",
        "2_sigma_mean_dh",
        "2_sigma_std_dh",
        "r2_hgt",
        "mean_dist_m",
        "std_dist_m",
    ]

    rows = []
    for label, data in (("signed", output["dh"]), ("absolute", np.abs(output["dh"]))):
        stats = compute_stats(data)
        rows.append(
            [
                label,
                num_points,
                *stats,
                p2p_r2_hgt,
                mean_sepdist,
                stddev_sepdist,
            ]
        )

    pd.DataFrame(rows, columns=columns).to_csv(output_file, index=False, float_format="%.6f")


def run_parallel(
    func: Callable[[Path], Optional[np.ndarray]],
    items: Iterable[Path],
    max_workers: int,
) -> np.ndarray:
    """
    Run a file-processing function in parallel and return unique x,y,h,date records.
    Deduplicates on (x,y,h)
    """
    items = list(items)
    if not items:
        return np.array([], dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")])

    chunksize = max(1, len(items) // (max_workers * 4))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        points = list(executor.map(func, items, chunksize=chunksize))

    populated_points = np.concatenate([result for result in points if result is not None])
    unique_indices = np.unique(populated_points[["x", "y", "h"]], return_index=True)[1]

    return populated_points[unique_indices]


if __name__ == "__main__":
    params = parse_args()
    AREA_OBJ = Area(params.area)

    # Get output directory location and reference data input location
    base_outdir = Path(params.outdir)
    base_reference = Path(params.reference_dir)
    if params.start_date and params.end_date:
        start = datetime.strptime(params.start_date, "%Y%m%d")
        end = datetime.strptime(params.end_date, "%Y%m%d")
        output_dir = base_outdir / f"{start:%Y%m%d}_{end:%Y%m%d}"

        reference_dir = base_reference
    else:
        DATE_YEAR = str(params.year)
        DATE_MONTH = str(params.month).zfill(2)
        output_dir = base_outdir / f"{DATE_YEAR}{DATE_MONTH}"

        if params.date_delta > 0:
            reference_dir = base_reference
        else:
            reference_dir = (
                base_reference / DATE_YEAR / DATE_MONTH
                if (base_reference / DATE_YEAR / DATE_MONTH).is_dir()
                else base_reference
            )

    # Set up Logging
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
    logger.info("Output directory set to %s", output_dir)
    logger.info("Reference input directory set to %s", reference_dir)

    first_day, last_day = get_date_bounds(params, True)
    logger.info("Getting reference files between %s and %s", first_day, last_day)
    reference_file_date = get_files_in_dir(
        reference_dir, first_day, last_day, logger, params.reference_filetype
    )  # is2/ icebridge / is1

    logger.info("Loaded %d reference data files", len(reference_file_date))

    processor = ProcessData(params, AREA_OBJ, reference_file_date, logger)
    reference_points = run_parallel(
        processor.process_reference_file,
        list(reference_file_date.keys()),
        params.max_workers,
    )
    logger.info("Loaded reference data points, len : %d", len(reference_points))

    # Load altimetry data #
    if params.compare_to_self:
        logger.info("Performing reference self-comparison")
        PREFIX = "neighbour_"
        first_day, last_day = get_date_bounds(params, False)
        logger.info("Filtering reference points between %s and %s", first_day, last_day)
        date_mask = (reference_points["date"] >= np.datetime64(first_day.date())) & (
            reference_points["date"] <= np.datetime64(last_day.date())
        )
        altimetry_points = reference_points[date_mask]  # Compare is2 to itself
        logger.info("Loaded altimetry points, len %d", len(altimetry_points))
    else:
        logger.info("Comparing reference to altimetry")
        PREFIX = ""
        altimetry_files = {}
        first_day, last_day = get_date_bounds(params, False)
        logger.info("Getting altimetry files between %s and %s", first_day, last_day)
        for basepath in params.altim_dir:  # Can have multiple altimetry directories e.g. LRM/SIN
            basepath = Path(basepath)
            alt_dir = basepath if params.start_date else basepath / f"{DATE_YEAR}/{DATE_MONTH}"
            altimetry_files.update(
                get_files_in_dir(
                    alt_dir if alt_dir.is_dir() else basepath,
                    first_day,
                    last_day,
                    logger,
                    params.altimetry_filetype,
                )
            )

        logger.info("Loaded %d altimetry data files", len(altimetry_files))
        processor.valid_files = altimetry_files  # Set valid files for processor

        altimetry_points = run_parallel(
            processor.get_altimetry_data_array,
            altimetry_files,
            params.max_workers,
        )
        logger.info("Loaded altimetry data points, len : %d", len(altimetry_points))

    outfile = output_dir / f"p2p_diffs_{params.area}"

    elev_differences = get_elev_differences(
        params, reference_points, altimetry_points, PREFIX, logger
    )
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
        f"{PREFIX}lons": alt_lons,
        f"{PREFIX}lats": alt_lats,
        "reference_lons": reference_lons,
        "reference_lats": reference_lats,
        **elev_differences,
    }

    np.savez(f"{outfile}.npz", **save_data)
    compute_elevation_stats(save_data, prefix=PREFIX, output_file=f"{outfile}_elevation_stats.csv")
    Polarplot(params.area).plot_points(
        {
            "name": "difference_in_height_(dh)",
            "lats": save_data[f"{PREFIX}lats"],
            "lons": save_data[f"{PREFIX}lons"],
            "vals": save_data["dh"],
            "cmap": "PuOr",
        },
        output_dir=str(output_dir),
    )
    elev_dh_histograms(save_data["dh"], f"{outfile}_dh_histogram.png", bins=params.bins)
    elevation_dh_cumulative_dist(save_data["dh"], f"{outfile}_dh_cumulative_distribution.png")

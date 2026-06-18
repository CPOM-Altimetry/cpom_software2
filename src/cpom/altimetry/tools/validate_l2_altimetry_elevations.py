"""cpom.altimetry.tools.validate_l2_altimetry_elevations.py

This tool compares a selected month of radar altimetry mission elevation data against laser
altimetry reference elevations.
It also allows comparing a reference mission to itself.

### **Supported Missions**

- **Altimetry Missions**: CS2 (Native L2 and CryoTempo), CRISTAL (CLEV2ER landice L2 POCA),
                        S3A, S3B, ENVISAT, ERS1, ERS2
- **Reference Missions**: ICESat-2 (ATL06), IceBridge (ILATM2, ILUTP2),
                        Pre-IceBridge (BRMCR2, BLATM2), ICESat1 (GLAH12)

**Please note you may need to add your dataset to 'get_default_variables' if you are testing
with a none standard altimetry data product.**

### **Notes to user**
When comparing to icebridge : The default radius increases from 20m to 10000m,
to capture enough points for validation. For the best performance alter the radius and
max_diff argument for your usecase, or provide a DEM to correct reference locations
to align with altimetry measurements.

When running for Cryotempo, the Cryotempo_Modes argument must be set.

When running for CRISTAL (CLEV2ER landice L2 POCA files, e.g. ``CRA_IR_GR_*MSSL_LIG*.NC``)
the data are stored in per-band NetCDF groups (``data/ku/...`` and ``data/ka/...``). Use
``--band`` to select ``ku`` or ``ka`` (run once per band). The POCA elevation/lat/lon are
read from the selected band's ``poca`` group automatically. ``--add_vars`` names without a
``/`` are resolved within that band's ``poca`` group, so to also extract backscatter and
coherence (the POCA uncertainty-LUT covariates) pass ``--add_vars sig0 coherence``; use
``{band}`` in a full path for variables elsewhere in the tree (e.g.
``--add_vars data/{band}/penetration_depth``). This produces the ``dh``/``lats``/``lons``
(+ covariate) ``.npz`` consumed by ``clev2er.utils.uncertainty`` LUT pre-processing.

### **Command Line Options**
**Required**
- `reference_dir`: Path to reference dataset.
- `altim_dir`: Path to altimetry dataset.
- `year`: Year of data.
- `month`: Month of data.
- `area`: CPOM area.
- `outdir`: Output directory.

**Optional**
- `dem`: Provide a DEM to correct location when comparing to IceBridge with a large radius,
        used in 'correct_elevation_using_slope'
- `radius`: Search radius (in meters) for validation points.
- `max_diff`: Maximum elevation difference between matched points.
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

import argparse
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import median_absolute_deviation, sigma_clipped_stats
from netCDF4 import Dataset, Variable  # pylint: disable=E0611
from pyproj import CRS, Transformer
from scipy.spatial import KDTree

from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area, list_all_area_definition_names_only
from cpom.dems.dems import Dem


def setup_logging(
    log_name="",
    log_file_info="file.log",
    log_file_error="file.err",
    console_level=logging.INFO,
):
    """Set up logging to two files (info + error) and to the console (stdout).

    A console (stdout) handler is added so the user sees progress in the
    terminal as the tool runs, in addition to the persistent log files.

    Args:
        log_name (str): Logger name ("" selects the root logger).
        log_file_info (str | Path): Path of the INFO-level log file.
        log_file_error (str | Path): Path of the ERROR-level log file.
        console_level (int): Logging level for the console handler.

    Returns:
        logging.Logger: The configured logger.
    """
    log = logging.getLogger(log_name)
    # let handlers decide what to emit; logger must pass the lowest level through
    log.setLevel(min(logging.INFO, console_level))
    log.handlers.clear()  # idempotent: avoid duplicate handlers if called again

    file_format = logging.Formatter(
        "%(levelname)s : %(asctime)s : %(message)s", datefmt="%d/%m/%Y %H:%M:%S"
    )
    for log_file, level in [(log_file_info, logging.INFO), (log_file_error, logging.ERROR)]:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(file_format)
        file_handler.setLevel(level)
        log.addHandler(file_handler)

    # Console handler: concise format (time + message) for readable progress
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    console.setLevel(console_level)
    log.addHandler(console)

    # Quieten third-party libraries that would otherwise flood the console at INFO
    for noisy in ("matplotlib", "PIL", "numexpr", "fiona", "rasterio", "cartopy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return log


def parse_args() -> argparse.Namespace:
    """Argument Parser to load command line arguments.
    Returns:
        argparse.ArgumentParser: Parsed input arguments
    """
    parser = argparse.ArgumentParser(description="Retrieve command line arguments")

    parser.add_argument(
        "--reference_dir",
        help="The is2_atl06, icebridge or pre-icebridge data directory \
            If a subdirectory matching 'YEAR/MONTH' exists it will be picked up automatically.",
        required=True,
    )
    parser.add_argument(
        "--altim_dir",
        nargs="+",
        help="Altimetry data directory('s) \
        Supports multiple directories by providing a space seperated list of path1 path2 \
        If a subdirectory matching 'YEAR/MONTH' exists it will be picked up automatically",
    )
    parser.add_argument("--year", type=int, help="Year (YYYY)", required=True)
    parser.add_argument(
        "--month", type=int, choices=range(1, 13), help="Month (1-12)", required=True
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
        "--band",
        choices={"ku", "ka"},
        default="ku",
        help="[optional, default=ku] altimeter band for grouped products (CRISTAL L2). "
        "Selects the data/<band>/ NetCDF group. Run once per band.",
    )
    parser.add_argument("--dem", help="[optional] DEM, used in 'correct_elevation_using_slope'")
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
    parser.add_argument(
        "--log_level",
        choices={"debug", "info", "warning"},
        default="info",
        help="[optional, default=info] console logging verbosity",
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
    return args


def get_variable(nc: Dataset | h5py.File, nc_var_path: str) -> np.ndarray:
    """Retrieve variable from NetCDF file, handling groups if necessary.

    Args:
        nc (Dataset | h5py.File): The dataset object
        nc_var_path (str): The path to the variable within the file,
                        with groups separated by '/'.

    Raises:
        KeyError: If the variable or group is not found in the file.

    Returns:
        np.array: The retrieved variable as an array.
    """
    try:
        parts = nc_var_path.split("/")
        var: Dataset | Variable = nc
        for part in parts:
            var = var[part]
            if var is None:
                raise IndexError(f"NetCDF parameter '{nc_var_path}' not found.")
        return var[:]  # type: ignore[index]
    except IndexError as err:
        raise IndexError(f"NetCDF parameter or group {err} not found") from err


def variable_exists(nc: Dataset | h5py.File, nc_var_path: str) -> bool:
    """Return True if a (possibly grouped) variable path exists in the file.

    Unlike ``var in nc.variables`` this walks NetCDF groups, so grouped paths
    such as ``data/ku/poca/sig0`` are handled. A bare name (no ``/``) keeps the
    original membership-test semantics against the top-level ``variables``.

    Args:
        nc (Dataset | h5py.File): The open dataset object.
        nc_var_path (str): Variable path, groups separated by '/'.

    Returns:
        bool: True if the variable exists, otherwise False.
    """
    parts = nc_var_path.split("/")
    node = nc
    for part in parts[:-1]:
        try:
            node = node.groups[part]
        except (KeyError, AttributeError):
            return False
    return parts[-1] in getattr(node, "variables", {})


def mask_to_nan(array: np.ndarray) -> np.ndarray:
    """Convert a masked array to a plain float array with NaN in masked cells.

    Scaled-integer NetCDF variables (used throughout the CRISTAL L2 products)
    are returned by netCDF4 as masked float arrays; downstream code here relies
    on ``np.isnan`` to detect missing values, so fill the mask with NaN. Plain
    (unmasked) arrays are returned unchanged.

    Args:
        array (np.ndarray): Array as returned by ``get_variable``.

    Returns:
        np.ndarray: NaN-filled float array, or the input unchanged if not masked.
    """
    if np.ma.isMaskedArray(array):
        return np.ma.filled(array.astype("float64"), np.nan)
    return array


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
        "CRA_IR_GR": {  # CRISTAL L2 land ice POCA (CLEV2ER landice chain), per-band groups
            "lat_nadir": "data/{band}/lat",
            "lon_nadir": "data/{band}/lon",
            "lat": "data/{band}/poca/lat_surf",
            "lon": "data/{band}/poca/lon_surf",
            "elev": "data/{band}/poca/land_ice_elevation",
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
        },
        # iceSAT1
        "GLAH12": {
            "lat": "Data_40HZ/Geolocation/d_lat",
            "lon": "Data_40HZ/Geolocation/d_lon",
            "elev": "Data_40HZ/Elevation_Surfaces/d_elev",
        },
        # Icebridge
        "ILATM2": {"lat": "lat", "lon": "lon", "elev": "hgt"},
        "ILUTP2": {"lat": "lat", "lon": "lon", "elev": "hgt"},
        "BLATM2": {"lat": "lat", "lon": "lon", "elev": "height"},
        "BRMCR2": {"lat": "lat", "lon": "lon", "elev": "elevation"},
        "ENVISAT_functional_TDS": {"lat": "latitude", "lon": "longitude", "elev": "elevation"},
    }

    for key, value in variable_map.items():  # Check for a matching pattern
        if key in basename:
            return value
    return {}


def get_files_in_dir(
    directory: Path,
    year: str,
    month: str,
    filetype: str,
) -> list[Path]:
    """
    Find .nc or.h5 files in the specified directory.

    Args:
        directory (Path): The directory to search for files.
        year (str): Year to process
        month (str): Month to process (0 filled)
        filetype (str) : nc or h5
    Returns:
        List[Path]: A list of found .nc or .NC files with their full paths.
    """
    extensions = {"nc": "*.nc", "h5": "*.h5"}
    all_files: list[Path] = []
    pattern = extensions[filetype]
    all_files.extend(Path(directory).rglob(pattern))

    # If no files found, try upper case suffix
    if not all_files:
        all_files.extend(Path(directory).rglob(pattern.upper()))

    # Matches: YYYYMMDDT | YYYYMMDD_ | YYMMDD_ |YYYYMMDDHHMMSS
    file_basename_regex = re.compile(
        rf"(?:{year}{month}\d{{2}}T\d*|{year}{month}\d{{2}}_\d*|{year[-2:]}{month}\d{{2}}_|{year}{month}\d{{8}})"  # noqa # pylint: disable=C0301
    )

    valid_files = [  # Exclude filenames containing multiple dots before the extension
        f
        for f in all_files
        if any(
            file_basename_regex.search(str(part)) for part in f.parts
        )  # Check if regex matches any part of the path
        and "." not in f.stem  # Exclude filenames containing multiple dots before the extension
    ]

    if valid_files:
        return valid_files

    # Match to YYYY.MM.DD in the input directory if no files found (Used for ISAT1 data load)
    valid_files = [f for f in all_files if re.compile(rf"{year}\.{month}\.\d{{2}}").search(str(f))]
    if valid_files:
        return valid_files

    return []


def collect_with_progress(results_iter, total: int, log: logging.Logger, label: str) -> list:
    """Drain a (lazy) results iterator into a list, logging progress as it goes.

    The parallel file loaders can run for a long time with no output; this
    reports progress roughly every 5% so the user can see the run advancing.

    Args:
        results_iter: Iterator yielding per-file results (e.g. from executor.map).
        total (int): Total number of items expected (used for the percentage).
        log (logging.Logger): Logger to report progress to.
        label (str): Short stage label, e.g. "reference" or "altimetry".

    Returns:
        list: All collected results, in iteration order.
    """
    every = max(1, total // 20)  # ~20 progress lines over the whole stage
    collected = []
    for count, result in enumerate(results_iter, start=1):
        collected.append(result)
        if count % every == 0 or count == total:
            log.info(
                "  %s files processed: %d/%d (%.0f%%)",
                label,
                count,
                total,
                100.0 * count / total,
            )
    return collected


def concat_altimetry_results(arrays: list) -> np.ndarray:
    """Concatenate per-file altimetry arrays that may carry different add_vars.

    Files of different mode/band can legitimately contain different optional
    ``--add_vars`` (e.g. HR Ku has ``coherence`` but LR Ku and Ka do not). A plain
    ``np.concatenate`` would fail on the mismatched dtypes, so this takes the union
    of the extra columns and fills a file's absent columns with NaN. The base
    ``x``/``y``/``h`` fields are always present. Extra columns are stored as float64
    so the NaN fill is representable.

    Args:
        arrays: Non-empty list of structured arrays, each with at least x, y, h.

    Returns:
        np.ndarray: One structured array with x, y, h + the union of extra fields.
    """
    base = ("x", "y", "h")
    extra_names: list[str] = []
    for arr in arrays:
        for name in arr.dtype.names or ():
            if name not in base and name not in extra_names:
                extra_names.append(name)
    out_dtype = [(name, "float64") for name in (*base, *extra_names)]
    total = sum(int(arr.size) for arr in arrays)
    out = np.empty(total, dtype=out_dtype)
    pos = 0
    for arr in arrays:
        n = int(arr.size)
        for name in base:
            out[name][pos : pos + n] = arr[name]
        for name in extra_names:
            out[name][pos : pos + n] = arr[name] if name in arr.dtype.names else np.nan
        pos += n
    return out


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

    def __init__(self, args, area, log):
        self.args = args
        self.area = area
        self.log = log

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
        self, nc: Dataset, lat: np.ndarray, lon: np.ndarray, config: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Populate empty poca lat/lon with nadir lat/lon
        Args:
            nc (Dataset): Open NetCDF dataset
            lat (np.array): Array of latitudes
            lon (np.array): Array of longitudes
            config (dict): Dictionary of variable names
        Returns:
            (np.array, np.arrray): Filled lat/lon arrays
        """
        bad_indices = np.isnan(lon)  # np.flatnonzero(lon.mask)  # Find empty longitude values
        if bad_indices.size > 0:
            lat_nadir = mask_to_nan(get_variable(nc, config["lat_nadir"]))
            lon_nadir = mask_to_nan(get_variable(nc, config["lon_nadir"]))
            lat[bad_indices] = lat_nadir[bad_indices]
            lon[bad_indices] = np.mod(lon_nadir[bad_indices], 360)
        return lat, lon

    @staticmethod
    def _resolve_altim_var(var: str, band: str, is_cristal: bool) -> str:
        """Resolve an ``--add_vars`` entry to a full NetCDF variable path.

        A ``{band}`` placeholder is substituted with the selected band. For
        CRISTAL L2 files a bare name (no '/') is resolved within the band's
        ``poca`` group, so ``sig0`` -> ``data/<band>/poca/sig0``.

        Args:
            var (str): The user-supplied ``--add_vars`` entry.
            band (str): Selected band ('ku' or 'ka').
            is_cristal (bool): True if the file is a CRISTAL grouped product.

        Returns:
            str: The resolved variable path (may still not exist in the file).
        """
        if "{band}" in var:
            return var.format(band=band)
        if is_cristal and "/" not in var:
            return f"data/{band}/poca/{var}"
        return var

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

                # Per-band grouped products (CRISTAL L2) carry a {band} placeholder
                # in their variable paths; resolve it from --band.
                band = getattr(self.args, "band", "ku")
                is_cristal = any("{band}" in path for path in config.values())
                if is_cristal:
                    config = {
                        key: (path.format(band=band) if "{band}" in path else path)
                        for key, path in config.items()
                    }

                lats, lons, elev = (
                    mask_to_nan(get_variable(nc, config["lat"])),
                    np.mod(mask_to_nan(get_variable(nc, config["lon"])), 360),
                    mask_to_nan(get_variable(nc, config["elev"])),
                )

                try:
                    add_vars = self.args.add_vars or []
                    additional_data = {}
                    missing = []
                    for var in add_vars:
                        resolved = self._resolve_altim_var(var, band, is_cristal)
                        if variable_exists(nc, resolved):
                            additional_data[resolved.rsplit("/", 1)[-1]] = mask_to_nan(
                                get_variable(nc, resolved)
                            )
                        else:
                            missing.append(var)
                    if missing:
                        self.log.info("Variable(s) %s missing from netcdf", set(missing))
                except Exception as err:
                    raise ValueError("Failed to load additional data") from err

                if self.args.cryotempo_modes:
                    mask = self.get_cryotempo_filters(nc, self.args)
                    if mask is not None:
                        lats, lons, elev = lats[mask], lons[mask], elev[mask]
                        for k in additional_data:
                            additional_data[k] = additional_data[k][mask]
                    else:
                        return None

                if "lat_nadir" in config and "lon_nadir" in config:
                    lats, lons = self.fill_empty_latlon_with_nadir(nc, lats, lons, config)

                lats, lons, _, _ = self.area.inside_latlon_bounds(lats, lons)
                x, y = self.area.latlon_to_xy(lats, lons)

                idx, _ = self.area.inside_mask(x, y)
                if not idx.size:
                    return None

                try:
                    x, y, elev = x[idx], y[idx], elev[idx]
                    for k in additional_data:
                        additional_data[k] = additional_data[k][idx]

                except Exception as err:
                    raise ValueError("Mismatch in variable lengths") from err

            # Structured NumPy array containing x, y, elevation, and any additional variables
            return np.array(
                list(zip(x, y, elev, *additional_data.values())),
                dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")]
                + [(var, str(data.dtype)) for var, data in additional_data.items()],
            )
        except OSError as err:
            self.log.error(f"Error loading altimetry data file {filename}, failed with : {err}")
            return None

    def get_is2_data_array(self, filename: Path) -> np.ndarray:
        """Extract and filter data from an is2 data file.
        Args:
            filename (Path): IS2 data filename (.h5)
        Raises:
            ValueError: If variable lengths do not match
        Returns:
            np.ndarray: Structured nd.array containing x,y,h fields
        """
        try:
            with h5py.File(filename, "r") as nc:
                points = []
                for beam in self.args.beams:
                    config = {
                        key: path.format(beam=beam)
                        for key, path in get_default_variables(filename).items()
                    }
                    # Filter out files in wrong hemisphere. No files cross hemispheres
                    try:
                        hemisphere = get_variable(nc, config["lat"])[0]
                        if (hemisphere < 0.0 and self.area.hemisphere == "north") or (
                            hemisphere > 0.0 and self.area.hemisphere == "south"
                        ):
                            break

                        elevation = get_variable(nc, config["elev"])
                        ok = np.flatnonzero(
                            (
                                get_variable(nc, f"{beam}/land_ice_segments/atl06_quality_summary")
                                == 0
                            )
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
                            raise ValueError("Mismatch in variable lengths for x,y,h")
                        # Structured NumPy array containing x, y, elevation
                        points.append(
                            np.array(
                                list(zip(x, y, elevation)),
                                dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")],
                            )
                        )
                    except KeyError:
                        self.log.error(f"Missing data in {filename} for beam {beam}")
                        continue
        except OSError as err:
            self.log.error(f"Error loading atl06 data file {filename} failed with : {err}")
            return np.array([], dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")])
        return (
            np.concatenate(points)
            if points
            else np.array([], dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")])
        )  # Concat point arrays into one, or return an empty structured array if none exist

    def get_icebridge_data_array(self, filename: Path) -> np.ndarray:
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
            x, y = self.area.latlon_to_xy(lats, lons)
            elevation = get_variable(nc, config["elev"])[bounded_indices]

            if not len(x) == len(y) == len(elevation):
                raise ValueError("Mismatch in variable lengths for x,y,h")

            # Structured NumPy array containing x, y, elevation
            return np.array(
                list(zip(x, y, elevation)),
                dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")],
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
                raise ValueError("Mismatch in variable lengths for x,y,h")

            return np.array(
                list(zip(x, y, elevation)),
                dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")],
            )

    def process_reference_file(self, file_path: Path):
        """Process reference files,
        determines which function to call based on file type.
        nc : Icebridge / pre-icebridge
        h5 : IS2
        H5 : IS1
        """
        if file_path.suffix == ".nc":
            return self.get_icebridge_data_array(file_path)
        if file_path.suffix == ".h5":
            return self.get_is2_data_array(file_path)
        if file_path.suffix == ".H5":
            return self.get_is1_data_array(file_path)
        self.log.error("Unsupported file format: %s", file_path)
        return None


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
    add_vars = [var for var in (altimeter_points.dtype.names or ()) if var not in {"x", "y", "h"}]
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
    }

    for altimeter_point in altimeter_points:
        if nearest_only:
            dist, idx = is2_tree.query((altimeter_point["x"], altimeter_point["y"]))
            indices = [idx] if dist <= args.radius else []
        else:
            indices = is2_tree.query_ball_point(  # Find reference points within search_radius
                (altimeter_point["x"], altimeter_point["y"]), args.radius
            )

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
    dem = Dem(args.dem)
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
    DATE_YEAR = str(params.year)
    DATE_MONTH = str(params.month).zfill(2)
    month_outdir = Path(params.outdir) / f"{DATE_YEAR}/{DATE_MONTH}"

    try:
        month_outdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"Failed to create directory {month_outdir}: {e}")

    with open(month_outdir / "command_line_args.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")

    logger = setup_logging(
        log_file_info=month_outdir / f"{DATE_YEAR}{DATE_MONTH}.info.log",
        log_file_error=month_outdir / f"{DATE_YEAR}{DATE_MONTH}.errors.log",
        console_level=getattr(logging, params.log_level.upper()),
    )
    t_start = time.perf_counter()
    logger.info("=" * 72)
    logger.info("validate_l2_altimetry_elevations : %s %s-%s", params.area, DATE_YEAR, DATE_MONTH)
    logger.info("  reference_dir : %s", params.reference_dir)
    if params.compare_to_self:
        logger.info("  mode          : reference self-comparison")
    else:
        logger.info("  altim_dir     : %s", ", ".join(params.altim_dir))
        logger.info("  band          : %s", getattr(params, "band", "ku"))
    if params.add_vars:
        logger.info("  add_vars      : %s", " ".join(params.add_vars))
    if params.beams:
        logger.info("  beams         : %s", " ".join(params.beams))
    logger.info(
        "  radius=%.1fm  maxdiff=%.1fm  workers=%d",
        params.radius,
        params.maxdiff,
        params.max_workers,
    )
    logger.info("  output_dir    : %s", month_outdir)
    logger.info("=" * 72)

    # Load reference data #
    reference_path = Path(params.reference_dir) / f"{DATE_YEAR}/{DATE_MONTH}"
    reference_dir = reference_path if reference_path.is_dir() else Path(params.reference_dir)
    processor = ProcessData(params, AREA_OBJ, logger)
    reference_files = get_files_in_dir(reference_dir, DATE_YEAR, DATE_MONTH, "h5")  # is1 & is2
    if reference_files == []:  # icebridge/pre-icebridge
        reference_files = get_files_in_dir(reference_dir, DATE_YEAR, DATE_MONTH, "nc")

    if not reference_files:
        logger.error(
            "No reference files found in %s for %s-%s", reference_dir, DATE_YEAR, DATE_MONTH
        )
        sys.exit(1)
    logger.info("Found %d reference files in %s", len(reference_files), reference_dir)

    t_ref = time.perf_counter()
    chunksize = max(1, len(reference_files) // (params.max_workers * 4))
    logger.info(
        "Loading reference data (%d workers, chunksize %d)...", params.max_workers, chunksize
    )
    with ProcessPoolExecutor(max_workers=params.max_workers) as executor:
        ref_results = collect_with_progress(
            executor.map(processor.process_reference_file, reference_files, chunksize=chunksize),
            len(reference_files),
            logger,
            "reference",
        )
    valid_results = [result for result in ref_results if result is not None and result.size > 0]
    if not valid_results:
        logger.error(
            "No valid reference points loaded from %d files (none in area/hemisphere?)",
            len(reference_files),
        )
        sys.exit(1)
    reference_points = np.concatenate(valid_results)
    logger.info("Loaded %d reference points; de-duplicating...", len(reference_points))
    unique_indices = np.unique(reference_points[["x", "y", "h"]], return_index=True)[1]
    reference_points = reference_points[unique_indices]
    logger.info(
        "  %d unique reference points (%.1fs)",
        len(reference_points),
        time.perf_counter() - t_ref,
    )

    # Load altimetry data #
    if params.compare_to_self:
        logger.info("Performing reference self-comparison (altimetry = reference)")
        PREFIX = "neighbour_"
        altimetry_points = reference_points  # Compare is2 to itself
    else:
        PREFIX = ""
        altimetry_files = []
        for basepath in params.altim_dir:
            alt_path = Path(basepath) / f"{DATE_YEAR}/{DATE_MONTH}"
            altimetry_dir = alt_path if alt_path.is_dir() else Path(basepath)
            found = get_files_in_dir(altimetry_dir, DATE_YEAR, DATE_MONTH, "nc")
            logger.info("Found %d altimetry files in %s", len(found), altimetry_dir)
            altimetry_files.extend(found)
        if not altimetry_files:
            logger.error("No altimetry files found for %s-%s", DATE_YEAR, DATE_MONTH)
            sys.exit(1)

        t_alt = time.perf_counter()
        chunksize = max(1, len(altimetry_files) // (params.max_workers * 4))
        logger.info(
            "Loading altimetry data (band=%s, %d workers, chunksize %d)...",
            getattr(params, "band", "ku"),
            params.max_workers,
            chunksize,
        )
        with ProcessPoolExecutor(max_workers=params.max_workers) as executor:
            altim_results = collect_with_progress(
                executor.map(
                    processor.get_altimetry_data_array, altimetry_files, chunksize=chunksize
                ),
                len(altimetry_files),
                logger,
                "altimetry",
            )
        valid_altim_results = [r for r in altim_results if r is not None and r.size > 0]
        if not valid_altim_results:
            logger.error(
                "No valid altimetry points loaded from %d files (none in area/hemisphere?)",
                len(altimetry_files),
            )
            sys.exit(1)
        # Different mode/band files can carry different optional add_vars (e.g. HR Ku has
        # coherence, LR Ku and Ka do not), so concatenate on the union of columns.
        if len({r.dtype for r in valid_altim_results}) == 1:
            altimetry_points = np.concatenate(valid_altim_results)
        else:
            altimetry_points = concat_altimetry_results(valid_altim_results)
        # Missing in *some* files is normal; warn only if a requested add_var was found
        # in NO file at all, which usually means a typo or the wrong band/product.
        if params.add_vars:
            present = set(altimetry_points.dtype.names or ())
            absent = [v for v in params.add_vars if v.rsplit("/", 1)[-1] not in present]
            if absent:
                logger.warning(
                    "Requested --add_vars found in NO altimetry file: %s "
                    "(check spelling, --band, or product type)",
                    ", ".join(absent),
                )
        logger.info("Loaded %d altimetry points; de-duplicating...", len(altimetry_points))
        unique_indices = np.unique(altimetry_points[["x", "y", "h"]], return_index=True)[1]
        altimetry_points = altimetry_points[unique_indices]
        logger.info(
            "  %d unique altimetry points (%.1fs)",
            len(altimetry_points),
            time.perf_counter() - t_alt,
        )

    outfile = month_outdir / f"p2p_diffs_{params.area}"

    # Get elevation differences #
    t_match = time.perf_counter()
    logger.info(
        "Matching %d altimetry to %d reference points (radius=%.1fm, maxdiff=%.1fm)...",
        len(altimetry_points),
        len(reference_points),
        params.radius,
        params.maxdiff,
    )
    elev_differences = get_elev_differences(params, reference_points, altimetry_points, PREFIX)
    logger.info(
        "Found %d matched pairs in %.1fs",
        len(elev_differences["dh"]),
        time.perf_counter() - t_match,
    )
    if params.dem:
        logger.info("Applying DEM slope correction using %s", params.dem)
        elev_differences = correct_elevation_using_slope(elev_differences, params, logger, PREFIX)
    if len(elev_differences["dh"]) == 0:
        logger.error("No matched pairs within radius/maxdiff - nothing to save; exiting")
        sys.exit(1)

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
    logger.info(
        "Wrote %s.npz (%d points, vars: %s)",
        outfile,
        len(save_data["dh"]),
        ", ".join(save_data.keys()),
    )

    compute_elevation_stats(save_data, prefix=PREFIX, output_file=f"{outfile}_elevation_stats.csv")
    logger.info("Wrote %s_elevation_stats.csv", outfile)

    logger.info("Generating plots...")
    Polarplot(params.area).plot_points(
        {
            "name": "Difference_in_height_(dh)",
            "lats": save_data[f"{PREFIX}lats"],
            "lons": save_data[f"{PREFIX}lons"],
            "vals": save_data["dh"],
        },
        output_dir=str(month_outdir),
    )
    elev_dh_histograms(save_data["dh"], f"{outfile}_dh_histogram.png", bins=params.bins)
    elevation_dh_cumulative_dist(save_data["dh"], f"{outfile}_dh_cumulative_distribution.png")

    # Concise result summary so the key numbers are visible on the console
    dh_arr = np.asarray(save_data["dh"], dtype=float)
    logger.info("-" * 72)
    logger.info(
        "Done in %.1fs | %d matched points | dh mean=%.3f median=%.3f std=%.3f RMS=%.3f m",
        time.perf_counter() - t_start,
        dh_arr.size,
        float(np.mean(dh_arr)),
        float(np.median(dh_arr)),
        float(np.std(dh_arr)),
        float(np.sqrt(np.mean(dh_arr**2))),
    )
    logger.info("Results in %s", month_outdir)
    logger.info("-" * 72)

""" cpom.altimetry.tools.validate_against_is2.py

A tool to compare a selected month of altimetry mission data elevations to
Laser reference elevations.
Options to add additional variables to the output or compare refernce mission to itself. 

Supports : CS2 (Native l2 and cryotempo), S3A , S3B , ENVISAT, ERS1 , ERS2 
Against : ICESAT-2 (ATL06), Icebridge(ILATM2, ILUTP2) , Pre-Icebridge(BRMCR2, BLATM2) 

For full list of command line options:  validate_l2_is2_atl06.py -h
```
Run for cs2 data sin and lrm mode, with 2 beams. Parallelised across 20 workers: 
for m in {1..12}
do
./validate_against_is2.py 
--altim_dir /media/luna/archive/SATS/RA/CRY/L2I/SIN /media/luna/archive/SATS/RA/CRY/L2I/LRM
--reference_dir /media/luna/archive/SATS/LASER/ICESAT-2/ATL-06/versions/006 --year 2022 
--month $m --area antarctica_is --outdir /tmp--beams gt1l gt1r
--add_vars uncertainty_variable_name --max_workers 20 --chunksize 50 &
done

```
Run for CS2 CryoTEMPO comparison over Antarctica using 1 beam :
Cryotempo data requires an additional parameter --cryotempo_modes to filter to mode.
./validate_against_is2.py 
--altim_dir /media/luna/archive/SATS/RA/CRY/Cryo-TEMPO/BASELINE-D101/LAND_ICE/ANTARC
--reference_dir /media/luna/archive/SATS/LASER/ICESAT-2/ATL-06/versions/006 
--outdir /tmp --year 2020 --month 1 --area antarctica_is --beams gt2r --cryotempo_modes lrm sin &

```
Run to compare IS2 to itself over Greenland :
./compare_to_is2_atl06.py 
--reference_dir /media/luna/archive/SATS/LASER/ICESAT-2/ATL-06/versions/006--outdir /tmp 
--year 2022 --month 1 --area greenland --beams gt2r --max_workers 20 --compare_to_self 
"""

import argparse
import logging
import os
import sys
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import median_absolute_deviation, sigma_clipped_stats
from netCDF4 import Dataset  # pylint: disable=E0611
from scipy.spatial import cKDTree

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

    for log_file, level in [(log_file_info, logging.INFO), (log_file_error, logging.ERROR)]:
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
        "--dem", help="Digital Elevation Model.(Required for icebridge/pre-icebridge)"
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
        "---add_vars",
        nargs="+",
        help="[optional] additional variables in the altimetry file to include in the output."
        "Space-seperated list of : var1 var2 .",
    )
    parser.add_argument(
        "--cryotempo_modes",
        default=None,
        choices={"lrm", "sin", "sar", "all"},
        nargs="+",
        help="[optional, default= all] CryoTempo modes to use. Space-separated list of: lrm sin"
        "For non-CryoTempo L2 CS2 data, specify multiple --altim_dir paths instead.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="[optional, default=10] number of worker processes to use",
    )
    parser.add_argument(  # Default is False, becomes True if provided
        "--compare_to_icebridge",
        action="store_true",
        help="[optional] When set, compare to icebridge/pre_icebridge",
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

    if any(word in str(args.reference_dir).upper() for word in ["ICEBRIDGE", "PRE_ICEBRIDGE"]):
        args.compare_to_icebridge = True

    if args.compare_to_icebridge:
        if args.radius == 20:  # Change default if no value passed.
            args.radius = 1e4
    return args


def get_variable(nc: Dataset, nc_var_path: str) -> np.array:
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
                raise KeyError(f"NetCDF parameter '{nc_var_path}' not found.")
        return var[:]
    except KeyError as err:
        raise KeyError(f"Get variable failed with error {err}") from err


def get_default_variables(file: str) -> dict:
    """
    Return default variable names based on file naming patterns.

    Args:
        file (str): Path to the input file.

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
        "SR_2_LAN_LI": {  # Sentinel data
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
        },
        # Icebridge
        "ILATM2": {"lat": "lat", "lon": "lon", "elev": "hgt"},
        "ILUTP2": {"lat": "lat", "lon": "lon", "elev": "hgt"},
        "BLATM2": {"lat": "lat", "lon": "lon", "elev": "height"},
        "BRMCR2": {"lat": "lat", "lon": "lon", "elev": "elevation"},
    }

    for key, value in variable_map.items():  # Check for a matching pattern
        if key in basename:
            return value

    return None


def get_files_in_dir(
    directory: str,
    year: str,
    month: str,
    filetype: str,
) -> list[str]:
    """
    Find .nc or.h5 files in the specified directory.

    Args:
        directory (str): The directory to search for files.
        year (str): Year to process
        month (str): Month to process (0 filled)
        filetype (str) : nc or h5
    Returns:
        List[str]: A list of found .nc or .NC files with their full paths.
    """
    extensions = {"nc": "*.nc", "h5": "*.h5"}
    all_files = []
    pattern = extensions[filetype]
    all_files.extend(Path(directory).rglob(pattern))
    print(all_files)
    # Matches: YYYYMMDDT | YYYYMMDD_ | YYMMDD_ |YYYYMMDDHHMMSS
    file_basename_regex = re.compile(
        rf"(?:{year}{month}\d{{2}}(?:T\d+|_\d+)?|{year[-2:]}{month}\d{{2}}_)"
    )
    
    valid_files = [  # Exclude filenames containing multiple dots before the extension
        f
        for f in all_files
        if any(file_basename_regex.search(part) for part in f.parts)  # Check if regex matches any part of the path
        and not re.search(r"\..+\.", f.stem)  # Exclude filenames containing multiple dots before the extension
    ]

    if valid_files:
        return valid_files

    # file_dir_regex = re.compile(rf"{year}\.{month}\.\d{{2}}")  # Matches YYYY.DD.MM/
    # return [
    #     str(file) for file in all_files if file_dir_regex.fullmatch(part) for part in file.parts
    # ]


class ProcessData:
    """Class to process laser and altimetry data files by extracting and filtering
    elevation data. To be run by a multiproccessor.

    Methods:
        get_altimetry_data_array: Extracts and processes altimetry data from NetCDF files.
        get_is2_data_array : Extracts and processes is2 elevation data from HDF5 files.
        get_icebridge_data_array : Extracts and processes icebridge data from NetCDF files. 
        _fill_empty_latlon_with_nadir: Fills missing lat/lon values using nadir coordinates.
        _get_cryotempo_filters : Filters cryotempo data to valid modes. 
    """

    def __init__(self, args, area, log):
        self.args = args
        self.area = area
        self.log = log

    def _get_cryotempo_filters(self, nc, args):
        """Check Cryotempo data is in a valid mode."""
        if args.cryotempo_modes == "all":
            return None

        mode_map = {"lrm": 1, "sar": 2, "sin": 3}
        valid_modes = {mode_map[mode] for mode in args.cryotempo_modes if mode in mode_map}

        instrument_mode = get_variable(nc, "instrument_mode")
        valid_mask = np.isin(instrument_mode, list(valid_modes))

        return valid_mask if valid_mask.any() else None

    def _fill_empty_latlon_with_nadir(
        self, nc: str, lat: np.array, lon: np.array, config: dict
    ) -> np.array:
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
        bad_indices = np.flatnonzero(lon.mask)  # Find empty longitude values
        if bad_indices.size > 0:
            lat_nadir = get_variable(nc, config["lat_nadir"])
            lon_nadir = get_variable(nc, config["lon_nadir"])
            lat[bad_indices] = lat_nadir[bad_indices]
            lon[bad_indices] = np.mod(lon_nadir[bad_indices], 360)
        return lat, lon

    def get_altimetry_data_array(self, filename: str) -> np.ndarray:
        """Extract and filter data from an altimetry data file.
        Args:
            filename (str): Altimetry data filename (*.nc)
        Raises:
            ValueError: If variable lengths do not match.
        Returns:
            np.ndarray: Structured nd.array containing x,y,h fields
        """
        try:
            with Dataset(filename) as nc:
                config = get_default_variables(filename)
                lats = get_variable(nc, config["lat"])
                lons = np.mod(get_variable(nc, config["lon"]), 360)

                if config["lat_nadir"]:
                    lats, lons = self._fill_empty_latlon_with_nadir(nc, lats, lons, config)

                lats, lons, _, _ = self.area.inside_latlon_bounds(lats, lons)
                x, y = self.area.latlon_to_xy(lats, lons)

                if self.args.cryotempo_modes is not None:  # Filter modes for cryotempo
                    surface_type_mask = self._get_cryotempo_filters(nc, self.args)
                    x, y = x[surface_type_mask], (
                        x[surface_type_mask] if surface_type_mask else (x, y)
                    )

                idx, _ = self.area.inside_mask(x, y)
                if not idx.size:
                    return None

                x, y, elev = x[idx], y[idx], get_variable(nc, config["elev"])[idx]

                if not len(x) == len(y) == len(elev):
                    raise ValueError("Mismatch in variable lengths for x,y,h")

                add_vars = self.args.add_vars or []

                additional_data = {
                    var.rsplit("/", 1)[-1]: get_variable(nc, var)[idx]
                    for var in add_vars
                    if var in nc.variables
                }
                if not all(len(data) == len(x) for data in additional_data.values()):
                    raise ValueError("Mismatch in additional variable lengths.")

                # dtype = [("x", "float64"), ("y", "float64"), ("h", "float64")] + [
                #     (var, str(data.dtype)) for var, data in additional_data.items()
                # ]
            return np.array(
                        list(zip(x, y, elev, *(additional_data[var] for var in additional_data))),
                        dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")]
                        + [(var, str(data.dtype)) for var, data in additional_data.items()],
                    )
        except OSError as err:
            self.log.error(f"Error loading altimetry data file {filename}, fialed with : {err}")
            return None

    def get_is2_data_array(self, filename: str) -> np.ndarray:
        """Extract and filter data from an is2 data file.
        Args:
            filename (str): IS2 data filename (.h5)

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
                        x, y = self.area.latlon_to_xy(lats, lons)
                        elevation = elevation[bounded_indices]

                        if not len(x) == len(y) == len(elevation):
                            raise ValueError("Mismatch in variable lengths for x,y,h")

                        points.append(
                            np.array(
                                list(zip(x, y, elevation)),
                                dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")],
                            )
                        )
                    except KeyError:
                        self.log.error("Missing data in %s for beam %s", filename, beam)
                        continue
        except OSError as err:
            self.log.error("Cannot open file %s: %s", filename, err)
            return np.array([], dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")])
        return (
            np.concatenate(points)
            if points
            else np.array([], dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")])
        )

    def get_icebridge_data_array(self, filename: str) -> np.ndarray:
        """Extract and filter icebridge and pre-icebridge data.
        Args:
            filename (str): Filename
        Raises:
            ValueError: If arrays not equal in length
        Returns:
            np.ndarray: Structured nd.array containing x,y,h fields
        """
        with Dataset(filename) as nc:
            config = get_default_variables(filename)
            print("THE CONFIG IS", config)
            lats, lons = get_variable(nc, config["lat"]), np.mod(
                get_variable(nc, config["lon"]), 360
            )

            lats, lons, bounded_indices, _ = self.area.inside_latlon_bounds(lats, lons)
            x, y = self.area.latlon_to_xy(lats, lons)
            elevation = get_variable(nc, config["elev"])[bounded_indices]

            if not len(x) == len(y) == len(elevation):
                raise ValueError("Mismatch in variable lengths for x,y,h")

            return np.array(
                list(zip(x, y, elevation)),
                dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")],
            )

    def process_reference_file(self, file_path: str):
        """Process reference files,
        Handling both ATL06 and Icebridge / Pre Icebridge data"""
        file_path = Path(file_path)

        if file_path.suffix == ".nc":
            return self.get_icebridge_data_array(file_path)
        if file_path.suffix == ".h5":
            return self.get_is2_data_array(file_path)
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
        nd,array: Dh between altimetry and is2 elevations.
    """

    is2_tree = cKDTree(np.c_[laser_points["x"], laser_points["y"]])
    add_vars = [var for var in altimetry_points.dtype.names if var not in {"x", "y", "h"}]
    results = {
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

            if np.abs(dh) < args.maxdiff:
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


def slope_correction(data, args, log, prefix)-> np.array:
    """Slope Correction from reference DEM 
    Args:
        data (nd.array): Output of "get_elev_differences"
        args (argparse.Namespace): Configuration parameters
        log (Logger): Logger
        prefix (str): Reference data prefix 
    Returns
        nd.array: Slope corrected Dh between altimetry and is2 elevations.
    """
    log.info("Performing slope correction")
    dem = Dem(args.dem)
    dh_dem_elev = (dem.interp_dem(data[f"{prefix}x"], data[f"{prefix}y"])) - (
        dem.interp_dem(data["reference_x"], data["reference_y"])
    )
    data["reference_h"] = data["reference_h"] + dh_dem_elev
    data["dh"] = data["dh"] - dh_dem_elev

    data = {key: np.array(data[key]) for key in data}
    return {key: val[~np.isnan(data["dh"])] for key, val in data.items()}


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
        sys.exit("Failed  create directory %s: %s", month_outdir, e)

    logger = setup_logging(
        log_file_info=month_outdir / f"{DATE_YEAR}{DATE_MONTH}.info.log",
        log_file_error=month_outdir / f"{DATE_YEAR}{DATE_MONTH}.errors.log",
    )
    logger.info("Start processing with arguments %s", params)

    # Load reference data #
    reference_path = Path(params.reference_dir) / f"{DATE_YEAR}/{DATE_MONTH}"
    reference_dir = reference_path if reference_path.is_dir() else Path(params.reference_dir)

    processor = ProcessData(params, AREA_OBJ, logger)
    file_ext = ("nc" if params.compare_to_icebridge else "h5")
    reference_files = get_files_in_dir(reference_dir, DATE_YEAR, DATE_MONTH, file_ext)
    logger.info("Loaded %d reference data files", len(reference_files))

    chunksize = max(1, len(reference_files) // (params.max_workers * 4))
    logger.info("Chunksize %d", chunksize)
    with ProcessPoolExecutor(max_workers=params.max_workers) as executor:
        ref_results = list(
            executor.map(processor.process_reference_file, reference_files, chunksize=chunksize)
        )
    valid_results = [result for result in ref_results if result is not None]
    reference_points = np.concatenate(valid_results)
    unique_indices = np.unique(reference_points[["x", "y", "h"]], return_index=True)[1]
    reference_points = reference_points[unique_indices]
    logger.info("Loaded reference data points, len : %d", len(reference_points))

    ref_missions = ["ATL06", "ILATM2", "ILUTP2", "BLATM2", "BRMCR2"]
    REF_MISSION = next(
        (m for m in ref_missions if m in os.path.basename(str(reference_files[0]))), "UNDEFINED"
    )

    # Load altimetry data #
    if params.compare_to_self:
        logger.info("Performing reference self-comparison")
        PREFIX = "neighbour_"
        altimetry_points = reference_points  # Compare is2 to itself
        outfile = month_outdir / f"{REF_MISSION}_vs_{REF_MISSION}_p2p_diffs_{params.area}"
        if params.beams:
            outfile = (
                month_outdir
                / f"{REF_MISSION}_vs_{REF_MISSION}_{''.join(params.beams)}_p2p_diffs_{params.area}"
            )
    else:
        logger.info("Comparing reference to altimetry")
        PREFIX = ""
        altimetry_files = []
        for basepath in params.altim_dir:
            alt_path = Path(basepath) / f"{DATE_YEAR}/{DATE_MONTH}"
            altimetry_dir = alt_path if alt_path.is_dir() else Path(basepath)

            altimetry_files.extend(get_files_in_dir(altimetry_dir, DATE_YEAR, DATE_MONTH, "nc"))
        logger.info("Loaded %d altimetry data files", len(altimetry_files))

        chunksize = max(1, len(altimetry_files) // (params.max_workers * 4))
        logger.info("Chunksize %d", chunksize)
        with ProcessPoolExecutor(max_workers=params.max_workers) as executor:
            altim_results = list(
                executor.map(processor.get_altimetry_data_array, altimetry_files, chunksize=chunksize)
            )
        valid_altim_results = [result for result in altim_results if result is not None]
        altimetry_points = np.concatenate(valid_altim_results)
        unique_indices = np.unique(altimetry_points[["x", "y", "h"]], return_index=True)[1]
        altimetry_points = altimetry_points[unique_indices]
        logger.info("Loaded altimetry data points, len : %d", len(altimetry_points))

        mission_ids = {"CRY", "S3A", "S3B", "ERS2", "ERS1", "ENV"}
        MISSION = next((i for i in Path(altimetry_files[0]).parts if i in mission_ids), "UNKNOWN")

        if params.beams:
            outfile = (
                month_outdir
                / f"{MISSION}_minus_{REF_MISSION}_{''.join(params.beams)}_p2p_diffs_{params.area}"
            )
        else:
            outfile = month_outdir / f"{MISSION}_minus_{REF_MISSION}_p2p_diffs_{params.area}"

    # Get elevation differences #
    elev_differences = get_elev_differences(
        params, reference_points, altimetry_points, PREFIX
    )
    if params.compare_to_icebridge:
        slope_correction(elev_differences, params, logger, PREFIX)

    # Convert to lat/lon #
    alt_lons, alt_lats = AREA_OBJ.xy_to_latlon(
        elev_differences[f"{PREFIX}x"], elev_differences[f"{PREFIX}y"]
    )
    reference_lons, reference_lats = AREA_OBJ.xy_to_latlon(
        elev_differences["reference_x"], elev_differences["reference_y"]
    )

    # Output                 #
    logger.info("Saving month data to %s", outfile)
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

    logger.info("Saving data to %s", outfile)
    np.savez(f"{outfile}.npz", **save_data)

    compute_elevation_stats(save_data, prefix=PREFIX, output_file=f"{outfile}_elevation_stats.csv")

    Polarplot(params.area).plot_points(
        {
            "name": "Difference_in_height_(dh)",
            "lats": save_data[f"{PREFIX}lats"],
            "lons": save_data[f"{PREFIX}lons"],
            "vals": save_data["dh"],
        },
        output_dir=month_outdir,
    )

    elev_dh_histograms(save_data["dh"], f"{outfile}_dh_histogram.png", bins=params.bins)
    elevation_dh_cumulative_dist(save_data["dh"], f"{outfile}_dh_cumulative_distribution.png")

""" cpom.altimetry.tools.validate_against_is2.py
# Purpose 

Tool to compare a selected month of altimetry mission data elevations to
ICESat-2 ATL-06 elevations. Optionally add additional variables , or compare is2 
to itself.

Examples:

For full list of command line options: 

```
validate_l2_is2_atl06.py -h
```

Example of a 12-month run for the Antarctica Ice Sheet. 
Run for cs2 data sin and lrm mode, with 2 beams. Parallelised across 20 workers. 
for m in {1..12}
do
./validate_against_is2.py 
--altim_dir /media/luna/archive/SATS/RA/CRY/L2I/SIN /media/luna/archive/SATS/RA/CRY/L2I/LRM
--reference_dir /media/luna/archive/SATS/LASER/ICESAT-2/ATL-06/versions/006 --year 2022 
--month $m --area antarctica_is --outdir /tmp--beams gt1l gt1r
--add_vars uncertainty_variable_name --max_workers 20 --chunksize 50 &
done

```
Run for CS2 CryoTEMPO comparison over Antarctica using 1 beam.
Cryotempo data requires an additional parameter --cryotempo_modes to filter to mode.
./validate_against_is2.py 
--altim_dir /media/luna/archive/SATS/RA/CRY/Cryo-TEMPO/BASELINE-D101/LAND_ICE/ANTARC
--reference_dir /media/luna/archive/SATS/LASER/ICESAT-2/ATL-06/versions/006 
--outdir /tmp --year 2020 --month 1 --area antarctica_is --beams gt2r --cryotempo_modes lrm sin &

```
Run to compare IS2 comparison over Greenland to nearby IS2 points:
for m in {1..12}
do
    ./compare_to_is2_atl06.py 
    --reference_dir /media/luna/archive/SATS/LASER/ICESAT-2/ATL-06/versions/006 --outdir /tmp
    --year 2022 --month $m --area greenland --beams gt2r --max_workers 20 
done
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
from matplotlib.gridspec import GridSpec
from netCDF4 import Dataset  # pylint: disable=E0611
from scipy import stats
from scipy.spatial import cKDTree

from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area, list_all_area_definition_names_only
from cpom.dems.dems import Dem

# import mpl_scatter_density


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

    # comment this to suppress console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    log.addHandler(stream_handler)

    for log_file, level in [(log_file_info, logging.INFO), (log_file_error, logging.ERROR)]:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(log_format)
        file_handler.setLevel(level)
        log.addHandler(file_handler)

    return log


def parse_args() -> argparse.Namespace:
    """Declare command line arguments and defaut values
    Returns:
        argparse.ArgumentParser: Parsed input arguments
    """
    # Initiate the command line parser
    parser = argparse.ArgumentParser(description="Retrieve command line arguments")

    # Add long and short command line arguments

    parser.add_argument(
        "--reference_dir",
        "-id",
        help="is2_atl06, icebridge or pre-icebridge data directory \
            If a subdirectory matching 'YEAR/MONTH' exists it will be picked up automatically.",
        required=True,
    )
    parser.add_argument(
        "--altim_dir",
        "-ad ",
        nargs="+",
        help="Altimetry data directory('s) \
        Supports multiple directories by providing a space seperated list of path1 path2\
        If a subdirectory matching 'YEAR/MONTH' exists it will be picked up automatically. \
        NOTE : If ommitted this script will compare the reference to itself",
    )
    parser.add_argument(
        "--year", "-y", type=int, help="year number (YYYY): comparison year", required=True
    )
    parser.add_argument(
        "--month",
        "-mn",
        type=int,
        choices=range(1, 13),
        help="comparison month (1-12)",
        required=True,
    )
    parser.add_argument(
        "--area",
        "-a",
        help="comparison area, cpom area name",
        required=True,
    )
    parser.add_argument("--dem", "-dem", help="Digital Elevation Model", required=True)

    parser.add_argument(
        "--max_workers",
        "-mw",
        type=int,
        default=10,
        help="[optional, default=10] number of worker processes to use",
    )
    parser.add_argument("--outdir", "-o", help="output directory path for results", required=True)
    parser.add_argument(
        "--beams",
        "-b",
        default=["gt2r"],
        choices={"gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"},
        nargs="+",
        help="[optional, default=gt2r ] IS2 beams to use. \
                        Space separated list of: gt1l gt1r gt2l gt2r gt3l gt3r",
    )
    parser.add_argument(
        "--radius",
        "-r",
        type=float,
        default=20.0,
        help="[optional] search radius in m, default (20m : atl06 , 20km : icebridge)",
    )
    parser.add_argument(
        "--maxdiff",
        "-md",
        type=float,
        default=100.0,
        help="[optional] maximum allowed difference in m between reference and \
        altimeter points, default (100.0 m : atl06, 1000.0m : icebridge). \
        Differences > are not saved.",
    )
    parser.add_argument(
        "---add_vars",
        "-av",
        nargs="+",
        help="[optional] additional variables in the altimetry file to include in the output."
        "Space-seperated list of : var1 var2 .",
    )
    parser.add_argument(
        "--cryotempo_modes",
        "-mo",
        default=None,
        choices={"lrm", "sin", "sar", "all"},
        nargs="+",
        help="[optional, default= all] CryoTempo modes to use. Space-separated list of: lrm sin"
        "For non-CryoTempo L2 CS2 data, specify multiple --altim_dir paths instead.",
    )
    parser.add_argument(
        "--compare_to_icebridge",
        action="store_true",  # Default is False, becomes True if provided
        help="[optional] When set, compare to icebridge/pre_icebridge rather than is2 atl06",
    )
    parser.add_argument(
        "--compare_to_self",
        action="store_true",  # Default is False, becomes True if provided
        help="[optional] When set, compare reference data to itself rather than altimetry",
    )
    args = parser.parse_args()

    if not args.compare_to_self and args.altim_dir is None:
        parser.error("--altim_dir is required unless --compare_to_self is set")

    if args.area not in list_all_area_definition_names_only():
        parser.error(f"{args.area} not a valid cpom area name")

    if any(word in str(args.reference_dir).upper() for word in ["ICEBRIDGE", "PRE_ICEBRIDGE"]):
        args.compare_to_icebridge = True

    if args.compare_to_icebridge:
        if args.radius == 20:  # Change default if no value passed.
            args.radius = 20000
        if args.maxdiff == 100:
            args.maxdiff = 1000

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
        file (str): path to input file.

    Returns:
        dict: Dictionary of default variable names
    """
    basename = os.path.basename(file)
    # Dictionary mapping filename patterns to default variables
    variable_map = {
        "CS_OFFL_SIR_LRM": {
            "lat_nadir": "lat_20_ku",
            "lon_nadir": "lon_20_ku",
            "lat": "lat_poca_20_ku",
            "lon": "lon_poca_20_ku",
            "elevation": "height_3_20_ku",
        },
        "CS_OFFL_SIR_SIN": {
            "lat_nadir": "lat_20_ku",
            "lon_nadir": "lon_20_ku",
            "lat": "lat_poca_20_ku",
            "lon": "lon_poca_20_ku",
            "elevation": "height_1_20_ku",
        },
        # Sentinel data
        "SR_2_LAN_LI": {
            "lat_nadir": "lat_20_ku",
            "lon_nadir": "lon_20_ku",
            "lat": "lat_cor_20_ku",
            "lon": "lon_cor_20_ku",
            "elevation": "elevation_ocog_20_ku_filt",
        },
        # Cryotempo
        "CS_OFFL_SIR_TDP": {
            "lat": "latitude",
            "lon": "longitude",
            "elevation": "elevation",
        },
        "ATL06": {
            "lat": "{beam}/land_ice_segments/latitude",
            "lon": "{beam}/land_ice_segments/longitude",
            "elevation": "{beam}/land_ice_segments/h_li",
        },
        # Icebridge
        "ILATM2": {
            "lat": "Latitude(deg)",
            "lon": "Longitude(deg)",
            "elevation": "WGS84_Ellipsoid_Height(m)",
        },
        "ILUTP2": {"lat": "LON", "lon": "LAT", "elevation": "SRF_ELEVATION"},
    }

    # Check basename for a matching pattern
    for key, value in variable_map.items():
        if key in basename:
            return value
    return None


def find_files_in_dir(
    directory: str,
    year: str,
    month: str,
    filetype: list[str],
) -> list[str]:
    """
    Find .nc or.h5 files in the specified directory.

    Args:
        directory (str): The directory to search for files.
        recursive (bool): If True, search recursively in subdirectories.
        max_files (int|None): if not None, limit number of files read to this number

    Returns:
        List[str]: A list of found .nc or .NC files with their full paths.
    """
    extensions = {"nc": "*.nc", "h5": "*.h5", "txt": "*.txt", "csv": "*.csv"}

    all_files = []
    for ft in filetype:
        if ft not in extensions:
            raise ValueError(f"Invalid filetype: {ft}. Must be 'nc' or 'h5'.")
        pattern = extensions[ft]
        all_files.extend(Path(directory).rglob(pattern))

    pattern = (
        rf"{year}{month}\d{{2}}[T_]|\d{{2}}{year}{month}\d{{2}}_"  # YYYYMMDDT / YYYYMMDD_ / YYMMDD_
    )
    regex = re.compile(pattern)
    return [file for file in all_files if regex.search(file.name)]


class ProcessData:
    """Class to process is2 and altimetry data files by extracting and filtering
    elevation data. To be run by a multiproccessor.

    Methods:
        process_file_altimetry: Extracts and processes altimetry data from NetCDF files.

        process_file_is2 : Extracts and processes is2 elevation data from HDF5 files.

        _fill_empty_latlon_with_nadir: Fills missing lat/lon values using nadir coordinates.
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
            lat (np.array): array of latitudes
            lon (np.array): array of longitudes
            config (dict): dictionary of variable names

        Returns:
            (np.array, np.arrray): Filled lat/lon arrays
        """
        # Find empty longitude values
        bad_indices = np.flatnonzero(lon.mask)  # Is this needed , check.
        if bad_indices.size > 0:
            lat_nadir = get_variable(nc, config["lat_nadir"])
            lon_nadir = get_variable(nc, config["lon_nadir"])
            lat[bad_indices] = lat_nadir[bad_indices]
            lon[bad_indices] = np.mod(lon_nadir[bad_indices], 360)
        return lat, lon

    def process_file_altimetry(self, filename: str) -> np.ndarray:
        """Extract and filter data from an altimetry data file.

        Args:
            filename (str): altimetry data filename (*.nc)

        Raises:
            ValueError: If variable lengths do not match.

        Returns:
            np.ndarray: structured nd.array containing x,y,h fields
                and additonal data fields passed by args.add_vars.
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
                    if surface_type_mask:
                        x, y = x[surface_type_mask], x[surface_type_mask]

                if not x.size:
                    return None

                idx, _ = self.area.inside_mask(x, y)
                if not idx.size:
                    return None

                elev = get_variable(nc, config["elevation"])[idx]
                x, y = x[idx], y[idx]

                if not len(x) == len(y) == len(elev):
                    raise ValueError("Mismatch in variable lengths for x,y,h")

                if self.args.add_vars:
                    # Add additonal variables
                    additional_data = {
                        var.rsplit("/", 1)[-1]: get_variable(nc, var)[idx]
                        for var in self.args.add_vars
                        if var in nc.variables
                    }

                    if not all(len(data) == len(x) for data in additional_data.values()):
                        self.log.error("Mismatch in variable lengths")
                        raise ValueError("Mismatch in variable lengths.")

                    altimeter_points = np.array(
                        list(zip(x, y, elev, *(additional_data[var] for var in additional_data))),
                        dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")]
                        + [(var, str(data.dtype)) for var, data in additional_data.items()],
                    )
                else:
                    altimeter_points = np.array(
                        list(zip(x, y, elev)),
                        dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")],
                    )
            return altimeter_points
        except OSError as err:
            self.log.error(f"Error loading altimetry data file {filename}, fialed with : {err}")
            return None

    def process_file_is2(self, filename: str) -> np.ndarray:
        """Extract and filter data from an is2 data file.
        Args:
            filename (str): IS2 data filename (.h5)

        Raises:
            ValueError: If variable lengths do not match

        Returns:
            np.ndarray: structured nd.array containing x,y,h fields
        """
        try:
            with h5py.File(filename, "r") as nc:
                points = []
                # self.log.info("filename %s", filename)
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

                        elevation = get_variable(nc, config["elevation"])
                        ok = np.flatnonzero(
                            (
                                get_variable(nc, f"{beam}/land_ice_segments/atl06_quality_summary")
                                == 0
                            )
                            & (elevation <= 10e3)
                        )
                        if len(ok) == 0:
                            continue

                        elevation = elevation[ok]
                        lats = get_variable(nc, config["lat"])[ok]
                        lons = get_variable(nc, config["lon"])[ok] % 360.0

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

    def process_files_icebridge(self, files: list[str]) -> np.ndarray:
        """Extract and filter data from icebridge/ pre-icebridge data files.
        Args:
            files (list[str]): List of icebridge files.
        Returns:
            np.ndarray: structured nd.array containing x,y,h fields
        """
        print(files[0])
        config = get_default_variables(files[0])
        self.log.info("Config: %s", config)
        with open(files[0], "r", encoding="utf-8") as f:
            last_line = (
                [line.strip() for line in f if line.startswith("#")][-1]
                .lstrip("#")
                .strip()
                .split(",")
            )
            column_names = [col.strip() for col in last_line]

        def read_file(file):
            try:
                df = pd.read_csv(
                    file, comment="#", names=column_names, usecols=column_names, dtype=float
                )
                return df
            except ValueError as e:
                self.log.error("Error reading file %s, failed with error : %d ", file, e)
                return None

        df = pd.concat(map(read_file, files), ignore_index=True)

        lats, lons = df[config["lat"]].values, np.mod(df[config["lon"]].values, 360)
        elevation = np.array(df[config["elevation"]])

        lats, lons, bounded_indices, _ = self.area.inside_latlon_bounds(lats, lons)
        x, y = self.area.latlon_to_xy(lats, lons)
        elevation = elevation[bounded_indices]

        if not len(x) == len(y) == len(elevation):
            raise ValueError("Mismatch in variable lengths for x,y,h")

        points = np.array(
            list(zip(x, y, elevation)),
            dtype=[("x", "float64"), ("y", "float64"), ("h", "float64")],
        )
        return points


def get_elev_differences(
    args: argparse.Namespace,
    is2_points: np.ndarray,
    altimeter_points: np.ndarray,
    dem: Dem,
    log,
    prefix: str = "",
) -> dict:
    """
    Calculate the elevation differences between IS2 points and altimeter points
    within a specified search radius. Filters to elevation points
    less than the maximum elevation difference.

    Args:
        args (argparse.Namespace): Configuration parameters
        is2_points (np.ndarray): is2 array containing x,y,h
        altimeter_points (np.ndarray): altimetry array containing x,y,h

    Returns:
        dict: Ouputted dh between altimetry and is2, plus associated variable data.
    """

    is2_tree = cKDTree(np.c_[is2_points["x"], is2_points["y"]])

    add_vars = (
        {
            var: []
            for var in [
                val for val in list(altimeter_points.dtype.names) if val not in {"x", "y", "h"}
            ]
        }
        if not args.compare_to_self
        else {}
    )

    results = {
        "dh": [],
        "sep_dist": [],
        "reference_x": [],
        "reference_y": [],
        "reference_h": [],
        f"{prefix}x": [],
        f"{prefix}y": [],
        f"{prefix}h": [],
        **{var: [] for var in add_vars},
    }

    for altimeter_point in altimeter_points:
        indices = is2_tree.query_ball_point(
            (altimeter_point["x"], altimeter_point["y"]), args.radius  # Query point
        )  # Find reference points within search_radius

        if args.compare_to_self:  # Remove comparision to self
            indices = [
                i
                for i in indices
                if is2_points[i]["x"] != altimeter_point["x"]
                and is2_points[i]["y"] != altimeter_point["y"]
            ]

        if len(indices) > 0:
            for idx in indices:
                is2_point = is2_points[idx]

                separation_distance = np.sqrt(
                    (altimeter_point["x"] - is2_point["x"]) ** 2
                    + (altimeter_point["y"] - is2_point["y"]) ** 2
                )

                dh = altimeter_point["h"] - is2_point["h"]
                if np.abs(dh) < args.maxdiff:
                    results["dh"].append(dh)
                    results["sep_dist"].append(separation_distance)
                    results[f"{prefix}x"].append(altimeter_point["x"])
                    results[f"{prefix}y"].append(altimeter_point["y"])
                    results[f"{prefix}h"].append(altimeter_point["h"])
                    results["reference_x"].append(is2_point["x"])
                    results["reference_y"].append(is2_point["y"])
                    results["reference_h"].append(is2_point["h"])

                    for var in add_vars:
                        results[var].append(altimeter_point[var])

    if args.compare_to_icebridge:
        log.info("Performing slope correction")
        dh_dem_elev = (dem.interp_dem(results[f"{prefix}x"], results[f"{prefix}y"])) - (
            dem.interp_dem(results["reference_x"], results["reference_y"])
        )
        results["reference_h"] = results["reference_h"] + dh_dem_elev
        results["dh"] = results["dh"] - dh_dem_elev

    return results


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
        output (dict): Dictionary containing elevation differences ('dh'), heights ('h'),
                        and separation distances ('sep_dist').
    Returns:
        txt file containing summary statistics
    """
    this_h_diff, this_h_diff_abs = output["dh"], np.abs(output["dh"])

    # Compute R² value for height correlation
    _, _, r_value, _, _ = stats.linregress(output[f"{prefix}h"], output[f"{prefix}h"])
    p2p_r2_hgt = r_value**2

    # Compute separation distance metrics if available
    p2p_mean_sep_dist = np.mean(output["seperation_distance"])
    p2p_std_dev_sep_dist = np.std(output["seperation_distance"])

    def compute_stats(data):
        mean, median, std = np.mean(data), np.median(data), np.std(data)
        mad, rms = median_absolute_deviation(data), np.sqrt(np.mean(np.square(data)))

        two_sigma_mean, _, two_sigma_std = sigma_clipped_stats(data, sigma=2)
        three_sigma_mean, _, three_sigma_std = sigma_clipped_stats(data, sigma=3)
        return [
            mean,
            median,
            std,
            mad,
            rms,
            two_sigma_mean,
            two_sigma_std,
            three_sigma_mean,
            three_sigma_std,
        ]

    # Compute stats for both signed and absolute differences
    stats_signed = compute_stats(this_h_diff) + [
        p2p_r2_hgt,
        p2p_mean_sep_dist,
        p2p_std_dev_sep_dist,
    ]
    stats_abs = compute_stats(this_h_diff_abs) + [
        p2p_r2_hgt,
        p2p_mean_sep_dist,
        p2p_std_dev_sep_dist,
    ]

    # Column names
    columns = [
        "mean_hgt",
        "median_hgt",
        "std_hgt",
        "MAD_hgt",
        "RMS_hgt",
        "2_sigma_mean_hgt",
        "2_sigma_std_hgt",
        "3_sigma_mean_hgt",
        "3_sigma_std_hgt",
        "r2",
        "mean_dist_(m)",
        "std_dist_(m)",
    ]

    pd.DataFrame(
        [stats_signed, stats_abs],
        index=["Signed Differences", "Absolute Differences"],
        columns=columns,
    ).to_csv(output_file)


if __name__ == "__main__":
    params = parse_args()

    area_obj = Area(params.area)
    date_year = params.year
    date_month = str(params.month).zfill(2)
    dem_obj = Dem(params.dem)
    # ----------------------------------#
    # Create or clear output directory #
    # ----------------------------------#
    month_outdir = Path(params.outdir) / f"{date_year}/{date_month}"
    try:
        month_outdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit("Failed  create directory %s: %s", month_outdir, e)

    logfile = f"{params.outdir}/{params.month:02d}{params.year:4d}"

    logger = setup_logging(
        log_file_info=logfile + ".info.log",
        log_file_error=logfile + ".errors.log",
    )
    logger.info("Start processing with arguments %s", params)

    # --------------------#
    # Load reference data #
    # --------------------#
    reference_path = Path(params.reference_dir) / f"{date_year}/{date_month}"
    reference_dir = reference_path if reference_path.is_dir() else Path(params.reference_dir)
    processor = ProcessData(params, area_obj, logger)

    if params.compare_to_icebridge:
        reference_files = find_files_in_dir(
            directory=reference_dir,
            year=date_year,
            month=date_month,
            filetype=["csv", "txt"],
        )
        logger.info("Loaded %d icebridge/pre-icebridge data files", len(reference_files))
        reference_points = processor.process_files_icebridge(reference_files)
        logger.info("Loaded icebridge/pre-icebridge data points, len : %d", len(reference_points))
    else:
        reference_files = find_files_in_dir(
            directory=reference_dir, year=date_year, month=date_month, filetype=["h5"]
        )

        logger.info("Loaded %d is2-atl06 data files", len(reference_files))
        chunksize = max(1, len(reference_files) // (params.max_workers * 4))
        logger.info("Chunksize %d", chunksize)

        with ProcessPoolExecutor(max_workers=params.max_workers) as executor:
            is2_results = list(
                executor.map(processor.process_file_is2, reference_files, chunksize=chunksize)
            )
        valid_is2_results = [result for result in is2_results if result is not None]
        reference_points = np.concatenate(valid_is2_results)

        logger.info("Loaded is2 data points, len : %d", len(reference_points))

    REF_MISSION = str(
        [
            i
            for i in Path(reference_files[0]).parts
            if i in {"ATL06", "ILATM2", "ILUTP2", "BLATM2", "BRMCR2"}
        ][0]
    )

    # ---------------------------#
    # Get elevation differences #
    # ---------------------------#
    if params.compare_to_self:
        logger.info("Performing reference self-comparison")
        PREFIX = "neighbour_"
        altimetry_points = reference_points  # Compare is2 to itself
        outfile = month_outdir / f"reference_vs_reference_p2p_diffs_{params.area}.npz"
        outfile = (
            month_outdir
            / f"{REF_MISSION}_vs_{REF_MISSION}_{''.join(params.beams)}_p2p_diffs_{params.area}"
        )

    else:
        # ---------------------#
        # Load altimetry data #
        # ---------------------#
        logger.info("Comparing reference to altimetry")
        PREFIX = ""

        altimetry_files = []
        for basepath in params.altim_dir:
            alt_path = Path(basepath) / f"{date_year}/{date_month}"
            altimetry_dir = alt_path if alt_path.is_dir() else Path(basepath)

            altim_files = find_files_in_dir(
                directory=altimetry_dir,
                year=date_year,
                month=date_month,
                filetype=["nc"],
            )
            altimetry_files.extend(altim_files)

        logger.info("Loaded %d altimetry data files", len(altimetry_files))

        chunksize = max(1, len(altimetry_files) // (params.max_workers * 4))
        logger.info("Chunksize %d", chunksize)
        with ProcessPoolExecutor(max_workers=params.max_workers) as executor:
            altim_results = list(
                executor.map(processor.process_file_altimetry, altimetry_files, chunksize=chunksize)
            )
        valid_altim_results = [result for result in altim_results if result is not None]
        altimetry_points = np.concatenate(valid_altim_results)
        logger.info("Loaded altimetry data points, len : %d", len(altimetry_points))
        MISSION = str(
            [
                i
                for i in Path(altimetry_files[0]).parts
                if i in {"CRY", "S3A", "S3B", "ERS2", "ERS1", "ENV"}
            ][0]
        )
        outfile = (
            month_outdir
            / f"{MISSION}_minus_{REF_MISSION}_{''.join(params.beams)}_p2p_diffs_{params.area}"
        )

    elev_differences = get_elev_differences(
        params, reference_points, altimetry_points, dem_obj, logger, PREFIX
    )
    logger.info("Got elevation differences len : %d", len(elev_differences))

    # ---------------------- #
    # Convert to lat/lon     #
    # ---------------------- #
    alt_lons, alt_lats = area_obj.xy_to_latlon(
        elev_differences[f"{PREFIX}x"], elev_differences[f"{PREFIX}y"]
    )
    reference_lons, reference_lats = area_obj.xy_to_latlon(
        elev_differences["reference_x"], elev_differences["reference_y"]
    )

    # ------------------------#
    # Output                 #
    # ------------------------#
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

    np.savez(f"{outfile}.npz", **save_data)

    compute_elevation_stats(save_data, prefix=PREFIX, output_file=f"{outfile}_elevation_stats.csv")

    Polarplot(params.area).plot_points(
        {
            "name": "difference_in_height_(dh)",
            "lats": save_data[f"{PREFIX}lats"],
            "lons": save_data[f"{PREFIX}lons"],
            "vals": save_data["dh"],
        },
        output_dir=month_outdir,
    )

    elev_dh_histograms(save_data["dh"], f"{outfile}_dh_histogram.png", bins=params.bins)
    elevation_dh_cumulative_dist(save_data["dh"], f"{outfile}_dh_cumulative_distribution.png")

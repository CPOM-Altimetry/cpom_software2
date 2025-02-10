"""
Tool to compare a selected month of ESA mission altimetry L2 elevations to
ICESat-2 ATL-06 elevations.
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from netCDF4 import Dataset  # pylint: disable=E0611
from scipy.spatial import cKDTree

from cpom.areas.areas import (
    Area,
    list_all_area_definition_names_only,
)


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


def parse_args() -> argparse.ArgumentParser:
    """Declare command line arguments and defaut values
    Returns:
        argparse.ArgumentParser: Parsed input arguments
    """

    # Initiate the command line parser
    parser = argparse.ArgumentParser(description="Retrieve command line arguments")

    # Add long and short command line arguments
    parser.add_argument("--altim_dir", "-ad", help="Directory path", required=True)
    parser.add_argument("--is2_dir", "-id", help="", required=True)

    # parser.add_argument("--mission", "-m", help="altimetry mission: cs2, s3a, s3b", required=True)
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
    parser.add_argument("--outdir", "-o", help="output directory path for results", required=True)
    parser.add_argument(
        "--modes",
        "-mo",
        default="all",
        choices={"all", "lrm", "sin", "sar"},
        nargs="+",
        help="[optional, default=all], limit to specific instrument mode(s). \
                                Comma separated list of: lrm,sin,sar or all",
    )
    parser.add_argument(
        "--beams",
        "-b",
        default="gt2r",
        choices={"gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"},
        nargs="+",
        help="[optional, default=gt2r ] IS2 beams to use. \
                        Comma separated list of: gt1l,gt1r,gt2l,gt2r,gt3l,gt3r",
    )
    parser.add_argument(
        "--radius",
        "-r",
        type=float,
        default=20.0,
        help="[optional] search radius in m, default is 20",
    )
    parser.add_argument(
        "--maxdiff",
        "-md",
        type=float,
        default=100.0,
        help="[optional] maximum allowed difference in m between IS2 and \
        altimeter points, default is 100.0 m. Differences > are not saved.",
    )
    parser.add_argument(
        "--logdir",
        "-ld",
        help="[optional] override default path of log directory",
    )
    parser.add_argument(
        "--add_vars", "-add", help="[optional] Add variables for output. e.g. uncertainty,retracker"
    )
    parser.add_argument("--chunksize", "-cs", type=int, default=5, help="")
    parser.add_argument("--max_workers", "-me", type=int, default=1, help="")

    args = parser.parse_args()

    if args.area not in list_all_area_definition_names_only():
        sys.exit(f"{args.area} not a valid cpom area name")

    args.logdir = args.logdir or args.outdir
    # if args.altdir and not Path(args.altdir).is_dir():
    #     args.altdir = None

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
            "lon": "lat_poca_20_ku",
            "elevation": "height_3_20_ku",
        },
        "CS_OFFL_SIR_SIN": {
            "lat_nadir": "lat_20_ku",
            "lon_nadir": "lon_20_ku",
            "lat": "lat_poca_20_ku",
            "lon": "lat_poca_20_ku",
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
    }

    # Check basename for a matching pattern
    for key, value in variable_map.items():
        if key in basename:
            return value
    return None


def find_nc_and_h5_files(
    directory: str,
    recursive: bool,
    include_string: str | None,
    filetype: str = "nc",
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
    extension = "*.h5" if filetype == "h5" else "*.nc"

    files = list(Path(directory).rglob(extension) if recursive else Path(directory).glob(extension))
    if include_string:
        files = [str(file) for file in files if include_string in file.name]
    else:
        files = [str(file) for file in files]

    return files


# def get_cryotempo_filters(nc, args, mask):
#     """Check Cryotempo data is in a valid mode."""
#     if args.modes == "all":
#         return mask

#     mode_map = {"lrm": 1, "sar": 2, "sin": 3}
#     valid_modes = {mode_map[mode] for mode in args.modes if mode in mode_map}

#     instrument_mode = get_variable(nc, "instrument_mode")[mask]
#     valid_mask = np.isin(instrument_mode, list(valid_modes))

#     return mask[valid_mask] if valid_mask.any() else mask


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
            lon_nadir = get_variable(nc, config["lon_nadir_name"])
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

                if not x.size:
                    return None

                idx, _ = self.area.inside_mask(x, y)
                if not idx.size:
                    return None

                elev = get_variable(nc, config["elevation"])[idx]
                x, y, lats, lons = x[idx], y[idx], lats[idx], lons[idx]

                if len(x) != len(y) != len(elev):
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
                    # self.log.info("CHECK THE RIGHT BEAM %s", beam)
                    config = {
                        key: path.format(beam=beam)
                        for key, path in get_default_variables(filename).items()
                    }
                    # self.log.info("CHECK THE config  %s", config)
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
                        if len(x) != len(y) != len(elevation):
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


def get_elev_differences(
    args: argparse.Namespace, is2_points: np.ndarray, altimeter_points: np.ndarray
) -> dict:
    """
    Calculate the elevation differences between IS2 points and altimeter points
    within a specified search radius. Uses KD-tree to find is2 points within a
    fixed distance from each altimeter point. Filters to elevation points
    less than the maximum elevation difference.

    Args:
        args (argparse.Namespace): Configuration parameters
        is2_points (np.ndarray): is2 array containing x,y,h
        altimeter_points (np.ndarray): altimetry array containing x,y,h

    Returns:
        dict: Ouputted dh between altimetry and is2, plus associated variable data.
    """

    is2_tree = cKDTree(np.c_[is2_points["x"], is2_points["y"]])
    add_vars = {
        var: []
        for var in [val for val in list(altimeter_points.dtype.names) if val not in {"x", "y", "h"}]
    }
    results = {
        "dh": [],
        "x": [],
        "y": [],
        "h": [],
        "is2_x": [],
        "is2_y": [],
        "is2_h": [],
        **{var: [] for var in add_vars},
    }

    for altimeter_point in altimeter_points:
        query_point = (altimeter_point["x"], altimeter_point["y"])

        # Find IS2 points within search_radius
        indices = is2_tree.query_ball_point(query_point, args.radius)
        if len(indices) > 0:
            for idx in indices:
                is2_point = is2_points[idx]
                # Check elevation difference
                dh = altimeter_point["h"] - is2_point["h"]
                if np.abs(dh) < args.maxdiff:
                    results["dh"].append(dh)
                    results["x"].append(altimeter_point["x"])
                    results["y"].append(altimeter_point["y"])
                    results["h"].append(altimeter_point["h"])
                    results["is2_x"].append(altimeter_point["x"])
                    results["is2_y"].append(altimeter_point["y"])
                    results["is2_h"].append(altimeter_point["h"])
                    for var in add_vars:
                        results[var].append(altimeter_point[var])
    return results

if __name__ == "__main__":
    params = parse_args()

    logfile = params.logdir + f"/{params.month:02d}{params.year:4d}"

    logger = setup_logging(
        log_file_info=logfile + ".info.log",
        log_file_error=logfile + ".errors.log",
    )
    logger.info("Start processing with arguments %s", params)

    area_obj = Area(params.area)
    year = params.year
    month = str(params.month).zfill(2)

    # Create or clear output directory
    month_outdir = params.outdir + f"/{year}/{month}"
    if not os.path.isdir(month_outdir):
        print("Creating output directory for results ", month_outdir)
    try:
        os.makedirs(month_outdir)
    except OSError as e:
        logger.error("%s %s", e.filename, e.strerror)
        sys.exit()

    if not Path(month_outdir).is_dir():
        logger.error("month_outdir %s directory. Please create the directory first", month_outdir)
        sys.exit()

    altimetry_files = []
    BASE_PATH = params.altim_dir.split(",")
    for basepath in BASE_PATH:
        if Path(f"{basepath}/{year}/{month}").is_dir():
            BASE_PATH = f"{basepath}/{year}/{month}"

        altim_files = find_nc_and_h5_files(
            directory=BASE_PATH,
            recursive=True,
            include_string=f"_{year}{month}",
            filetype="nc",
        )
        altimetry_files.extend(altim_files)

    logger.info("Loaded %d altimetry data files", len(altimetry_files))

    if Path(f"{params.is2_dir}/{year}/{month}").is_dir():
        params.is2_dir = f"{params.is2_dir}/{year}/{month}"

    is2_files = find_nc_and_h5_files(
        directory=params.is2_dir,
        recursive=True,
        include_string=f"_{year}{month}",
        filetype="h5",
    )

    logger.info("Loaded %d is2-atl06 data files", len(is2_files))

    processor = ProcessData(params, area_obj, logger)
    with ProcessPoolExecutor(max_workers=params.max_workers) as executor:
        altim_results = list(
            executor.map(
                processor.process_file_altimetry, altimetry_files, chunksize=params.chunksize
            )
        )
    valid_altim_results = [result for result in altim_results if result is not None]
    altimetry_points = np.concatenate(valid_altim_results)

    logger.info("Loaded altimetry data points, len : %d", len(altimetry_points))

    with ProcessPoolExecutor(max_workers=params.max_workers) as executor:
        is2_results = list(
            executor.map(processor.process_file_is2, is2_files, chunksize=params.chunksize)
        )
    valid_is2_results = [result for result in is2_results if result is not None]
    laser_is2_points = np.concatenate(valid_is2_results)

    logger.info("Loaded is2 data points, len : %d", len(laser_is2_points))

    elev_differences = get_elev_differences(params, laser_is2_points, altimetry_points)

    logger.info("Got elevation differences len : %d", elev_differences)

    alt_lats, alt_lons = area_obj.xy_to_latlon(elev_differences["x"], elev_differences["y"])
    is2_lats, is2_lons = area_obj.xy_to_latlon(elev_differences["is2_x"], elev_differences["is2_y"])

    BEAMS_STR = "".join(params.beams)
    MISSION = str(
        [
            i
            for i in Path(altimetry_points[0]).parts
            if i in {"CRY", "S3A", "S3B", "ERS2", "ERS1", "ENV"}
        ][0]
    )

    outfile = params.outdir + f"/{MISSION}_minus_is2_{BEAMS_STR}_p2p_diffs_{params.area}.npz"

    logger.info("Saving month data to %s", outfile)
    np.savez(
        outfile,
        dh=elev_differences["dh"],
        lons=alt_lons,
        lats=alt_lats,
        x=elev_differences["x"],
        y=elev_differences["y"],
        h=elev_differences["h"],
        is2_lons=is2_lons,
        is2_lats=is2_lats,
        is2_x=elev_differences["is2_x"],
        is2_y=elev_differences["is2_y"],
        is2_h=elev_differences["is2_h"],
    )

    # print(time.time())

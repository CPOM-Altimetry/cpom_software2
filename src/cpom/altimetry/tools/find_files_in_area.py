#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cpom.altimetry.tools.find_files_in_area.py

tool to find files (containing lat/lon locations) that contain locations within an area's mask
"""

import argparse
import glob
import os
import sys
from typing import List

import numpy as np
from netCDF4 import Dataset, Variable  # pylint: disable=E0611

from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area, list_all_area_definition_names

# pylint: disable=too-many-return-statements

# Colour constants for highlighting terminal output text
RED = "\033[0;31m"  # pylint: disable=invalid-name
BLUE = "\033[0;34m"  # pylint: disable=invalid-name
BLACK_BOLD = "\033[1;30m"  # pylint: disable=invalid-name
ORANGE = "\033[38;5;208m"  # pylint: disable=invalid-name
NC = "\033[0m"  # No Color, pylint: disable=invalid-name


def find_nc_files(
    directory: str, recursive: bool, max_files: int | None, include_string: str | None
) -> List[str]:
    """
    Find .nc or .NC files in the specified directory.

    Args:
        directory (str): The directory to search for .nc files.
        recursive (bool): If True, search recursively in subdirectories.
        max_files (int|None): if not None, limit number of files read to this number

    Returns:
        List[str]: A list of found .nc or .NC files with their full paths.
    """
    files = []
    if recursive:
        for root, _, _ in os.walk(directory):
            files.extend(glob.glob(os.path.join(root, "*.nc")))
            files.extend(glob.glob(os.path.join(root, "*.NC")))
    else:
        files = glob.glob(os.path.join(directory, "*.nc"))
        if len(files) < 1:
            files = glob.glob(os.path.join(directory, "*.NC"))

    if max_files is not None:
        if len(files) > max_files:
            files = files[:max_files]

    if include_string:
        files = [file for file in files if include_string in os.path.basename(file)]

    return files


def get_default_latlon_names(filename: str) -> tuple[str, str]:
    """Get a default latitude and longitude names to plot from filename

    Args:
        filename (str): path of NetCDF file

    Returns:
        (str,str) : latname, lonname
    """

    basename = os.path.basename(filename)

    # E2,E1 ENVISAT FDGR4ALT Land Ice
    if "F4A_ALT_TDP_LI" in basename:
        return "expert/ice_sheet_lat_poca", "expert/ice_sheet_lon_poca"

    # S3 Thematic
    if basename in ("enhanced_measurement.nc", "standard_measurement.nc"):
        return "lat_20_ku", "lon_20_ku"

    # CS2 L1b
    if "CS_OFFL_SIR_SIN_1B" in basename[: len("CS_OFFL_SIR_SIN_1B")]:
        # CS2 L1b SIN
        return "lat_20_ku", "lon_20_ku"
    if "CS_LTA__SIR_SIN_1B" in basename[: len("CS_LTA__SIR_SIN_1B")]:
        # CS2 L1b SIN
        return "lat_20_ku", "lon_20_ku"
    if "CS_OFFL_SIR_LRM_1B" in basename[: len("CS_OFFL_SIR_LRM_1B")]:
        # CS2 L1b LRM
        return "lat_20_ku", "lon_20_ku"
    if "CS_LTA__SIR_LRM_1B" in basename[: len("CS_LTA__SIR_LRM_1B")]:
        # CS2 L1b LRM
        return "lat_20_ku", "lon_20_ku"
    if "CS_OFFL_SIR_SAR_1B" in basename[: len("CS_OFFL_SIR_SAR_1B")]:
        # CS2 L1b SAR
        return "lat_20_ku", "lon_20_ku"
    if "CS_LTA__SIR_SAR_1B" in basename[: len("CS_LTA__SIR_SAR_1B")]:
        # CS2 L1b SAR
        return "lat_20_ku", "lon_20_ku"
    # CryoTEMPO Products
    if "CS_OFFL_SIR_TDP_LI" in basename[: len("CS_OFFL_SIR_TDP_LI")]:
        # CRYO_TEMPO Land Ice file
        return ("latitude", "longitude")
    if "CS_OFFL_SIR_TDP_SI" in basename[: len("CS_OFFL_SIR_TDP_SI")]:
        # CRYO_TEMPO Sea Ice file
        return ("latitude", "longitude")
    if "CS_OFFL_SIR_TDP_PO" in basename[: len("CS_OFFL_SIR_TDP_PO")]:
        # CRYO_TEMPO Polar Oceans file
        return ("latitude", "longitude")
    if "CS_OFFL_SIR_TDP_SS" in basename[: len("CS_OFFL_SIR_TDP_SS")]:
        # CRYO_TEMPO Polar Oceans file
        return ("latitude", "longitude")
    if "CS_NRT__SIR_TDP_PO" in basename[: len("CS_NRT__SIR_TDP_PO")]:
        # CRYO_TEMPO NRT Polar Oceans file
        return ("latitude", "longitude")
    if "CS_NRT__SIR_TDP_SI" in basename[: len("CS_NRT__SIR_TDP_SI")]:
        # CRYO_TEMPO NRT Sea Ice file
        return ("latitude", "longitude")
    # CRISTAL L1b Products
    if "CRA_IR_1B_HR__" in basename[: len("CRA_IR_1B_HR__")]:
        # CRISTAL L1b HR
        return "data/ku/latitude", "data/ku/longitude"
    if "CRA_IR_1B_LMC_" in basename[: len("CRA_IR_1B_LMC_")]:
        # CRISTAL L1b LMC
        return "data_20/ku/latitude", "data_20/ku/longitude"

    print(
        f"{ORANGE}Format of {basename} not recognized - "
        f"so not using defaults for lat/lon parameters{NC}"
    )

    return ("", "")


def get_variable(nc: Dataset, nc_var_path: str) -> Variable:
    """Retrieve variable from NetCDF file, handling groups if necessary.

    This function navigates through groups in a NetCDF file to retrieve the specified variable.

    Args:
        nc (Dataset): The NetCDF dataset object.
        nc_var_path (str): The path to the variable within the NetCDF file,
                        with groups separated by '/'.

    Returns:
        Variable: The retrieved NetCDF variable.

    Raises:
        SystemExit: If the variable or group is not found in the NetCDF file.
    """
    parts = nc_var_path.split("/")
    var = nc
    for part in parts:
        try:
            var = var[part]
        except (KeyError, IndexError):
            sys.exit(f"{RED}NetCDF parameter '{nc_var_path}' not found{NC}")
    return var


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth.

    Parameters:
    lat1, lon1: Latitude and longitude of point 1 in degrees.
    lat2, lon2: Latitude and longitude of point 2 in degrees.

    Returns:
    Distance in meters.
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth's radius in meters (mean radius) = 6371000
    return 6371000 * c


def is_within_distance(
    lats: np.ndarray, lons: np.ndarray, target_lat: float, target_lon: float, distance: float
) -> bool:
    """
    Return True if any of (lats, lons) are within distance from (target_lat, target_lon),
    where distance is in meters, else return False.

    Parameters:
    lats (np.ndarray): Array of latitudes.
    lons (np.ndarray): Array of longitudes.
    target_lat (float): Target latitude.
    target_lon (float): Target longitude.
    distance (float): Distance threshold in km.

    Returns:
    bool: True if any point is within the distance, False otherwise.
    """
    for lat, lon in zip(lats, lons):
        if haversine(lat, lon, target_lat, target_lon) <= (distance * 1000):
            return True

    return False


def main(args):
    """main function of tool

    Returns:
        None
    """

    # ----------------------------------------------------------------------
    # Process Command Line Arguments for tool
    # ----------------------------------------------------------------------

    # initiate the command line parser
    parser = argparse.ArgumentParser(
        description=(
            "Tool to to find files (containing lat/lon locations) that contain locations "
            "within an area's mask"
        )
    )

    parser.add_argument(
        "--area",
        "-a",
        help=("CPOM area definition name. See --list_areas for a full list"),
        required=False,
    )

    parser.add_argument(
        "--dir",
        "-d",
        help=("[Optional] path of a directory containing netcdf files to search."),
        required=False,
    )

    parser.add_argument(
        "--extent_only",
        "-e",
        help=(
            "[Optional] test area's approx. rectangular extent only. This is quicker than testing "
            "the polygon mask(s), but may result in occasional files that are inside the "
            "rectangular extent but not inside the exact mask"
        ),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--filename_only",
        "-fo",
        help=(
            "[Optional] print only selected file name paths and not any other "
            "information. Useful if creating a list"
        ),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--include",
        "-i",
        help=("[optional] include only files with this string in their filename"),
        required=False,
    )

    parser.add_argument(
        "--latname",
        "-lat",
        help=(
            "name of netcdf latitude parameters with optional "
            "group path. Group paths are separated by /  "
            "Example: --latname latitude , --latname data/ku/latitude , --latname lats1 ."
            "If not used, default values for file type will be used if possible."
        ),
        required=False,
    )

    parser.add_argument(
        "--list_areas",
        "-ls",
        help=("[optional] list allowed area names and exit"),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--lonname",
        "-lon",
        help=(
            "name of netcdf longitude parameter with optional "
            "group path. Group paths are separated by / .  "
            "Example: --lonname longitude , --lonname data/ku/longitude , --lonname lons1"
            "If not used, default values for file type will be used if possible."
        ),
        required=False,
    )

    parser.add_argument(
        "--max_files",
        "-mf",
        help=("[optional] only read the first N input netcdf files (if multiple files input)"),
        required=False,
        type=int,
    )

    parser.add_argument(
        "--plot",
        "-p",
        help=("[optional] plot tracks on map of selected area"),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--radius_search",
        "-rs",
        help=(
            "[optional] search for files within a radius (km) of a target latitude and longitude. "
            "Enter --radius_search latitude longitude radius_km"
        ),
        required=False,
        nargs=3,
        metavar=("LAT", "LON", "RADIUS"),
        type=float,
    )

    parser.add_argument(
        "--not_recursive",
        "-nr",
        help=(
            "[optional] flat directory search (not recursive, which is the default) for input files"
        ),
        required=False,
        action="store_true",
    )

    # read arguments from the command line
    args = parser.parse_args(args)

    if not args.area and not args.radius_search:
        sys.exit("must use one of --area or --radius_search")

    if not args.area and args.plot:
        sys.exit("must use --area with --plot")

    if args.latname and not args.lonname:
        sys.exit("both --lonname and --latname must be used together")

    # Print a list of available area definitions
    if args.list_areas:
        area_list = list_all_area_definition_names()
        mystr = f"{BLACK_BOLD}List of Available Area Names from CPOM Area Definitions{NC}"
        print("-" * (len(mystr) - 8))
        print(mystr)
        print("-" * (len(mystr) - 8))
        for area_name in area_list:
            print(f"{area_name}")
        sys.exit(0)

    thisarea = Area(args.area)

    files = find_nc_files(args.dir, not args.not_recursive, args.max_files, args.include)

    file_number_found = 0
    found_file_list = []
    lat_list = []
    lon_list = []
    val_list = []

    if args.radius_search:
        target_lat, target_lon, radius = args.radius_search

    for file in files:

        if args.latname and args.lonname:
            lat_name = args.latname
            lon_name = args.lonname
        else:
            lat_name, lon_name = get_default_latlon_names(file)

        with Dataset(file) as nc:

            lats = get_variable(nc, lat_name)[:].data
            lons = get_variable(nc, lon_name)[:].data % 360.0

            n_vals = lats.size

            if args.radius_search:
                if is_within_distance(lats, lons, target_lat, target_lon, radius):
                    file_number_found += 1
                    if args.filename_only:
                        print(f"{file}")
                    else:
                        print(f"{file_number_found}: {file} : has points within radius")

                    if args.plot:
                        if file_number_found <= 20:
                            found_file_list.append(file)
                            lat_list.extend(lats)
                            lon_list.extend(lons)
                            val_list.extend(np.full(len(lats), file_number_found))
            else:
                if thisarea.specify_by_bounding_lat:
                    (lats_inside, lons_inside, indices_inside, n_inside) = (
                        thisarea.inside_latlon_bounds(lats, lons)
                    )
                    x_inside, y_inside = thisarea.latlon_to_xy(lats_inside, lons_inside)
                else:

                    (lats_inside, lons_inside, x_inside, y_inside, indices_inside, n_inside) = (
                        thisarea.inside_xy_extent(lats, lons)
                    )

                if n_inside > 0 and args.extent_only:
                    file_number_found += 1
                    percent_inside = 100.0 * n_inside / n_vals
                    if args.filename_only:
                        print(f"{file}")
                    else:
                        print(
                            f"{file_number_found}: {file} : {n_inside} points "
                            f"({percent_inside:.2f}%) inside {thisarea.long_name}"
                        )
                    if args.plot:
                        if file_number_found <= 20:
                            found_file_list.append(file)
                            lat_list.extend(lats_inside)
                            lon_list.extend(lons_inside)
                            val_list.extend(np.full(len(lats_inside), file_number_found))
                elif n_inside > 0:
                    (indices_inside, n_inside) = thisarea.inside_mask(x_inside, y_inside)
                    if n_inside > 0:
                        file_number_found += 1
                        percent_inside = 100.0 * n_inside / n_vals
                        if args.filename_only:
                            print(f"{file}")
                        else:
                            print(
                                f"{file_number_found}: {file} : {n_inside} points  "
                                f"({percent_inside:.2f}%) inside {thisarea.long_name}"
                            )

                        if args.plot:
                            found_file_list.append(file)
                            if file_number_found <= 20:
                                lat_list.extend(lats_inside[indices_inside])
                                lon_list.extend(lons_inside[indices_inside])
                                val_list.extend(np.full(len(indices_inside), file_number_found))

    if args.plot and file_number_found > 0:
        data_set = {
            "name": "track number",
            "lats": np.array(lat_list),
            "lons": np.array(lon_list),
            "vals": np.array(val_list),
            "flag_values": list(range(1, file_number_found + 1)),
            "flag_names": [str(s) for s in list(range(1, file_number_found + 1))],
        }
        Polarplot(args.area, {"show_bad_data_map": False}).plot_points(
            data_set,
        )

    if file_number_found == 0 and not args.filename_only:
        print("No files found matching search criteria")


if __name__ == "__main__":
    main(sys.argv[1:])

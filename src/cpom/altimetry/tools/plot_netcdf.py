#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot variables from NetCDF files using Polarplot"""

import argparse
import logging
import sys

import numpy as np
from netCDF4 import Dataset, Variable  # pylint: disable=E0611

from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area, list_all_area_definition_names

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements


def setup_logging():
    """Setup logging configuration"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def main():
    """main function of tool

    Returns:
        None
    """
    RED = "\033[0;31m"  # pylint: disable=invalid-name
    # BLUE = "\033[0;34m"
    # YELLOW = "\033[1;33m"
    # BLACK_BOLD = "\033[1;30m"
    NC = "\033[0m"  # No Color, pylint: disable=invalid-name

    log = setup_logging()  # Initialize logging configuration

    # ----------------------------------------------------------------------
    # Process Command Line Arguments for tool
    # ----------------------------------------------------------------------

    # initiate the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        "-f",
        help=("[Optional] path of a single input L1b file"),
        required=True,
    )

    parser.add_argument(
        "--params",
        "-p",
        help=(
            "comma separated list of netcdf parameters with optional group path "
            "(e.g., 'group/subgroup/parameter'). "
            "Example: -p elevation , -p elevation1,elevation2 , -p data/ku/height"
        ),
        required=True,
    )

    parser.add_argument(
        "--area",
        "-a",
        help=("area name"),
        required=True,
    )

    parser.add_argument(
        "--latname",
        "-lat",
        help=(
            "comma separated list of names of netcdf latitude parameters with optional "
            "group path. Group paths are separated by / . If you have multiple data sets separate "
            "by commas. "
            "Example: --latname latitude , --latname data/ku/latitude , --latname lats1,lats2"
        ),
        required=True,
    )

    parser.add_argument(
        "--lonname",
        "-lon",
        help=(
            "comma separated list of names of netcdf longitude parameters with optional "
            "group path. Group paths are separated by / . If you have multiple data sets separate "
            "by commas. "
            "Example: --lonname longitude , --lonname data/ku/longitude , --lonname lons1,lons2"
        ),
        required=True,
    )

    parser.add_argument(
        "--out_file",
        "-o",
        help=("/some/path/filename.png :save plot as this .png output file name"),
        required=False,
    )

    parser.add_argument(
        "--out_dir",
        "-d",
        help=(
            "/some/path :save plot to this directory path, using a standard file name"
            "of plot_<param>_<area>.png, where param is the first "
            "--params arg (with any / replaced by _)"
        ),
        required=False,
    )

    parser.add_argument(
        "--map_only",
        "-m",
        help=("plot map only, not histograms"),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--list_areas",
        "-l",
        help=("list allowed area names and exit"),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--simulate",
        "-s",
        help=(
            "step size in degrees. This option simulates a mesh of data points in area. "
            "Used for area plot testing, not for plotting netcdf values"
        ),
        required=False,
        type=float,
    )

    # read arguments from the command line
    args = parser.parse_args()

    if args.out_file and args.out_dir:
        log.error("Only one of --out_file and --out_dir allowed")
        sys.exit(1)

    if args.list_areas:
        area_list = list_all_area_definition_names()
        for area_name in area_list:
            print(f"{area_name}")
        sys.exit(0)

    # Open the NetCDF file
    try:
        nc = Dataset(args.file)
    except IOError:
        sys.exit(f"{RED}Can not open {args.file}{NC}")

    def get_variable(nc: Dataset, var_path: str) -> Variable:
        """Retrieve variable from NetCDF file, handling groups if necessary.

        This function navigates through groups in a NetCDF file to retrieve the specified variable.

        Args:
            nc (Dataset): The NetCDF dataset object.
            var_path (str): The path to the variable within the NetCDF file,
                            with groups separated by '/'.

        Returns:
            Variable: The retrieved NetCDF variable.

        Raises:
            SystemExit: If the variable or group is not found in the NetCDF file.
        """
        parts = var_path.split("/")
        var = nc
        for part in parts:
            try:
                var = var[part]
            except (KeyError, IndexError):
                sys.exit(f"{RED}NetCDF parameter '{var_path}' not found{NC}")
        return var

    params = args.params.split(",")
    latnames = args.latname.split(",")
    lonnames = args.lonname.split(",")

    if len(params) != len(latnames) or len(latnames) != len(lonnames):
        sys.exit(f"{RED}The number of parameters, latitudes, and longitudes must be equal{NC}")

    num_data_sets = len(params)
    print(f"Number of datasets: {num_data_sets}")

    datasets = []

    for i in range(num_data_sets):
        vals = get_variable(nc, params[i])[:].data
        lats = get_variable(nc, latnames[i])[:].data
        lons = get_variable(nc, lonnames[i])[:].data % 360.0

        if args.simulate:
            thisarea = Area(args.area)
            # Define the step size for the mesh grid
            # Smaller step size means more data points and a finer mesh
            lon_step = args.simulate
            lat_step = args.simulate

            # Generate the grid of points
            lon_values = np.arange(thisarea.minlon, thisarea.maxlon + lon_step, lon_step)
            lat_values = np.arange(thisarea.minlat, thisarea.maxlat + lat_step, lat_step)

            # Create the mesh grid
            lons, lats = np.meshgrid(lon_values, lat_values)
            vals = lats

        dataset = {
            "name": params[i],
            "units": "m",
            "lats": lats,
            "lons": lons,
            "vals": vals,
            # "flag_colors": ["b"],
            # "flag_names": [
            #     "1",
            # ],
            # "flag_values": [
            #     1,
            # ],
        }

        datasets.append(dataset)

    try:
        _ = Area(args.area)
    except ImportError:
        sys.exit(
            f"{RED}Area name {args.area} not found in list of area definitions "
            f"src/cpom/areas/definitions{NC}"
        )

    Polarplot(args.area).plot_points(
        *datasets,
        output_file=args.out_file,
        output_dir=args.out_dir,
        map_only=args.map_only,
    )


if __name__ == "__main__":
    main()

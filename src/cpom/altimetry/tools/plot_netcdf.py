#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot variables from NetCDF files using Polarplot"""

import argparse
import glob
import logging
import sys

import numpy as np
from netCDF4 import Dataset, Variable  # pylint: disable=E0611

from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area, list_all_area_definition_names

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements

RED = "\033[0;31m"  # pylint: disable=invalid-name
BLUE = "\033[0;34m"
# YELLOW = "\033[1;33m"
BLACK_BOLD = "\033[1;30m"
ORANGE = "\033[38;5;208m"
NC = "\033[0m"  # No Color, pylint: disable=invalid-name

# pylint: disable=too-many-branches


def setup_logging():
    """Setup logging configuration"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


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


def main():
    """main function of tool

    Returns:
        None
    """

    log = setup_logging()  # Initialize logging configuration

    # ----------------------------------------------------------------------
    # Process Command Line Arguments for tool
    # ----------------------------------------------------------------------

    # initiate the command line parser
    parser = argparse.ArgumentParser(
        description=(
            "Tool to plot netcdf parameter(s) from one or more netcdf files"
            " on cryosphere maps using the CPOM area"
            " definitions and basemaps to define the maps"
        )
    )

    parser.add_argument(
        "--area",
        "-a",
        help=("CPOM area definition name. See --list_areas for a full list"),
        required=False,
    )

    parser.add_argument(
        "--cmap",
        "-cm",
        help=("colourmap name to to use. Default is RdYlBu_r"),
        required=False,
        default="RdYlBu_r",
    )

    parser.add_argument(
        "--cmap_over",
        "-cmo",
        help=("colourmap over color name to to use. Default is #A85754"),
        required=False,
        default="#A85754",
    )

    parser.add_argument(
        "--cmap_under",
        "-cmu",
        help=("colourmap under color name to to use. Default is #3E4371"),
        required=False,
        default="#3E4371",
    )

    parser.add_argument(
        "--cmap_extend",
        "-cme",
        help=("colourmap extend policy. One of 'neither','min', 'max','both'"),
        required=False,
        default="both",
        choices=["neither", "min", "max", "both"],
    )

    parser.add_argument(
        "--dir",
        "-d",
        help=(
            "[Optional] path of a directory containing netcdf files to plot."
            "All files are plotted within this directory."
        ),
        required=False,
    )

    parser.add_argument(
        "--file",
        "-f",
        help=("[Optional] path of a single input netcdf file"),
        required=False,
    )

    parser.add_argument(
        "--fill_value",
        "-fv",
        help=(
            "[Optional] assign fill value (value to be treated as bad values)."
            "If multiple parameters use comma separated list."
        ),
        required=False,
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
        required=False,
    )

    parser.add_argument(
        "--list_areas",
        "-l",
        help=("list allowed area names and exit"),
        required=False,
        action="store_true",
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
        required=False,
    )

    parser.add_argument(
        "--out_dir",
        "-od",
        help=(
            "/some/path :save plot to this directory path, using a standard file name"
            "of plot_<param>_<area>.png, where param is the first "
            "--params arg (with any / replaced by _)"
        ),
        required=False,
    )

    parser.add_argument(
        "--out_file",
        "-o",
        help=("/some/path/filename.png :save plot as this .png output file name"),
        required=False,
    )

    parser.add_argument(
        "--params",
        "-p",
        help=(
            "comma separated list of netcdf parameters with optional group path "
            "(e.g., 'group/subgroup/parameter'). "
            "Example: -p elevation , -p elevation1,elevation2 , -p data/ku/height"
        ),
        required=False,
    )

    parser.add_argument(
        "--point_size_factor",
        "-ps",
        help=(
            "point size scale factor (default is 1.0). Use this to scale the "
            "plotted point size up or down. Needs some experimentation as to the correct value "
            "which may be 0.01 or 100. It is more of a log scale. For plotting multiple "
            "parameters different sizes you can use a comma separated list of scale factors."
        ),
        required=False,
        default="1.0",
    )

    parser.add_argument(
        "--map_only",
        "-m",
        help=("plot map only, not histograms"),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--max_plot_range",
        "-pr2",
        help=(
            "maximum plot range value."
            " Default is the maximum value of the parameter plotted."
            " For multiple parameters use comma separated list."
        ),
        required=False,
        default="",
    )

    parser.add_argument(
        "--min_plot_range",
        "-pr1",
        help=(
            "minimum plot range value."
            " Default is the minimum value of the parameter plotted."
            " For multiple parameters use comma separated list."
        ),
        required=False,
        default="",
    )

    parser.add_argument(
        "--step",
        "-s",
        help=(
            "step size in degrees. This option simulates a mesh of data points in area. "
            "Used for area plot testing, not for plotting netcdf values. "
            "ie  -a antarctica -s 5"
        ),
        required=False,
        type=float,
    )

    parser.add_argument(
        "--valid_min",
        "-vmin",
        help=(
            "minimum valid value to allow. Default is actual minimum. Comma"
            "separated list of min_vals if multiple parameters used. "
            "Values lower than this are flagged as bad."
        ),
        required=False,
    )
    parser.add_argument(
        "--valid_max",
        "-vmax",
        help=(
            "maximum valid value to allow. Default is actual maximum. Comma"
            "separated list of max_vals if multiple parameters used. "
            "Values higher than this are flagged as bad."
        ),
        required=False,
    )

    # read arguments from the command line
    args = parser.parse_args()

    # Print a list of available area definitions
    if args.list_areas:
        area_list = list_all_area_definition_names()
        mystr = f"{BLACK_BOLD}List of Available Area Names from CPOM Area Definitions{NC}"
        print("-" * (len(mystr) - 8))
        print(str)
        print("-" * (len(mystr) - 8))
        for area_name in area_list:
            print(f"{area_name}")
        sys.exit(0)

    if not args.area:
        sys.exit(
            f"{RED}--area area_name{NC} missing. Use --list_areas to show available area names"
        )

    try:
        thisarea = Area(args.area)
    except ImportError:
        sys.exit(
            f"{RED}Area name {BLUE}{args.area}{RED} not found in list of area definitions "
            f"src/cpom/areas/definitions{NC}."
            " Use --listareas to show available areas"
        )

    if args.out_file and args.out_dir:
        log.error("Only one of --out_file and --out_dir allowed")
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Get datasets to plot
    # ---------------------------------------------------------------------------

    datasets = []

    if not args.file and not args.dir:
        if args.step:
            # Generate some simulated data
            lon_step = args.step
            lat_step = args.step

            # Generate the grid of points
            lon_values = np.arange(thisarea.minlon, thisarea.maxlon + lon_step, lon_step)
            lat_values = np.arange(thisarea.minlat, thisarea.maxlat + lat_step, lat_step)

            # Create the mesh grid
            lons, lats = np.meshgrid(lon_values, lat_values)
            vals = lats
        else:
            lats = []
            lons = []
            vals = []

        dataset = {
            "name": "simulated" if len(lats) > 0 else "None",
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
    elif args.file or args.dir:
        if args.file:
            files = [args.file]
        else:
            files = glob.glob(f"{args.dir}/*.nc")
            if len(files) < 1:
                files = glob.glob(f"{args.dir}/*.NC")

        params = args.params.split(",")
        num_data_sets = len(params)
        print(f"Number of datasets: {num_data_sets}")

        error_str = ""
        latnames = args.latname.split(",")
        if len(latnames) != num_data_sets:
            latnames = [latnames[0]] * num_data_sets
            error_str += "latname "
        lonnames = args.lonname.split(",")
        if len(lonnames) != num_data_sets:
            lonnames = [lonnames[0]] * num_data_sets
            error_str += "lonname "
        plot_size_scale_factor = args.point_size_factor.split(",")
        if len(plot_size_scale_factor) != num_data_sets:
            plot_size_scale_factor = [plot_size_scale_factor[0]] * num_data_sets
            error_str += "point_size_factor "
        min_plot_range = args.min_plot_range.split(",")
        if len(min_plot_range) != num_data_sets:
            min_plot_range = [min_plot_range[0]] * num_data_sets
            error_str += "min_plot_range "
        max_plot_range = args.max_plot_range.split(",")
        if len(max_plot_range) != num_data_sets:
            max_plot_range = [max_plot_range[0]] * num_data_sets
            error_str += "max_plot_range "
        cmap_names = args.cmap.split(",")
        if len(cmap_names) != num_data_sets:
            cmap_names = [cmap_names[0]] * num_data_sets
            error_str += "cmap "
        cmap_over = args.cmap_over.split(",")
        if len(cmap_over) != num_data_sets:
            cmap_over = [cmap_over[0]] * num_data_sets
            error_str += "cmap_over "
        cmap_under = args.cmap_under.split(",")
        if len(cmap_under) != num_data_sets:
            cmap_under = [cmap_under[0]] * num_data_sets
            error_str += "cmap_under "
        cmap_extend = args.cmap_extend.split(",")
        if len(cmap_extend) != num_data_sets:
            cmap_extend = [cmap_extend[0]] * num_data_sets
            error_str += "cmap_extend "
        if args.valid_min:
            valid_min = args.valid_min.split(",")
            if len(valid_min) != num_data_sets:
                valid_min = [valid_min[0]] * num_data_sets
                error_str += "valid_min "
        if args.valid_max:
            valid_max = args.valid_max.split(",")
            if len(valid_max) != num_data_sets:
                valid_max = [valid_max[0]] * num_data_sets
                error_str += "valid_max "

        if args.fill_value:
            fill_value = args.fill_value.split(",")
            if len(fill_value) != num_data_sets:
                fill_value = [fill_value[0]] * num_data_sets
                error_str += "fill_value "

        if len(error_str) > 0:
            print(
                f"{ORANGE} WARNING: If you have {num_data_sets} commas separated --params args,"
                f" you can have an equal number of comma separated\n"
                f" {error_str} arguments{NC}"
            )
        for i in range(num_data_sets):
            ds = {
                "name": params[i],
                "units": "m",
                "lats": [],
                "lons": [],
                "vals": [],
                "plot_size_scale_factor": float(plot_size_scale_factor[i]),
                "cmap_name": cmap_names[i],
                "cmap_over_color": cmap_over[i],
                "cmap_under_color": cmap_under[i],
                "cmap_extend": cmap_extend[i],
            }
            if min_plot_range[i] != "":
                ds["min_plot_range"] = float(min_plot_range[i])
            if max_plot_range[i] != "":
                ds["max_plot_range"] = float(max_plot_range[i])
            if args.valid_min and args.valid_max is None:
                ds["valid_range"] = [float(valid_min[i]), None]
            if args.valid_max and args.valid_min is None:
                ds["valid_range"] = [None, float(valid_max[i])]
            if args.valid_max and args.valid_min:
                ds["valid_range"] = [float(valid_min[i]), float(valid_max[i])]
            if args.fill_value:
                ds["fill_value"] = float(fill_value[i])

            datasets.append(ds)

            # "flag_colors": ["b"],
            # "flag_names": [
            #     "1",
            # ],
            # "flag_values": [
            #     1,
            # ],

        for filename in files:
            # Open the NetCDF file
            try:
                with Dataset(filename) as nc:
                    for i in range(num_data_sets):
                        vals = get_variable(nc, params[i])[:].data
                        lats = get_variable(nc, latnames[i])[:].data
                        lons = get_variable(nc, lonnames[i])[:].data % 360.0

                        datasets[i]["lats"].extend(lats)
                        datasets[i]["lons"].extend(lons)
                        datasets[i]["vals"].extend(vals)

            except IOError:
                sys.exit(f"{RED}Can not open {filename}{NC}")
    else:
        sys.exit('feature Not implemented yet"')

    Polarplot(args.area).plot_points(
        *datasets,
        output_file=args.out_file,
        output_dir=args.out_dir,
        map_only=args.map_only,
    )


if __name__ == "__main__":
    main()

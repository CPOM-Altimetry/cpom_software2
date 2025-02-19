#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool to plot variables (including flags) from NetCDF file(s) on a selectable cryosphere map

For full list of command line args:

`plot_map.py --help`

Most settings are configurable from the tool using the command line arguments, although the tool
will try and automatically identify default parameters and select an area to plot 
(from contained lat/lon values) from most altimetry formats. If not you can choose these from the
command line.

## Examples

List all available area definitions (ie the areas you can select to plot your data on):

`plot_map.py --list_areas`

Plot a parameter **elevation** from a CryTEMPO netcdf file, and display in area 
definition **antarctica_hs_is** which is an Antarctic map with hillshading and a 
grounded icesheet mask:

`plot_map.py -f  /cpdata/SATS/RA/CRY/Cryo-TEMPO/BASELINE-B/001/LAND_ICE/\
ANTARC/2010/07/CS_OFFL_SIR_TDP_LI_ANTARC_20100717T141004_20100717T141229_01_02644_B001.nc \
-p elevation -a antarctica_hs_is`

![my image](/cpom_software2/images/plot_map_example1.png "my image")

Plot all the netcdf files in the given directory. Automatically select the default
parameter and area to plot.

`plot_map.py -d /cpdata/SATS/RA/CRY/Cryo-TEMPO/BASELINE-B/001/LAND_ICE/ANTARC/2010/07`

![my image](/cpom_software2/images/plot_map_example2.png "my image")

Plot a simulated grid of values at 0.01 deg separation over Lake Vostok, with
point size 1.0 and colormap set to viridis

`plot_map.py -a vostok -s 0.01 -ps 1 --cmap viridis`

![my image](/cpom_software2/images/plot_map_example3.png "my image")

Plot the instrument_mode parameter files in a named directory, and use the flag parameter settings
shown to specify flag values, names and colours:

`plot_map.py -d /cpdata/SATS/RA/CRY/Cryo-TEMPO/BASELINE-B/001/LAND_ICE/ANTARC/2010/07 \
    -p instrument_mode \
    --flag_params 1:LRM:blue/2:SAR:pink/3:SIN:red`

![my image](/cpom_software2/images/plot_map_example4.png "my image")

"""

__all__ = ["main"]

import argparse
import glob
import logging
import os
import re
import sys
from typing import List

import numpy as np
from netCDF4 import Dataset, Variable  # pylint: disable=E0611

from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area, list_all_area_definition_names

# pylint: disable=too-many-branches
# pylint: disable=too-many-return-statements
# pylint: disable=too-many-lines


# Colour constants for highlighting terminal output text
RED = "\033[0;31m"  # pylint: disable=invalid-name
BLUE = "\033[0;34m"  # pylint: disable=invalid-name
BLACK_BOLD = "\033[1;30m"  # pylint: disable=invalid-name
ORANGE = "\033[38;5;208m"  # pylint: disable=invalid-name
NC = "\033[0m"  # No Color, pylint: disable=invalid-name


def setup_logging():
    """Setup logging handlers"""
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


def get_default_param(
    filename: str,
) -> tuple[str, str]:
    """Get a default parameter to plot from known file types

    Supported types:
    Cryo-TEMPO Land Ice: file name starts with CS_OFFL_SIR_TDP_LI
    Cryo-TEMPO Sea Ice: file name starts with CS_OFFL_SIR_TDP_SI


    Args:
        filename (str): path of NetCDF file

    Returns:
        str,str: parameter name or '' if not found,units or '' if not found
    """

    basename = os.path.basename(filename)

    print(f"finding default params for {filename}")

    # E2,E1 ENVISAT FDGR4ALT Land Ice
    if "F4A_ALT_TDP_LI" in basename:
        return "expert/ice_sheet_elevation_ice1_roemer", "m"

    # S3 Thematic
    if basename in ("enhanced_measurement.nc", "standard_measurement.nc"):
        if "SR_2_LAN_LI" in filename:
            return "elevation_ocog_20_ku", "m"
        if "SR_2_LAN_SI" in filename:
            return "freeboard_20_ku", "m"
        return "lat_20_ku", "degs"

    # CS2 L1b SIN
    if "CS_OFFL_SIR_SIN_1B" in basename[: len("CS_OFFL_SIR_SIN_1B")]:
        return "lat_20_ku", "degs"
    if "CS_LTA__SIR_SIN_1B" in basename[: len("CS_LTA__SIR_SIN_1B")]:
        return "lat_20_ku", "degs"
    # CS2 L1b LRM
    if "CS_OFFL_SIR_LRM_1B" in basename[: len("CS_LTA__SIR_LRM_1B")]:
        return "lat_20_ku", "degs"
    if "CS_LTA__SIR_LRM_1B" in basename[: len("CS_LTA__SIR_LRM_1B")]:
        return "lat_20_ku", "degs"
    # CS2 L1b SAR
    if "CS_OFFL_SIR_SAR_1B" in basename[: len("CS_LTA__SIR_SAR_1B")]:
        return "lat_20_ku", "degs"
    if "CS_LTA__SIR_SAR_1B" in basename[: len("CS_LTA__SIR_SAR_1B")]:
        return "lat_20_ku", "degs"

    # CryoTEMPO Products
    if "CS_OFFL_SIR_TDP_LI" in basename[: len("CS_OFFL_SIR_TDP_LI")]:
        # CRYO_TEMPO Land Ice file
        return "elevation", "m"
    if "CS_OFFL_SIR_TDP_SI" in basename[: len("CS_OFFL_SIR_TDP_SI")]:
        # CRYO_TEMPO Sea Ice file
        return "radar_freeboard", "m"
    if "CS_OFFL_SIR_TDP_PO" in basename[: len("CS_OFFL_SIR_TDP_PO")]:
        # CRYO_TEMPO Polar Oceans file
        return "dynamic_ocean_topography", "m"
    if "CS_OFFL_SIR_TDP_SS" in basename[: len("CS_OFFL_SIR_TDP_SS")]:
        # CRYO_TEMPO Summer Sea Ice file
        return "radar_freeboard", "m"
    if "CS_NRT__SIR_TDP_PO" in basename[: len("CS_NRT__SIR_TDP_PO")]:
        # CRYO_TEMPO NRT Polar Oceans
        return "dynamic_ocean_topography", "m"
    if "CS_NRT__SIR_TDP_SI" in basename[: len("CS_NRT__SIR_TDP_SI")]:
        # CRYO_TEMPO NRT Sea Ice
        return "radar_freeboard", "m"
    # CRISTAL L1b Products
    if "CRA_IR_1B_HR__" in basename[: len("CRA_IR_1B_HR__")]:
        # CRISTAL L1b HR
        return "data/ku/tracker_range_calibrated", "m"
    if "CRA_IR_1B_LMC_" in basename[: len("CRA_IR_1B_LMC_")]:
        # CRISTAL L1b LMC
        return "data_20/ku/tracker_range_calibrated", "m"
    if "CRA_IR_1B_LR__" in basename[: len("CRA_IR_1B_LR__")]:
        # CRISTAL L1b LR
        return "data_20/ku/tracker_range_calibrated", "m"

    print(
        f"{ORANGE}Format of {basename} not recognized - "
        f"so not using defaults for parameter name{NC}"
    )

    return "", ""


def get_default_latlon_names(filename: str) -> tuple[str, str]:
    """Get a default latitude and longitude names to plot from filename

    Args:
        filename (str): path of NetCDF file

    Returns:
        (str,str) : latname, lonname or ("","") if not found
    """

    print(f"finding default lat/lon parameters for {filename}")

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
    if "CRA_IR_1B_LR__" in basename[: len("CRA_IR_1B_LR__")]:
        # CRISTAL L1b LR
        return "data_20/ku/latitude", "data_20/ku/longitude"
    print(
        f"{ORANGE}Format of {basename} not recognized - "
        f"so not using defaults for lat/lon parameters{NC}"
    )

    return ("", "")


def get_default_area(lat: float, filename: str) -> str:
    """select a default area definition based on latitude

    Args:
        lat (float): latitude in degs
        filename (str): file name

    Returns:
        str: cpom area definition name
    """
    if lat < -50.0:
        return "antarctica_hs"
    if lat > 50.0:
        basename = os.path.basename(filename)
        if "GREENL" in basename:  # for CryoTEMPO Greenland files
            return "greenland_hs"
        return "arctic_cpy"
    return "global"


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


def main(args):
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
        "--and_flag",
        "-af",
        help=(
            "[optional, str:int], flag_name:flag_value, main plot parameter is "
            "included if flag == value"
        ),
        required=False,
    )

    parser.add_argument(
        "--area",
        "-a",
        help=("CPOM area definition name. See --list_areas for a full list"),
        required=False,
    )

    parser.add_argument(
        "--areadef_file",
        "-df",
        help=(
            "[optional] path of CPOM area definition file. "
            "Not necessary if using standard area definitions in cpom.areas.definitions."
            " Can be used instead of standard area definitions"
        ),
        required=False,
    )

    parser.add_argument(
        "--bool_mask_params",
        "-bm",
        help=(
            "comma separated list of netcdf parameters containing a boolean mask "
            "with optional group path (e.g., 'group/subgroup/parameter'). "
            "Example: -bm ku/original/bool_mask_area -p ku/original/surface_type."
            " The boolean mask is used to subset the parameter(s) set by --params"
        ),
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
        "--dpi",
        "-dpi",
        help=("[Optional, int] set the dpi to use when writing to an image file (def=85)."),
        type=int,
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
        "--flag",
        "-fl",
        help=(
            "[Optional] treat input values as flag data when plotting. "
            "If no other flag options used then, flag values will be chosen "
            "from unique values in data set, and flags will be named with that number. "
            "Use --flag_params instead to set exact flag values/names/colors to use."
        ),
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--flag_params",
        "-fp",
        help=(
            "[Optional] comma separated list per parameter plotted of \n"
            "value:name:color/value:name:color/.....,value:name:color/... \n"
            "where value is the flag value, name is the associated name of this flag, "
            "color is a python color used to plot the flag"
        ),
        required=False,
    )

    parser.add_argument(
        "--hillshade",
        "-hs",
        help=("[optional] apply hillshade to values"),
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
            "comma separated list of names of netcdf latitude parameters with optional "
            "group path. Group paths are separated by / . If you have multiple data sets separate "
            "by commas. "
            "Example: --latname latitude , --latname data/ku/latitude , --latname lats1,lats2"
        ),
        required=False,
    )

    parser.add_argument(
        "--list_areas",
        "-ls",
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
        "--map_only",
        "-m",
        help=("plot map only, not histograms"),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--max_files",
        "-mf",
        help=("only read the first N input netcdf files (if multiple files input)"),
        required=False,
        type=int,
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
        "--not_apply_mask",
        "-nam",
        help=(
            "if set, do not apply area's data mask to filter input locations."
            " Applies to all parameters when multiple parameters used."
        ),
        required=False,
        action="store_true",
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
        default="0.1",
    )

    parser.add_argument(
        "--recursive",
        "-r",
        help=("recursive directory search for input files"),
        required=False,
        action="store_true",
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
        "--units",
        "-u",
        help=(
            "[optional] units of parameters to plot. Used in annotation. Comma"
            "separated list of units if multiple parameters used. "
        ),
        required=False,
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
    args = parser.parse_args(args)

    if args.step:
        if args.step > 20:
            sys.exit(f"command line error: --step {args.step} : value in degs must be less than 20")

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

    if args.out_file and args.out_dir:
        sys.exit("{RED}Only one of --out_file and --out_dir allowed{NC}")

    if args.areadef_file:
        if not os.path.exists(args.areadef_file):
            sys.exit(
                f"{RED}area definition file does not exist: {args.areadef_file}{NC}\n"
                "It should be an existing file called /somepath/<yourareaname>.py, "
                "and have the same format as area definitions in src/cpom/areas/definitions"
            )

    # ------------------------------------------------------------------------------------------
    #  Extract list of parameters and lat/lon names from arguments if available
    # ------------------------------------------------------------------------------------------

    if args.params:
        params = args.params.split(",")
    else:
        params = []

    if args.bool_mask_params:
        bool_mask_params = args.bool_mask_params.split(",")
    else:
        bool_mask_params = []

    if args.units:
        units = args.units.split(",")
    else:
        units = []

    num_data_sets = len(params)

    if args.latname:
        latnames = args.latname.split(",")
    else:
        latnames = []

    if args.lonname:
        lonnames = args.lonname.split(",")
    else:
        lonnames = []

    # ------------------------------------------------------------------------------------------
    #  Find list of files to plot (or could be an empty list if just plotting map)
    # ------------------------------------------------------------------------------------------

    # Find list of input netcdf files .nc, or .NC

    if args.file or args.dir:
        if args.file:
            files = [args.file]
        else:
            files = find_nc_files(args.dir, args.recursive, args.max_files, args.include)
    else:
        files = []

    log.info("Number of netcdf files found: %d", len(files))

    # ------------------------------------------------------------------------------------------
    # Attempt to identify file types and get default latitude, longitude names
    # ------------------------------------------------------------------------------------------

    if len(files) > 0:
        if len(params) == 0:  # ie if no parameters given on command line
            # find default parameter names for this file type
            def_param, def_units = get_default_param(files[0])
            if len(def_param) > 0:
                params = [def_param]
                units = [def_units]
                num_data_sets = len(params)
            else:
                sys.exit(
                    "No default parameter available for this file type. "
                    "Please specify using --params"
                )
        if num_data_sets > 0:
            if len(latnames) == 0 | len(lonnames) == 0:
                def_latname, def_lonname = get_default_latlon_names(files[0])
                if len(def_latname) > 0:
                    latnames = [def_latname] * num_data_sets
                    lonnames = [def_lonname] * num_data_sets
                else:
                    sys.exit(
                        "{RED}no default lat/lon parameters available, so please specify with"
                        " --latname , --lonname command line args{NC}"
                    )
                if len(latnames) != num_data_sets:
                    latnames = [latnames[0]] * num_data_sets
                    lonnames = [lonnames[0]] * num_data_sets
    else:
        num_data_sets = 1
        params = ["None"]

    log.info("param names: %s", str(params))
    log.info("unit names: %s", str(units))
    log.info("latnames : %s", str(latnames))
    log.info("lonnames : %s", str(lonnames))

    # ---------------------------------------------------------------------------
    # Build dataset dicts to plot
    # ---------------------------------------------------------------------------

    datasets = []

    error_str = ""
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

    if args.units:
        units = args.units.split(",")
        if len(units) != num_data_sets:
            units = [units[0]] * num_data_sets
            error_str += "units "

    if len(error_str) > 0:
        print(
            f"{ORANGE} WARNING: If you have {num_data_sets} commas separated --params args,"
            f" you can have an equal number of comma separated\n"
            f" {error_str} arguments{NC}"
        )

    flag_values_from_data = []
    flag_names_from_data = []

    for i in range(num_data_sets):
        ds = {
            "name": params[i],
            "lats": [],
            "lons": [],
            "vals": [],
            "plot_size_scale_factor": float(plot_size_scale_factor[i]),
            "cmap_name": cmap_names[i],
            "cmap_over_color": cmap_over[i],
            "cmap_under_color": cmap_under[i],
            "cmap_extend": cmap_extend[i],
            "apply_area_mask_to_data": not args.not_apply_mask,
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
        if units:
            ds["units"] = units[i]

        datasets.append(ds)
        flag_values_from_data.append([])
        flag_names_from_data.append([])

    def_area = ""
    if len(files) > 0:
        log.info("reading netcdf files...")
        for fnum, filename in enumerate(files):
            log.debug("%s", filename)
            # Open the NetCDF file
            try:
                with Dataset(filename) as nc:
                    for i in range(num_data_sets):
                        vals = get_variable(nc, params[i])[:].data
                        if vals.ndim == 3:
                            vals = vals[0].flatten()
                        try:
                            fill_value = get_variable(  # pylint: disable=protected-access
                                nc, params[i]
                            )._FillValue
                            if np.isinstance(vals[0], np.float):
                                vals[vals == fill_value] = np.nan
                            elif np.isinstance(vals[0], int):
                                vals[vals == fill_value] = None
                        except AttributeError:
                            pass
                        lats = get_variable(nc, latnames[i])[:].data
                        if lats.ndim == 2:
                            lats = lats.flatten()

                        lons = get_variable(nc, lonnames[i])[:].data % 360.0
                        if lons.ndim == 2:
                            lons = lons.flatten()
                        if len(bool_mask_params) == len(params):
                            bool_mask = get_variable(nc, bool_mask_params[i])[:].data.astype(bool)
                            vals = vals[bool_mask]
                            lats = lats[bool_mask]
                            lons = lons[bool_mask]

                        if args.and_flag:
                            flag_name, flag_value = args.and_flag.split(":")
                            flag_value = int(flag_value)
                            flag_vals = get_variable(nc, flag_name)[:].data
                            vals = vals[flag_vals == flag_value]
                            lats = lats[flag_vals == flag_value]
                            lons = lons[flag_vals == flag_value]

                        datasets[i]["lats"].extend(lats)
                        datasets[i]["lons"].extend(lons)
                        datasets[i]["vals"].extend(vals)

                        if not args.area and not def_area and not args.areadef_file:
                            def_area = get_default_area(lats[0], filename)
                    if fnum == 0:
                        try:
                            flag_values_from_data[i] = get_variable(nc, params[i]).flag_values
                            flag_names_from_data[i] = get_variable(nc, params[i]).flag_meanings

                            args.flag = True
                        except AttributeError:
                            pass
            except IOError:
                sys.exit(f"{RED}Can not open {filename}{NC}")

    if not args.area and not def_area and not args.areadef_file:
        sys.exit(
            f"{RED}--area area_name{NC} missing. Use --list_areas to show available area names"
        )

    try:
        if args.areadef_file:
            thisarea = Area("none", area_filename=args.areadef_file)
        elif args.area:
            thisarea = Area(args.area)
            def_area = args.area
        elif def_area:
            thisarea = Area(def_area)
        else:
            sys.exit("no area defined")

    except ImportError:
        sys.exit(
            f"{RED}Area name {BLUE}{def_area}{RED} not found in list of area definitions "
            f"src/cpom/areas/definitions{NC}."
            " Use --listareas to show available areas"
        )

    if len(files) == 0:
        if args.step:
            # Generate some simulated data
            lon_step = args.step
            lat_step = args.step

            # Generate the grid of points
            lon_values = np.arange(thisarea.minlon, thisarea.maxlon + lon_step, lon_step)
            lat_values = np.arange(thisarea.minlat, thisarea.maxlat + lat_step, lat_step)

            # Create the mesh grid
            lons, lats = np.meshgrid(lon_values, lat_values)
            vals = lats.copy()

            # Calculate the number of elements to set to NaN
            num_nan = int(len(vals) * 0.1)

            # Randomly select indices to be set to NaN
            nan_indices = np.random.choice(len(vals), num_nan, replace=False)

            # Set the selected indices to NaN
            vals[nan_indices] = np.nan

            datasets[0]["units"] = "degs N"
        else:
            lats = []
            lons = []
            vals = []
        datasets[0]["lats"].extend(lats)
        datasets[0]["lons"].extend(lons)
        datasets[0]["vals"].extend(vals)
        datasets[0]["name"] = "simulated" if args.step else "None"

    if args.flag or args.flag_params:
        if args.flag_params:
            flag_params = args.flag_params.split(",")
            if len(flag_params) == 1 and num_data_sets > 1:
                flag_params = flag_params * num_data_sets
        for i in range(num_data_sets):
            if args.flag_params:
                unique_vals = []
                unique_names = []
                unique_colors = []
                sub_params = flag_params[i].split("/")
                for sub_param in sub_params:
                    fvalues = sub_param.split(":")
                    unique_vals.append(int(fvalues[0]))
                    unique_names.append(fvalues[1])
                    unique_colors.append(fvalues[2])

                datasets[i]["flag_values"] = unique_vals
                datasets[i]["flag_names"] = unique_names
                datasets[i]["flag_colors"] = unique_colors
            else:
                if len(flag_values_from_data[i]) > 0 and len(flag_names_from_data[i]) > 0:
                    log.info("Using flag values from data: %s", flag_values_from_data[i])
                    unique_vals = [
                        int(re.findall(r"\d+", s)[0]) for s in flag_values_from_data[i].split(",")
                    ]
                    unique_names = flag_names_from_data[i].split(",")
                    if len(unique_names) != len(unique_vals):
                        log.error(
                            (
                                f"{ORANGE}length of flag names (%d) and values derived from "
                                f"attributes  (%d) is different %s %s {NC}"
                            ),
                            len(unique_names),
                            len(unique_vals),
                            str(unique_names),
                            str(unique_vals),
                        )
                        datasets[i]["flag_names"] = [str(n) for n in unique_vals]
                    else:
                        datasets[i]["flag_names"] = unique_names
                    datasets[i]["flag_values"] = unique_vals

                else:
                    unique_vals = np.unique(datasets[i]["vals"])
                    if len(unique_vals) > 30:
                        sys.exit(
                            f"{RED}Number of unique flag values found > 30 for parameter "
                            f"{params[i]}. "
                            f"Only use --flag with flag parameters with < 30 unique values{NC}"
                        )
                    log.info("unique flag vals %s", str(unique_vals))
                    datasets[i]["flag_values"] = unique_vals
                    datasets[i]["flag_names"] = [str(n) for n in unique_vals]

    area_overrides = {}
    if args.hillshade:
        area_overrides["apply_hillshade_to_vals"] = True

    Polarplot(def_area, area_overrides=area_overrides, area_file=args.areadef_file).plot_points(
        *datasets,
        output_file=args.out_file,
        output_dir=args.out_dir,
        map_only=args.map_only,
        dpi=85 if not args.dpi else args.dpi,
    )

    log.info("plot completed ok")


if __name__ == "__main__":
    main(sys.argv[1:])

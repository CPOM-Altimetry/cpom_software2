#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cpom.altimetry.tools.plot_3dmap.py

# Purpose

Tool to display (lat,lon,values) from one or more generic NetCDF files (containing latitude,
longitude and value data of any sort and naming) on an interactive  3d DEM map 
in a browser window. Will optionally recursively search a directory for netcdf files (optionally
matching a pattern). It is assumed that netcdf files matched are of the same type.

**Note** : this tool requires a graphics card or GPU on the system it is run on
and displays within the current default browser. So, will not work on headless
servers with no graphics cards or GPUs installed. Tested on a MacbookPro.


For full list of command line args:

`plot_3dmap.py --help`

Most settings are configurable from the tool using the command line arguments, although the tool
will try and automatically identify default parameters and select an area to plot 
(from contained lat/lon values) from most altimetry formats. If not you can choose these from the
command line.

## Examples

List all available 3D area definitions (ie the areas you can select to plot your data on):

`plot_3dmap.py --list_areas`

Plot a parameter **elevation** from a CryTEMPO netcdf file, and display in area 
definition **vostok**. 
Use colormap RdYlBu to display elevation values between 3400m and 3500m range. 

`plot_3dmap.py -f CS_OFFL_SIR_TDP_LI_ANTARC_20200911T023800_20200911T024631_18_12333_D001.nc \
    -a vostok -p elevation -c cmap:RdYlBu -pr 3400:3500`

Plot parameters **elevation** and **backscatter** from a CryTEMPO netcdf file, and display in area 
definition **vostok**. 
Use colormap RdYlBu to display **elevation** values between 3400m and 3500m range. 
Use colormap Viridis to display **backscatter** values between default ranges. 
Raise both parameters above the surface by 1m and 50m respectively
Set the point size to be 6 and 10 respectively

`plot_3dmap.py -f CS_OFFL_SIR_TDP_LI_ANTARC_20200911T023800_20200911T024631_18_12333_D001.nc \
    -a vostok -p elevation -c cmap:RdYlBu,cmap:viridis -pr 3400:3500, -re 1,50 -ps 6,10`


## Issues

Odd display issues with some DEM Zarr resolutions. For example using
'rema_gapless_100m_zarr' with area vostok_600km. Seems to display ok when
no parameters loaded, but incorrectly with them. 'rema_ant_1km_zarr' works
fine. Needs further investigation.

"""

import argparse
import glob
import logging
import os
import re
import sys
from typing import List

import numpy as np
from netCDF4 import Dataset, Variable  # pylint: disable=E0611

from cpom.areas.area_plot3d import plot_3d_area
from cpom.areas.areas3d import Area3d, list_all_3d_area_definition_names

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
        return "antarctica"
    if lat > 50.0:
        basename = os.path.basename(filename)
        if "GREENL" in basename:  # for CryoTEMPO Greenland files
            return "greenland"
        return "greenland"
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
    """main function of 3d plotting tool

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
            " on 3d DEM views in an interactive browser."
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
            "[optional] path of CPOM 3d area definition file. "
            "Not necessary if using standard area definitions in cpom.areas.definitions_3d."
            " Can be used instead of standard 3d area definitions"
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
        "--colours",
        "-c",
        help=(
            "[optional] comma separated list of colours (either single (ie red) or colormaps"
            " (ie RdYlBu) to use for plotting each parameter variable. Must be one entry "
            "per variable."
            "If using a colourmap then use the syntax cmap:colourmap, where colourmap is the"
            "name of the colourmap (ie RdYlBu). So to use a single colour for variable 1, and"
            "a colourmap for variable 2, use the following:  red,cmap:RdYlBu"
            " (default = single colour red for all variables)."
        ),
        required=False,
    )

    parser.add_argument(
        "--cmap_over",
        "-cmo",
        help=("colourmap over color name to to use."),
        required=False,
        default="",
    )

    parser.add_argument(
        "--cmap_under",
        "-cmu",
        help=("colourmap under color name to to use"),
        required=False,
        default="",
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
        "--dem_stride",
        "-ds",
        help=(
            "[Optional, int from 1] DEM stride to use (ie sampling of DEM)."
            "Default is to use that stored in the 3d area definition: dem_stride."
            "Reason for changing this is to experiment with the best DEM resolution to use"
            " for best visuals, but maintaining browser display performance. The higher"
            " this number the faster the scene will display and respond."
        ),
        required=False,
        type=int,
    )

    parser.add_argument(
        "--dir",
        "-d",
        help=(
            "[Optional] path of a directory containing netcdf files to plot."
            "All files are plotted within this directory. "
            "Use with --recursive for recursive search."
        ),
        required=False,
    )

    parser.add_argument(
        "--file",
        "-f",
        help=("[Optional] path of a single input netcdf file to plot parameters from"),
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
        "--include",
        "-i",
        help=("[optional] include only netcdf files with this string in their filename"),
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
        "--max_files",
        "-mf",
        help=("only read the first N input netcdf files (if multiple files input)"),
        required=False,
        type=int,
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
        "--no_place_annotations",
        "-npa",
        help=("disable default area place annotations on the 3D plot"),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--no_latlon_annotations",
        "-nla",
        help=("disable default latitude and longitude annotations on the 3D plot"),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--opacity",
        "-op",
        help=(
            "opacity of plotted parameter in range 0 to 1 (opaque)"
            "Comma separated list if more than 1 data set."
            "default is opacity of 1"
        ),
        required=False,
    )

    parser.add_argument(
        "--plot_ranges",
        "-pr",
        help=(
            "[optional,str] comma separated list of plot ranges to use with colourmaps "
            "per plot variable."
            "Format for n variables is min1:max1,min2:max2,..minn,maxn. "
            "If a variable is not using a "
            "colourmap then just use a comma but no spaces to skip it, "
            "ie min1:max1,,min3:max3. Note that if the value of min1 is negative"
            " it should be escaped with a \\ "
        ),
        required=False,
        type=str,  # must be str as is a comma separated list
    )

    parser.add_argument(
        "--plot_nan_fv",
        "-pn",
        help=(
            "[optional] when selected plot just the locations of the Nan or FillValue values in"
            "the parameter(s) being plotted and not the values themselves."
        ),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--pointsizes",
        "-ps",
        help=(
            "[optional,str] comma separated list of integer point sizes to use plotting vars"
            " default is 2 for all variables"
        ),
        required=False,
        type=str,  # must be str as is a comma separated list
    )

    parser.add_argument(
        "--raise_elevation",
        "-re",
        help=(
            "[optional] raise the elevation of the plot variable on the DEM by n meters."
            "Default is 1m to raise it above the background DEM surface. "
            "Comma separated if multiple parameters. example: --raise 1,100"
        ),
        required=False,
        type=str,
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
        "--time_index",
        "-ti",
        help=(
            "[optional] for netcdf parameters that are indexed by time period, select the"
            "index to plot. ie for a parameter like sec(time_period, ny, nx), the default"
            "is to plot sec(0,nx,ny), but you can select a different period."
        ),
        required=False,
        type=int,
        default=0,
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

    parser.add_argument(
        "--var_stride",
        "-vs",
        help=(
            "plot every Nth parameter value. Comma separated list if more"
            " than one parameter. Default is 1. Can speed up plotting and make"
            "3D browser window more responsive if the stride is increased."
        ),
        required=False,
    )

    parser.add_argument(
        "--zscale",
        "-zs",
        help=(
            "(float) : scale the DEM in the Z-axis. Default is in the area definitions"
            " as zaxis_multiplier: <value>. Raising this will increase the visual scale"
            " in z-direction (ie make terrain and mountains look less flat)"
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
        area_list = list_all_3d_area_definition_names()
        mystr = f"{BLACK_BOLD}List of Available Area Names from CPOM Area Definitions{NC}"
        print("-" * (len(mystr) - 8))
        print(mystr)
        print("-" * (len(mystr) - 8))
        for area_name in area_list:
            print(f"{area_name}")
        sys.exit(0)

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
    #  Find list of files to plot (or could be an empty list if just plotting 3d scene from DEM)
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

    error_str = ""

    if args.pointsizes:
        pointsizes = args.pointsizes.split(",")
        pointsizes = [int(pointsize) for pointsize in pointsizes]
        while len(pointsizes) < num_data_sets:
            pointsizes.append(pointsizes[-1])
    else:
        pointsizes = [2] * num_data_sets

    # Decide on which colours to use for each variable. This can either be a single colour
    # or a colourmap (with colours mapped to a specific variable data range)
    colours = ["red"] * num_data_sets  # default is single colour red for each variable
    cmaps = [None] * num_data_sets
    usecmaps = [False] * num_data_sets
    if args.colours:
        colours = args.colours.split(",")
        if len(colours) != num_data_sets:
            sys.exit("--colours error : must include an entry for each variable. ")
        for varnum, colour in enumerate(colours):
            colour_split = colour.split(":")
            if len(colour_split) == 2:
                if colour_split[0] != "cmap":
                    sys.exit("--colours syntax error, colourmap entries must be cmap:colourmapstr")
                usecmaps[varnum] = True
                cmaps[varnum] = colour_split[1]

    if args.raise_elevation:
        raise_elevation = args.raise_elevation.split(",")
        while len(raise_elevation) < num_data_sets:
            raise_elevation.append(raise_elevation[-1])
    else:
        raise_elevation = [1.0] * num_data_sets

    plot_ranges = [None] * num_data_sets

    if args.plot_ranges:
        plot_ranges = args.plot_ranges.split(",")
        # min1:max1,min2:max2,..minn,maxn or min1:max1,, if plot range not requirwd
        # for variable
        for varnum, plot_range in enumerate(plot_ranges):
            minmax = plot_range.split(":")
            if len(minmax) == 2:
                tmp_pr = np.array([float(minmax[0]), float(minmax[1])])
                plot_ranges[varnum] = [tmp_pr.min(), tmp_pr.max()]
            elif len(minmax) == 1 and minmax[0] != "":
                sys.exit(
                    f"--plot_ranges syntax error : args used: {args.plot_ranges}. "
                    f"Error in {minmax} part. Must be pairs of min:max or empty as ,,"
                )

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

    if args.opacity:
        opacity = args.opacity.split(",")
        if len(opacity) != num_data_sets:
            opacity = [opacity[0]] * num_data_sets
            error_str += "opacity "
    else:
        opacity = [1.0] * num_data_sets

    if args.units:
        units = args.units.split(",")
        if len(units) != num_data_sets:
            units = [units[0]] * num_data_sets
            error_str += "units "

    if args.var_stride:
        var_stride = args.var_stride.split(",")
        if len(var_stride) != num_data_sets:
            var_stride = [var_stride[0]] * num_data_sets
            error_str += "var_stride "
    else:
        var_stride = [1] * num_data_sets

    if len(error_str) > 0:
        print(
            f"{ORANGE} WARNING: If you have {num_data_sets} commas separated --params args,"
            f" you can have an equal number of comma separated\n"
            f" {error_str} arguments{NC}"
        )

    flag_values_from_data = []
    flag_names_from_data = []
    datasets = []

    for i in range(num_data_sets):
        ds = {
            "name": params[i],
            "lats": [],  # added in later
            "lons": [],  # added in later
            "vals": [],  # added in later
            "point_size": float(pointsizes[i]),
            "color": colours[i],
            "raise_elevation": float(raise_elevation[i]),
            "use_colourmap": usecmaps[i],
            "colourmap": cmaps[i],
            "cmap_under_color": cmap_under[i],
            "cmap_over_color": cmap_over[i],
            "plot_range": plot_ranges[i],
            "apply_area_mask_to_data": not args.not_apply_mask,
            "var_stride": int(var_stride[i]),
            "plot_alpha": opacity[i],
            "plot_nan_and_fv": args.plot_nan_fv,
        }

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
                            if args.time_index > (vals.shape[0] - 1):
                                sys.exit(
                                    f"shape of parameter {params[i]} is {vals.shape}, "
                                    f"but index {args.time_index} selected with --time_index."
                                    f"Max allowed {(vals.shape[0] -1)} "
                                )
                            vals = vals[args.time_index].flatten()
                        try:
                            fill_value = get_variable(  # pylint: disable=protected-access
                                nc, params[i]
                            )._FillValue
                            vals[vals == fill_value] = np.nan
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
            thisarea = Area3d("none", area_filename=args.areadef_file)
        elif args.area:
            thisarea = Area3d(args.area)
            def_area = args.area
        elif def_area:
            thisarea = Area3d(def_area)
        else:
            sys.exit("no area defined")

    except ImportError:
        sys.exit(
            f"{RED}Area name {BLUE}{def_area}{RED} not found in list of area definitions "
            f"src/cpom/areas/definitions_3d{NC}."
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
                    unique_names = flag_names_from_data[i].split(" ")
                    if len(unique_names) != len(unique_vals):
                        log.error(
                            (
                                f"{ORANGE}length of flag names (%d) and values derived from "
                                f"attributes  (%d) is different{NC}"
                            ),
                            len(unique_names),
                            len(unique_vals),
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

    if args.no_place_annotations:
        area_overrides["place_annotations"] = {}
    if args.no_latlon_annotations:
        area_overrides["lat_annotations"] = {}
        area_overrides["lon_annotations"] = {}
    if args.zscale:
        area_overrides["zaxis_multiplier"] = float(args.zscale)
    if args.dem_stride:
        if args.dem_stride < 1:
            sys.exit("--dem_stride n must be > 0")
        area_overrides["dem_stride"] = args.dem_stride

    # Create the 3D plot in a browser page
    plot_3d_area(args.area, *datasets, area_overrides=area_overrides)

    log.info("plot completed ok")


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate sea ice parameter plots for CPOM sea ice portal

NRT CS2 Thickness Input:
========================
/cpnet/altimetry/seaice/nrt_system/latest/thk_02.map   # sparse grid
/cpnet/altimetry/seaice/nrt_system/latest/thk_14.map
/cpnet/altimetry/seaice/nrt_system/latest/thk_28.map

/cpnet/altimetry/seaice/nrt_system/latest/thk_02.info  # start/end date, 
ie: 29 1 2022 25 2 2022
/cpnet/altimetry/seaice/nrt_system/latest/thk_14.info
/cpnet/altimetry/seaice/nrt_system/latest/thk_28.info

Monthly
==============================

/cpnet/altimetry/seaice/CS2,S3A,S3B/arco,anto/archive/YYYYMM.map  [Thickness]
/cpnet/altimetry/seaice/CS2,S3A,S3B/arco,anto/archive/<YYYYMM>.<parameter>.map

<parameter> : FloeChordLength, IceConcentration, IceType,LeadFloeFraction, 
 RadarFreeboard, SeaLevelAnomaly, WarrenSnowDepth, UnkFraction

Seasonal
==============================

/cpnet/altimetry/seaice/CS2,S3A,S3B/arco,anto/archive/MarApr<YYYY>.map
/cpnet/altimetry/seaice/CS2,S3A,S3B/arco,anto/archive/OctNov<YYYY>.map

.map format:
1st line: start end dates
2nd line+: lat lon-e thickness stdev numvals cog

4 1 2022 31 1 2022
48.327957 -45.000000 0.0000 0.0000 0 0.0000

"""

import argparse  # for command line arguments
import os
import sys
from os.path import exists

import matplotlib.pyplot as plt
import pandas as pd
from cmocean import cm  # pylint: disable=unused-import, no-member # noqa
from si_params import SIParams, all_si_params
from update_stats import update_availability_database

from cpom.areas.area_plot import (  # cryosphere map plotting functions
    Annotation,
    Polarplot,
)
from cpom.areas.areas import Area  # cryosphere area definitions

# pylint: disable=invalid-name

PLOT_SCALE_FACTOR = 0.02

# ----------------------------------------------------------------------------------------------
#  Command line option processing
# ------------------------------------------------------------------------------------------------

# initiate the command line parser
parser = argparse.ArgumentParser()

# add long and short command line arguments
parser.add_argument("--mission", "-m", help="mission: one of cs2, s3a, s3b, env")
parser.add_argument(
    "--south",
    "-s",
    help="[optional] process southern hemisphere instead of north",
    action="store_const",
    const=1,
)
parser.add_argument(
    "--area",
    "-a",
    help=(
        "[optional for testing single areas]. Two arguments:"
        " plot_area area_name_for_output. plot_area must be in area_plot_list,"
        " area_output_list below: eg arctic0_seaiceportal, arctic_basin1,.."
    ),
    nargs=2,
)
parser.add_argument(
    "--param", "-p", help="[optional] parameter_name: only plot this sea ice parameter"
)
parser.add_argument("--year", "-y", help="YYYY")
parser.add_argument(
    "--month", "-mo", help="[optional, default is to process all months 1-12] month number"
)
parser.add_argument("--nrt", "-n", help="process NRT files only", action="store_const", const=1)
parser.add_argument(
    "--season",
    "-se",
    help="process seasonal (autumn/spring) files only",
    action="store_const",
    const=1,
)
parser.add_argument("--outdir", "-o", help="Output base directory for plots")

# read arguments from the command line
args = parser.parse_args()

if not args.outdir:
    sys.exit("--outdir missing")
if not args.mission:
    sys.exit("--mission missing")

mission = args.mission.lower()
if mission == "cs2":
    mission_longname = "CryoSat-2"
elif mission == "s3a":
    mission_longname = "Sentinel-3A"
elif mission == "s3b":
    mission_longname = "Sentinel-3B"
elif mission == "env":
    mission_longname = "Envisat"
else:
    sys.exit(f"Unknown mission: {mission}")

year = 0
if args.year:
    year = int(args.year)

if args.season:
    if not args.year:
        sys.exit("--season must also include --year")

if not args.nrt:
    if not (args.year or args.season):
        sys.exit("Must include either --nrt, --year or --season")


hemisphere = "north"
if args.south:
    hemisphere = "south"

# -------------------------------------------------------------------------------------
# Area lists to process for gridded and along-track plots
# -------------------------------------------------------------------------------------


# if hemisphere == "south":
#     area_plot_list = [
#         "antarctica_ocean_seaiceportal",
#         "antarctic_ocean_sector_ross",
#         "antarctic_ocean_sector_pacific",
#         "antarctic_ocean_sector_indian",
#         "antarctic_ocean_sector_east_weddell",
#         "antarctic_ocean_sector_west_weddell",
#         "antarctic_ocean_sector_coastal_amundsen_bellingshausen",
#         "antarctic_ocean_sector_amundsen_bellingshausen",
#     ]
#     area_output_list = ["anto", "anto1", "anto2", "anto3", "anto4", "anto5", "anto6", "anto7"]
#     archive_area = "anto"
# else:
#     area_plot_list = ["arctic0_seaiceportal"]
#     area_output_list = ["arco"]
#     archive_area = "arco"

#     # append basins 1..17 to area_plot_list
#     for i in range(1, 18):
#         area_plot_list.append(f"arctic_basin{i}")
#         area_output_list.append(f"arco{i}")

# if args.area:
#     area_plot_list = [args.area[0]]
#     area_output_list = [args.area[1]]

if hemisphere == "south":
    archive_area = "anto"
    area_plot_list = ["antarctica_ocean_seaiceportal"]
    area_output_list = ["anto"]
else:
    archive_area = "arco"
    area_plot_list = ["arctic0_seaiceportal"]
    area_output_list = ["arco"]


# -------------------------------------------------------------------------------------
# Logo
# -------------------------------------------------------------------------------------

# Get the CPOM Logo and set position as top left corner
logo_image = plt.imread("images/cpom_mini.png")

logo_width = 0.09  # in axis coordinates
logo_height = 0.09
logo_position = (
    0.02,
    1 - logo_height - 0.02,
    logo_width,
    logo_height,
)  # [left, bottom, width, height]

common_annotations = []
common_annotations.append(
    Annotation(
        0.388,
        0.310,
        "76°N",
        fontsize=9,
        fontweight="normal",
        color="#626262",
    )
)
common_annotations.append(
    Annotation(
        0.37,
        0.22,
        "68°N",
        fontsize=9,
        fontweight="normal",
        color="#626262",
    )
)

common_annotations.append(
    Annotation(
        0.354,
        0.122,
        "60°N",
        fontsize=9,
        fontweight="normal",
        color="#626262",
    )
)

# parameter string:
common_annotations.append(
    Annotation(0.21, 0.955, "Parameter:", fontsize=10, fontweight="normal", color="grey")
)


# -------------------------------------------------------------------------------------
# Process Monthly Plots for each area
#   from
#   - gridded
#   - along-track
#   also create an availability database of measurements per month
# -------------------------------------------------------------------------------------

if args.year and not args.season:
    # --------------------------------------------------------------------------------------------
    #  Process Months
    # ==============================
    #
    # Gridded at 5km:
    #    6 columns of txt  : <lat> <lon-e> <param> <stdev> <numvals> <cog>
    #
    # /cpnet/altimetry/seaice/CS2,S3A,S3B,ENV/arco,anto/archive/YYYYMM.map  [Thickness]
    # /cpnet/altimetry/seaice/CS2,S3A,S3B,ENV/arco,anto/archive/<YYYYMM>.<parameter>.map
    #
    # Along-track:
    #   11 columns of txt : <int> <int> <float> <lat> <lon-e> <int> <thk> <int><int><float><float>
    # /cpnet/altimetry/seaice/CS2,S3A,S3B,ENV/arco,anto/archive/YYYYMM.thk  [Thickness]
    #   3 columns of txt : <lat> <lon-e> <param>
    # /cpnet/altimetry/seaice/CS2,S3A,S3B,ENV/arco,anto/archive/<YYYYMM>.<parameter>
    #
    # <parameter> : FloeChordLength, IceConcentration, IceType,LeadFloeFraction,  RadarFreeboard,
    # SeaLevelAnomaly, WarrenSnowDepth, UnkFractionfile
    #
    # Output to
    # <outdir>/<mission>/<arco,anto>/<YYYY>/<mission>_<YYYYMM>_<parameter>.<imagewidth>.png
    #
    # --------------------------------------------------------------------------------------------

    for month in range(1, 13):
        if args.month:
            if month != int(args.month):
                continue

        latency = "Final, Precise Orbit"

        # Find number of along track thickness measurements for this month

        along_track_thickness_file = (
            f"/cpnet/altimetry/seaice/{mission.upper()}/"
            f"{archive_area}/archive/{args.year}{month:02d}.thk"
        )
        if not exists(along_track_thickness_file):
            print(f"{along_track_thickness_file} does not exist")
            continue

        count = 0
        with open(along_track_thickness_file, "r", encoding="utf-8") as fp:
            for count, line in enumerate(fp):
                pass
        print("Total Lines", count + 1)

        update_availability_database(
            args.outdir, mission, archive_area, args.year, month, count + 1
        )

        for i, param in enumerate(all_si_params):
            if args.param:
                if param != args.param:
                    continue

            si_param = SIParams(param)

            # -----------------------------------------------------------------------------------
            # Read 5km gridded map file  (.map)
            # -----------------------------------------------------------------------------------

            map_file = (
                f"/cpnet/altimetry/seaice/{mission.upper()}/"
                f"{archive_area}/archive/{args.year}{month:02d}.{param}.map"
            )
            if param == "Thickness":
                map_file = (
                    f"/cpnet/altimetry/seaice/{mission.upper()}/"
                    f"{archive_area}/archive/{args.year}{month:02d}.map"
                )

            print("Reading gridded map file: ", map_file)

            if not exists(map_file):
                print(f"{map_file} does not exist")
                continue

            pd_data = pd.read_csv(map_file, sep=r"\s+")
            pd_data.columns = ["lat", "lon", "thickness", "stdev", "numvals", "dist"]

            lats = pd_data["lat"]
            lons = pd_data["lon"]
            vals = pd_data["thickness"]

            # ------------------------------------------------------------------------------------
            # Read along-track parameter file
            #    Thickness: 201812.thk (11 cols)
            #    RadarFreeboard: 201812.RadarFreeboard (3 cols)
            #    SeaLevelAnomaly: 201812.SeaLevelAnomaly (3 cols)
            #    IceType: 201812.IceType (3 cols)
            #    IceConcentration: 201812.IceConcentration (3 cols)
            #    FloeChordLength : 201812.FloeChordLength (6 cols)
            #    (201812.LeadFloeFraction (7 cols)) not used
            #    UnkFraction: 201812.UnkFraction (none)
            #    LeadFraction: 201812.LeadFraction (none)
            #    FloeFraction: 201812.FloeFraction (none)
            #    WarrenSnowDepth: 201812.WarrenSnowDepth (3 cols)
            # -----------------------------------------------------------------------------------

            if "Fraction" not in param:
                alongtrack_file = (
                    f"/cpnet/altimetry/seaice/{mission.upper()}"
                    f"/{archive_area}/archive/{args.year}{month:02d}.{param}"
                )
                if param == "Thickness":
                    alongtrack_file = (
                        f"/cpnet/altimetry/seaice/{mission.upper()}"
                        f"/{archive_area}/archive/{args.year}{month:02d}.thk"
                    )

                if not exists(alongtrack_file):
                    sys.exit(f"{alongtrack_file} does not exist")

                print("Reading along track file: ", alongtrack_file)

                pd_data = pd.read_csv(alongtrack_file, sep=r"\s+")

                if param == "Thickness":  # 11 columns
                    pd_data.columns = [
                        "int1",
                        "int2",
                        "float1",
                        "lat",
                        "lon",
                        "int3",
                        "param",
                        "int4",
                        "int5",
                        "float2",
                        "float3",
                    ]
                elif param == "FloeChordLength":  # 6 columns
                    pd_data.columns = ["lat", "lon", "param", "float2", "int1", "int2"]
                else:
                    pd_data.columns = ["lat", "lon", "param"]

                lats_alongtrack = pd_data["lat"]
                lons_alongtrack = pd_data["lon"]
                vals_alongtrack = pd_data["param"]
            else:
                lats_alongtrack = []
                lons_alongtrack = []
                vals_alongtrack = []

            for i, area in enumerate(area_plot_list):
                thisarea = Area(area)

                # -------------------------------------------------------------------------------
                # Annotations
                # -------------------------------------------------------------------------------
                annotation_list = []  # an empty annotation list

                # ------------------------------------------------------------------
                # param_longname:  rounded blue box to right of logo

                # parameter string:

                props = {
                    "boxstyle": "round",
                    "facecolor": "aliceblue",
                    "alpha": 1.0,
                    "edgecolor": "lightgrey",
                }
                thisfontsize = 25
                if len(si_param.long_name) > 25:
                    thisfontsize = 15
                annotation_list.append(
                    Annotation(
                        0.24,
                        0.92,
                        si_param.long_name,
                        bbox=props,
                        fontsize=thisfontsize,
                        fontweight="bold",
                    )
                )

                # processor string:
                annotation_list.append(
                    Annotation(
                        0.02,
                        0.878,
                        "Sea Ice Processor",
                        fontsize=9,
                        fontweight="normal",
                        color="#000049",
                    )
                )

                # Area
                annotation_list.append(
                    Annotation(
                        0.40 - 0.005 * (len(thisarea.long_name) - 6),
                        0.87,
                        f"{thisarea.long_name}",
                        fontsize=15,
                        fontweight="normal",
                    )
                )

                # Mission:
                annotation_list.append(
                    Annotation(
                        0.685,
                        0.96,
                        f"Mission: {mission_longname}",
                        fontsize=18,
                        fontweight="bold",
                    )
                )

                # Latency:
                annotation_list.append(
                    Annotation(0.685, 0.92, f"Latency: {latency}", fontsize=14, fontweight="normal")
                )

                # Period:
                annotation_list.append(
                    Annotation(0.685, 0.88, "Month:", fontsize=12, fontweight="normal")
                )
                annotation_list.append(
                    Annotation(
                        0.745, 0.87, f"{month:02d}/{args.year}", fontsize=28, fontweight="bold"
                    )
                )

                # Latitude annotation
                annotation_list.append(
                    Annotation(
                        0.024,
                        0.874,
                        "70°N",
                        fontsize=8,
                        fontweight="normal",
                        color="#000049",
                    )
                )

                annotation_list.extend(common_annotations)

                # Create output directories
                #  <outdir>/<mission>/<arco,anto>/<YYYY>
                outdir = f"{args.outdir}/{mission}/ntc/{archive_area}/{args.year}"

                if not os.path.exists(outdir):
                    try:
                        print("Creating ", outdir)
                        os.makedirs(outdir)
                    except OSError as exc:
                        raise OSError(f"Can't create directory ({outdir})!") from exc
                else:
                    print("Output dir: ", outdir)

                if len(lats_alongtrack) > 0:

                    dataset = {
                        "lats": lats_alongtrack,
                        "lons": lons_alongtrack,
                        "vals": vals_alongtrack,
                        "name": si_param.long_name,
                        "units": si_param.units,
                        "plot_size_scale_factor": PLOT_SCALE_FACTOR,
                        "plot_alpha": 1.0,
                        "apply_area_mask_to_data": True,
                        "min_plot_range": si_param.plot_range_low,
                        "max_plot_range": si_param.plot_range_high,
                        "cmap_name": "cmo.thermal",
                        "cmap_over_color": "yellow",
                        "cmap_under_color": "black",
                    }
                    Polarplot(area).plot_points(
                        dataset,
                        output_dir=outdir,
                        output_file=(
                            f"{mission}_{archive_area}_{args.year}"
                            f"{month:02d}_{si_param.param.lower()}"
                        ),
                        annotation_list=annotation_list,
                        use_default_annotation=False,
                        logo_image=logo_image,
                        logo_position=logo_position,
                        figure_height=12,
                        figure_width=12,
                        image_format="webp",
                    )

                annotation_list.append(
                    Annotation(
                        0.685, 0.84, "Monthly Mean (5km grid)", fontsize=12, fontweight="normal"
                    )
                )

                annotation_list.extend(common_annotations)

                dataset = {
                    "lats": lats,
                    "lons": lons,
                    "vals": vals,
                    "name": si_param.long_name,
                    "units": si_param.units,
                    "plot_size_scale_factor": PLOT_SCALE_FACTOR,
                    "apply_area_mask_to_data": True,
                    "min_plot_range": si_param.plot_range_low,
                    "max_plot_range": si_param.plot_range_high,
                    "cmap_name": "cmo.thermal",
                    "cmap_over_color": "yellow",
                    "cmap_under_color": "black",
                }

                Polarplot(area).plot_points(
                    dataset,
                    output_dir=outdir,
                    output_file=(
                        f"{mission}_{area_output_list[i]}_{args.year}{month:02d}"
                        f"_{si_param.param.lower()}"
                    ),
                    annotation_list=annotation_list,
                    use_default_annotation=False,
                    logo_image=logo_image,
                    logo_position=logo_position,
                    figure_height=12,
                    figure_width=12,
                    image_format="webp",
                )


if args.nrt:
    # --------------------------------------------------------------------------------------------
    #  Process CS2 NRT (currently only thickness)
    #
    # /cpnet/altimetry/seaice/nrt_system/latest/thk_02.map   # sparse grid
    # /cpnet/altimetry/seaice/nrt_system/latest/thk_14.map
    # /cpnet/altimetry/seaice/nrt_system/latest/thk_28.map
    #
    # /cpnet/altimetry/seaice/nrt_system/latest/thk_02.info #start/end date, 29 1 2022 25 2 2022
    # /cpnet/altimetry/seaice/nrt_system/latest/thk_14.info
    # /cpnet/altimetry/seaice/nrt_system/latest/thk_28.info
    #
    # Output to
    # <outdir>/<mission>/<arco,anto>/<mission>_<area>_<period>days_<parameter>.<image width>.png
    # --------------------------------------------------------------------------------------------

    si_param = SIParams("Thickness")
    nrt_periods = ["02", "14", "28"]
    latency = "Near Real Time"

    for period in nrt_periods:
        map_file = f"/cpnet/altimetry/seaice/nrt_system/latest/thk_{period}.map"
        info_file = f"/cpnet/altimetry/seaice/nrt_system/latest/thk_{period}.info"

        # Read the .info file
        with open(info_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[0].split()  # list containing lines of file
            start_day = int(lines[0])
            start_month = int(lines[1])
            start_year = int(lines[2])
            end_day = int(lines[3])
            end_month = int(lines[4])
            end_year = int(lines[5])

        # Read map file

        print("Reading map file: ", map_file)

        pd_data = pd.read_csv(map_file, sep=r"\s+")
        pd_data.columns = ["lat", "lon", "thickness", "stdev", "numvals", "dist"]

        lats = pd_data["lat"]
        lons = pd_data["lon"]
        vals = pd_data["thickness"]

        for i, area in enumerate(area_plot_list):
            thisarea = Area(area)

            # -------------------------------------------------------------------------------------
            # Annotations
            # -------------------------------------------------------------------------------------
            annotation_list = []  # an empty annotation list

            props = {
                "boxstyle": "round",
                "facecolor": "aliceblue",
                "alpha": 1.0,
                "edgecolor": "lightgrey",
            }
            thisfontsize = 20
            if len(si_param.long_name) > 25:
                thisfontsize = 15
            annotation_list.append(
                Annotation(
                    0.21,
                    0.92,
                    si_param.long_name,
                    bbox=props,
                    fontsize=thisfontsize,
                    fontweight="bold",
                )
            )

            annotation_list.append(
                Annotation(0.4, 0.87, f"{thisarea.long_name}", fontsize=17, fontweight="normal")
            )

            # Mission:
            annotation_list.append(
                Annotation(
                    0.685, 0.95, f"Mission: {mission_longname}", fontsize=18, fontweight="bold"
                )
            )

            # Latency:
            annotation_list.append(
                Annotation(0.02, 0.81, f"{latency}", fontsize=10, fontweight="normal")
            )
            # Latency:
            annotation_list.append(
                Annotation(0.02, 0.79, "Sea Ice Processor", fontsize=10, fontweight="normal")
            )

            # Period:
            annotation_list.append(
                Annotation(0.685, 0.89, "Latest:", fontsize=14, fontweight="normal")
            )
            # Period:
            annotation_list.append(
                Annotation(0.75, 0.89, f"{int(period)} days", fontsize=24, fontweight="bold")
            )

            # Period:
            annotation_list.append(
                Annotation(
                    0.03,
                    0.85,
                    "NRT",
                    fontsize=24,
                    fontweight="normal",
                    color="white",
                    bbox={
                        "boxstyle": "round",
                        "facecolor": "#0e2a47",
                        "alpha": 1.0,
                        "edgecolor": "#0e2a47",
                    },
                )
            )

            # Start Date:
            annotation_list.append(
                Annotation(
                    0.685,
                    0.84,
                    f"Start: {start_day}-{start_month}-{start_year}",
                    fontsize=14,
                    fontweight="normal",
                )
            )

            # End Date:
            annotation_list.append(
                Annotation(
                    0.685,
                    0.81,
                    f"End:  {end_day}-{end_month}-{end_year}",
                    fontsize=14,
                    fontweight="normal",
                )
            )

            annotation_list.extend(common_annotations)

            # Create output directories
            #  <outdir>/<mission>/<arco,anto>/nrt
            outdir = f"{args.outdir}/{mission}/nrt/{area_output_list[i]}"

            if not os.path.exists(outdir):
                try:
                    print("Creating ", outdir)
                    os.makedirs(outdir)
                except OSError as exc:
                    raise OSError(f"Can't create directory ({outdir})!") from exc
            else:
                print("Output dir: ", outdir)

            dataset = {
                "lats": lats,
                "lons": lons,
                "vals": vals,
                "name": si_param.long_name,
                "units": "m",
                "plot_size_scale_factor": PLOT_SCALE_FACTOR,
                "min_plot_range": 0.0,
                "max_plot_range": 3.5,
                "cmap_name": "cmo.thermal",
                "cmap_over_color": "yellow",
                "cmap_under_color": "black",
            }

            Polarplot(area).plot_points(
                dataset,
                output_dir=outdir,
                output_file=(
                    f"{mission}_{area_output_list[i]}_"
                    f"{int(period)}days_{si_param.param.lower()}"
                ),
                annotation_list=annotation_list,
                use_default_annotation=False,
                logo_image=logo_image,
                logo_position=logo_position,
                figure_height=12,
                figure_width=12,
                image_format="webp",
            )

if args.season:
    # ------------------------------------------------------------------------------------------
    #  Process CS2 NRT (currently only thickness)
    #
    #  Seasonal
    #  == == == == == == == == == == == == == == ==

    #  /cpnet/altimetry/seaice/CS2, S3A, S3B, ENV/arco, anto/archive/MarApr<YYYY>.map
    #  /cpnet/altimetry/seaice/CS2, S3A, S3B, ENV/arco, anto/archive/OctNov<YYYY>.map
    #
    # Output to <outdir>/<mission>/<arco,anto>/<YYYY>/<mission>_<area>_<season>_
    # <parameter>.<image width>.png
    # ------------------------------------------------------------------------------------------

    si_param = SIParams("Thickness")
    seasons = ["MarApr", "OctNov"]
    latency = "Final, Precise Orbit"

    for season in seasons:
        map_file = (
            f"/cpnet/altimetry/seaice/{mission.upper()}/{archive_area}/archive/{season}{year}.map"
        )

        # Read map file

        pd_data = pd.read_csv(map_file, sep=r"\s+")
        pd_data.columns = ["lat", "lon", "thickness", "stdev", "numvals", "dist"]

        lats = pd_data["lat"]
        lons = pd_data["lon"]
        vals = pd_data["thickness"]

        for i, area in enumerate(area_plot_list):
            thisarea = Area(area)

            # -------------------------------------------------------------------------------------
            # Annotations
            # -------------------------------------------------------------------------------------
            annotation_list = []  # an empty annotation list

            props = {
                "boxstyle": "round",
                "facecolor": "aliceblue",
                "alpha": 1.0,
                "edgecolor": "lightgrey",
            }
            thisfontsize = 20
            if len(si_param.long_name) > 25:
                thisfontsize = 15
            annotation_list.append(
                Annotation(
                    0.21,
                    0.92,
                    si_param.long_name,
                    bbox=props,
                    fontsize=thisfontsize,
                    fontweight="bold",
                )
            )

            annotation_list.append(
                Annotation(0.32, 0.86, f"{thisarea.long_name}", fontsize=14, fontweight="normal")
            )

            annotation_list.append(
                Annotation(
                    0.02,
                    0.878,
                    "Sea Ice Processor",
                    fontsize=9,
                    fontweight="normal",
                    color="#000049",
                )
            )

            # Mission:
            annotation_list.append(
                Annotation(
                    0.685,
                    0.95,
                    f"Mission: {mission_longname}",
                    fontsize=18,
                    fontweight="bold",
                )
            )

            # Latency:
            annotation_list.append(
                Annotation(0.685, 0.9, f"Latency: {latency}", fontsize=14, fontweight="normal")
            )

            # Period:
            if season == "MarApr":
                season_str = "Spring (Mar/Apr)"
            elif season == "OctNov":
                season_str = "Autumn (Oct/Nov)"
            else:
                season_str = season

            annotation_list.append(
                Annotation(0.685, 0.82, f"{season_str}", fontsize=24, fontweight="normal")
            )

            annotation_list.append(
                Annotation(0.685, 0.76, f"{year}", fontsize=30, fontweight="bold")
            )

            annotation_list.extend(common_annotations)

            # Create output directories
            #  <outdir>/<mission>/<arco,anto>/<YYYY>
            outdir = f"{args.outdir}/{mission}/{area}/{year}"

            if not os.path.exists(outdir):
                try:
                    print("Creating ", outdir)
                    os.makedirs(outdir)
                except OSError as e:
                    raise OSError(f"Can't create directory ({outdir})! {e}") from e
            else:
                print("Output dir: ", outdir)

            dataset = {
                "lats": lats,
                "lons": lons,
                "vals": vals,
                "name": si_param.long_name,
                "units": "m",
                "plot_size_scale_factor": PLOT_SCALE_FACTOR,
                "min_plot_range": 0.0,
                "max_plot_range": 3.5,
                "cmap_name": "cmo.thermal",
                "cmap_over_color": "yellow",
                "cmap_under_color": "black",
            }

            Polarplot(area).plot_points(
                dataset,
                output_dir=outdir,
                output_file=f"{mission}_{area}_{season.lower()}_{si_param.param.lower()}",
                annotation_list=annotation_list,
                use_default_annotation=False,
                logo_image=logo_image,
                logo_position=logo_position,
                figure_height=12,
                figure_width=12,
                image_format="webp",
            )

"""
cpom.altimetry.projects.ais_cci_plus_phase2.plot_annual_sec.py

# Purpose

Plot parameters from AIS CCI+ phase-2 multi-mission sec products

Product naming:
ESACCI-AIS-L3C-SEC-MULTIMISSION_ANNUAL_CUMULATIVE_sec_GRID-5KM_199201_<YYYY><MM>-fv2.nc
where <MM> is normally 12 except for the last year where it could be 01..12
example:
ESACCI-AIS-L3C-SEC-MULTIMISSION_ANNUAL_CUMULATIVE_sec_GRID-5KM_199201_202212-fv2.nc


"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from cpom.areas.area_plot import Annotation, Polarplot


def main():
    """main function for tool"""

    print("in main")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prod_dir",
        "-d",
        help="path of input multi-mission sec products",
    )

    parser.add_argument(
        "--prod_filename",
        "-f",
        help="path of input multi-mission sec product file",
    )

    parser.add_argument(
        "--outdir",
        "-od",
        help=("file name in output directory for output file."),
    )
    parser.add_argument(
        "--area",
        "-a",
        help="Area of interest, antarctica_cci or ase_cci.",
        default="antarctica_cci",
    )

    parser.add_argument(
        "--parameter",
        "-p",
        help=("parameter to plot: sec, uncertainty, basin_id, surface_type"),
        default="sec",
    )

    if len(sys.argv) == 1:
        print("no args provided")
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    # Extract CCI netcdf file: example:
    # ESACCI-AIS-L3C-SEC-MULTIMISSION-5KM-5YEAR-MEANS-202006-202506-fv2.nc

    if args.prod_dir:
        input_files = glob.glob(
            f"{args.prod_dir}/ESACCI-AIS-L3C-SEC-MULTIMISSION-5KM-5YEAR-MEANS*fv2.nc"
        )
    elif args.prod_filename:
        input_files = [args.prod_filename]
    else:
        sys.exit("Must have either --prod_filename or --prod_dir")

    output_dir = args.outdir
    if not output_dir:
        output_dir = os.path.dirname(input_files[0])

    for input_file in input_files:
        print(f"Processing {input_file}")
        prod_name = os.path.basename(input_file)
        print(f"Base name {prod_name}")

        out_file = f"{output_dir}/{prod_name.replace('.nc',f'-{args.parameter.lower()}')}"
        print(f"Output file base {out_file}")

        if "ase" in args.area:
            out_file = f"{out_file}-ase"

        if args.parameter == "sec":
            param_long_name = "5-Year Multi-Mission Rate of Surface Elevation Change"
        elif args.parameter == "sec_uncertainty":
            param_long_name = "Uncertainty of SEC"
        elif args.parameter == "basin_id":
            param_long_name = "Glaciological Basin ID (Rignot 2016)"
        elif args.parameter == "surface_type":
            param_long_name = "Ice Surface Type"
        else:
            param_long_name = args.parameter

        sec_start_year = prod_name[48:52]
        sec_start_month = prod_name[52:54]
        sec_end_year = prod_name[55:59]
        sec_end_month = prod_name[59:61]
        print(f"sec_start_year={sec_start_year}")
        print(f"sec_start_month={sec_start_month}")
        print(f"sec_end_year={sec_end_year}")
        print(f"sec_end_month={sec_end_month}")

        with Dataset(input_file) as nc:

            lats = np.ma.filled(nc.variables["lat"][:], np.nan)
            lons = np.ma.filled(nc.variables["lon"][:], np.nan)

            if "sec" in args.parameter:
                plot_var = np.ma.filled(nc.variables[args.parameter][0][:], np.nan)
            else:
                plot_var = np.ma.filled(nc.variables[args.parameter][:], np.nan)

            long_name = nc[args.parameter].long_name
            try:
                units = nc[args.parameter].units
            except (KeyError, AttributeError):
                units = ""

            # Print numpy array dimensions: shape and number of dimensions
            print(f"lats.shape = {lats.shape}, lats.ndim = {lats.ndim}")
            print(f"lons.shape = {lons.shape}, lons.ndim = {lons.ndim}")
            print(f"plot_var.shape = {plot_var.shape}, plot_var.ndim = {plot_var.ndim}")

            # Plot parameter

            # Creating the dataset
            dataset = {
                "name": long_name,
                "units": units,
                "lats": lats,
                "lons": lons,
                "vals": plot_var,
                "plot_size_scale_factor": 0.01,
                "min_plot_range": -1.0,
                "max_plot_range": 1.0,
                "cmap_name": "RdBu",  # Colormap name, could use RdYlBu
                "cmap_over_color": "#150685",  # Optional: Over color for colormap
                "cmap_under_color": "#9E0005",  # Optional: Under color for colormap
                "cmap_extend": "both",  # Optional: Extend colormap
            }
            if "ase" in args.area:
                dataset["plot_size_scale_factor"] = 1.0
                dataset["min_plot_range"] = -2.0
                dataset["max_plot_range"] = 2.0

            if "uncertainty" in args.parameter:
                dataset["cmap_name"] = "RdBu_r"
                dataset["cmap_over_color"] = "#9E0005"
                dataset["cmap_under_color"] = "#150685"
                dataset["min_plot_range"] = 0.0
                dataset["max_plot_range"] = 0.3

            if args.parameter == "surface_type":
                dataset = {
                    "name": long_name,
                    "units": units,
                    "lats": lats,
                    "lons": lons,
                    "vals": plot_var,
                    "plot_size_scale_factor": 0.01,
                    "flag_values": [0, 1, 2, 3, 4],  # Optional: List of flag values
                    "flag_names": [
                        "Ocean",
                        "Ice-free Land",
                        "Grounded Ice",
                        "Floating Ice",
                        "Lake Vostok",
                    ],  # Optional: List of flag names
                    "flag_colors": [
                        "#F2F7FF",
                        "green",
                        "orange",
                        "yellow",
                        "red",
                    ],  # Optional: Colors for flags or colormap
                }
                if "ase" in args.area:
                    dataset["plot_size_scale_factor"] = 0.8

            if args.parameter == "basin_id":
                flag_colors = [
                    "#F2F7FF",
                    "blue",
                    "green",
                    "orange",
                    "purple",
                    "brown",
                    "pink",
                    "gray",
                    "olive",
                    "cyan",
                    "magenta",
                    "gold",
                    "navy",
                    "teal",
                    "coral",
                    "lime",
                    "indigo",
                    "maroon",
                    "orchid",
                ]

                dataset = {
                    "name": long_name,
                    "units": units,
                    "lats": lats,
                    "lons": lons,
                    "vals": plot_var,
                    "plot_size_scale_factor": 0.01,
                    "flag_values": list(range(0, 19)),
                    "flag_names": [
                        "Outside:00",
                        "West H-Hp:01",
                        "West F-G:02",
                        "East E-Ep:03",
                        "East D-Dp:04",
                        "East Cp-D:05",
                        "East B-C:06",
                        "East A-Ap:07",
                        "East Jpp-K:08",
                        "West G-H:09",
                        "East Dp-E:10",
                        "East Ap-B:11",
                        "East C-Cp:12",
                        "East K-A:13",
                        "West J-Jpp:14",
                        "Peninsula Ipp-J:15",
                        "Peninsula I-Ipp:16",
                        "Peninsula Hp-I:17",
                        "West Ep-F:18",
                    ],
                    "flag_colors": flag_colors,
                }

                if "ase" in args.area:
                    dataset["plot_size_scale_factor"] = 0.0

                print(f"min {np.nanmin(plot_var)} max {np.nanmax(plot_var)}")

            logo_image = plt.imread("ais_cci_phase2_logo.png")
            logo_width = 0.225  # in axis coordinates
            logo_height = 0.225
            logo_position = (
                -0.0,
                1 - logo_height + 0.05,
                logo_width,
                logo_height,
            )  # [left, bottom, width, height]

            xpos = 0.02
            if "ase" in args.area:
                xpos = 0.003
            ypos = 0.86
            ysep = 0.04
            xsep = 0.05

            if "ase" in args.area:
                annot = Annotation(
                    0.23,
                    0.958,
                    f"{param_long_name}",
                    {
                        "boxstyle": "round",  # Style of the box (e.g.,'round','square')
                        "facecolor": "aliceblue",  # Background color of the box
                        "alpha": 1.0,  # Transparency of the box (0-1)
                        "edgecolor": "lightgrey",  # Color of the box edge
                    },
                    18,
                    fontweight="bold",
                )
            else:

                annot = Annotation(
                    0.269,
                    0.958,
                    f"{param_long_name}",
                    {
                        "boxstyle": "round",  # Style of the box (e.g.,'round','square')
                        "facecolor": "aliceblue",  # Background color of the box
                        "alpha": 1.0,  # Transparency of the box (0-1)
                        "edgecolor": "lightgrey",  # Color of the box edge
                    },
                    18,
                    fontweight="bold",
                )
            annotation_list = [annot]

            if "ase" in args.area:
                annotation_list.append(
                    Annotation(
                        xpos + 0.05 + 0.08,
                        ypos - 0.03 + 0.016,
                        "Period start of:",
                        None,
                        10,
                    )
                )
                annotation_list.append(
                    Annotation(
                        xpos + 0.05 + 0.3,
                        ypos - 0.03 + 0.016,
                        "to end of:",
                        None,
                        10,
                    )
                )
            else:
                annotation_list.append(
                    Annotation(
                        xpos,
                        ypos + 0.01,
                        "Period start of:",
                        None,
                        10,
                    )
                )
                annotation_list.append(
                    Annotation(
                        xpos,
                        ypos - 0.08 + 0.02,
                        "Period end of:",
                        None,
                        10,
                    )
                )

            if "ase" in args.area:
                annotation_list.append(
                    Annotation(
                        xpos + 0.12 + 0.1,
                        ypos - ysep + 0.003 + 0.02,
                        f"{sec_start_month}",
                        None,
                        16,
                        fontweight="normal",
                        color="grey",
                    )
                )
                annotation_list.append(
                    Annotation(
                        xpos + xsep + 0.1 + 0.1,
                        ypos - ysep + 0.02,
                        f"{sec_start_year}",
                        None,
                        24,
                        fontweight="bold",
                    )
                )

                annotation_list.append(
                    Annotation(
                        xpos + xsep + 0.21 + 0.15,
                        ypos - ysep + 0.003 + 0.02,
                        f"{sec_end_month}",
                        None,
                        16,
                        fontweight="normal",
                        color="grey",
                    )
                )
                annotation_list.append(
                    Annotation(
                        xpos + xsep + 0.24 + 0.15,
                        ypos - ysep + 0.02,
                        f"{sec_end_year}",
                        None,
                        24,
                        fontweight="bold",
                    )
                )
            else:
                annotation_list.append(
                    Annotation(
                        xpos,
                        ypos - ysep + 0.007 + 0.01,
                        f"{sec_start_month}",
                        None,
                        16,
                        fontweight="normal",
                        color="grey",
                    )
                )
                annotation_list.append(
                    Annotation(
                        xpos + xsep - 0.02,
                        ypos - ysep + 0.01,
                        f"{sec_start_year}",
                        None,
                        24,
                        fontweight="bold",
                    )
                )

                annotation_list.append(
                    Annotation(
                        xpos,
                        ypos - ysep * 2 + 0.007 - 0.02,
                        f"{sec_end_month}",
                        None,
                        16,
                        fontweight="normal",
                        color="grey",
                    )
                )
                annotation_list.append(
                    Annotation(
                        xpos + xsep - 0.02,
                        ypos - ysep * 2 - 0.02,
                        f"{sec_end_year}",
                        None,
                        24,
                        fontweight="bold",
                    )
                )

            if "ase" in args.area:
                annotation_list.append(
                    Annotation(
                        0.23,
                        0.89,
                        f"{prod_name}",
                        None,
                        11,
                        fontweight="normal",
                    )
                )
            else:
                annotation_list.append(
                    Annotation(
                        0.261,
                        0.89,
                        f"{prod_name}",
                        None,
                        12,
                        fontweight="normal",
                    )
                )

            if "ase" in args.area:

                annotation_list.append(
                    Annotation(
                        0.07,
                        0.475,
                        "100W",
                        None,
                        8,
                        fontweight="normal",
                        color="grey",
                    )
                )
                annotation_list.append(
                    Annotation(
                        0.07,
                        0.17,
                        "110W",
                        None,
                        8,
                        fontweight="normal",
                        color="grey",
                    )
                )

                annotation_list.append(
                    Annotation(
                        0.161,
                        0.23,
                        "72S",
                        None,
                        8,
                        fontweight="normal",
                        color="grey",
                    )
                )
                annotation_list.append(
                    Annotation(
                        0.30,
                        0.29,
                        "74S",
                        None,
                        8,
                        fontweight="normal",
                        color="grey",
                    )
                )
            else:
                annotation_list.append(
                    Annotation(
                        0.237,
                        0.235,
                        "66S",
                        None,
                        8,
                        fontweight="normal",
                        color="grey",
                    )
                )
                annotation_list.append(
                    Annotation(
                        0.275,
                        0.283,
                        "70S",
                        None,
                        8,
                        fontweight="normal",
                        color="grey",
                    )
                )
                annotation_list.append(
                    Annotation(
                        0.315,
                        0.33,
                        "74S",
                        None,
                        8,
                        fontweight="normal",
                        color="grey",
                    )
                )

            area_overrides = {
                "show_bad_data_map": False,
            }

            if args.parameter == "basin_id":
                area_overrides["flag_perc_axis"] = (
                    0.8,
                    0.1,
                    0.05,
                )  # [left,bottom, width] of axis. Note height is auto set

            if args.parameter == "surface_type":
                area_overrides["flag_perc_axis"] = (
                    0.82,
                    0.4,
                    0.1,
                )  # [left,bottom, width] of axis. Note height is auto set

            Polarplot(args.area, area_overrides).plot_points(
                dataset,
                # map_only=True,
                output_file=out_file,
                use_default_annotation=False,
                annotation_list=annotation_list,
                logo_image=logo_image,
                logo_position=logo_position,
                image_format="webp",
                use_cmap_in_hist=(args.parameter != "sec"),
                dpi=75,
                webp_settings=(95, 6),
            )

            # -----------------------------------------------------------------------
            # Redo plots with hillshade
            # -----------------------------------------------------------------------

            area_overrides["apply_hillshade_to_vals"] = True

            Polarplot(args.area, area_overrides).plot_points(
                dataset,
                # map_only=True,
                output_file=f"{out_file}-hs",
                use_default_annotation=False,
                annotation_list=annotation_list,
                logo_image=logo_image,
                logo_position=logo_position,
                image_format="webp",
                use_cmap_in_hist=(args.parameter != "sec"),
                dpi=75,
                webp_settings=(95, 6),
            )

            print(f"Output: {out_file}")


if __name__ == "__main__":
    main()

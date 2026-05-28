"""
cpom.altimetry.projects.landice_portal.plot_multi_mission_dhmt.py

# Purpose

Plot parameters from multi mission dh/dt and dm/dt products

# Example

python plot_multi_mission_dhmt.py -f \
/cpnet/altimetry/landice/cpom/product_files/mass_change/\
    CPOM-AIS-L3C-DMDT-MULTIMISSION-5KM-199505-202601-fv1.nc
--outdir /tmp



"""

import argparse
import os
import re
import sys

import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from cpom.areas.area_plot import Annotation, Polarplot


def main():
    """main function for tool"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prod_filename",
        "-f",
        help="path of input multi-mission dh/dt or dm/dt file",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        "-od",
        help=("file name in output directory for output file."),
    )
    parser.add_argument(
        "--area",
        "-a",
        help="Area of interest, greenland or antarctica. Default is antarctica.",
        default="antarctica_cpom_portal",
    )

    parser.add_argument(
        "--parameter",
        "-p",
        help=("parameter to plot: dMdt, basin_id, surface_type"),
        default="dMdt",
    )

    parser.add_argument(
        "--basin",
        "-b",
        help=("basin to plot: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,..18"),
        default=None,
    )

    if len(sys.argv) == 1:
        print("no args provided")
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    if "DHDT" in args.prod_filename:
        args.parameter = "sec"

    input_files = [args.prod_filename]

    output_dir = args.outdir
    if args.basin:
        output_dir += f"/basin{args.basin}"
    if not output_dir:
        output_dir = os.path.dirname(input_files[0])

    for input_file in input_files:
        prod_name = os.path.basename(input_file)

        # Strip dates YYYYMM-YYYYMM and version suffix like -fv1
        clean_name = re.sub(r"-\d{6}-\d{6}(-fv\d+)?", "", prod_name.replace(".nc", ""))
        if args.parameter in ("dMdt", "sec"):
            out_file = f"{output_dir}/{clean_name}"
            if args.basin:
                out_file += f"-basin{args.basin}"
        else:
            out_file = f"{output_dir}/{clean_name}-{args.parameter}"

        if args.parameter == "dMdt":
            param_long_name = "Ice Sheet Mass Change Rate"
        elif args.parameter == "sec":
            param_long_name = "Ice Sheet Surface Elevation Change Rate"
        elif args.parameter == "basin_id":
            param_long_name = "Glaciological Basin ID (Rignot 2016)"
        elif args.parameter == "surface_type":
            param_long_name = "Ice Surface Type"
        else:
            param_long_name = args.parameter

        # Try to find dates YYYYMM-YYYYMM using regex
        date_match = re.search(r"(\d{6})-(\d{6})", prod_name)
        if date_match:
            start_date = date_match.group(1)
            end_date = date_match.group(2)
            start_year = start_date[:4]
            start_month = start_date[4:]
            end_year = end_date[:4]
            end_month = end_date[4:]
        else:
            # Fallback
            start_year = prod_name[48:52]
            start_month = prod_name[52:54]
            end_year = prod_name[55:59]
            end_month = prod_name[59:61]

        months_dict = {
            "01": "Jan",
            "02": "Feb",
            "03": "Mar",
            "04": "Apr",
            "05": "May",
            "06": "Jun",
            "07": "Jul",
            "08": "Aug",
            "09": "Sep",
            "10": "Oct",
            "11": "Nov",
            "12": "Dec",
        }
        start_month_str = months_dict.get(start_month, start_month)
        end_month_str = months_dict.get(end_month, end_month)
        print(f"Start: {start_month_str} {start_year}, End: {end_month_str} {end_year}")

        map_only = True
        figure_width = 10
        figure_height = 10

        with Dataset(input_file) as nc:
            lats = np.ma.filled(nc.variables["lat"][:], np.nan)
            lons = np.ma.filled(nc.variables["lon"][:], np.nan)

            # Slicing: dynamic handling of 2D (ny, nx) vs 3D (time, ny, nx) arrays
            var_data = nc.variables[args.parameter]
            plot_data = var_data[0][:] if var_data.ndim == 3 else var_data[:]
            if isinstance(plot_data, np.ma.MaskedArray):
                plot_var = plot_data.astype(float).filled(np.nan)
            else:
                plot_var = np.array(plot_data, dtype=float)

            long_name = nc[args.parameter].long_name
            try:
                units = nc[args.parameter].units
            except (KeyError, AttributeError):
                units = ""

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

            if args.parameter == "dMdt":
                dataset["min_plot_range"] = -500.0
                dataset["max_plot_range"] = 500.0

            if args.parameter == "sec":
                dataset["min_plot_range"] = -1.0
                dataset["max_plot_range"] = 1.0

            if args.basin:
                dataset["plot_size_scale_factor"] = 2.0
                dataset["min_plot_range"] = -2000.0
                dataset["max_plot_range"] = 2000.0
                map_only = False
                figure_width = 12
                figure_height = 10
                if args.basin == "5":
                    dataset["min_plot_range"] = -400.0
                    dataset["max_plot_range"] = 400.0

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

                print(f"min {np.nanmin(plot_var)} max {np.nanmax(plot_var)}")

            xpos = 0.02
            ypos = 0.85
            ysep = 0.036

            # Calculate spanned years dynamically

            if args.basin:
                annot = Annotation(
                    0.2,
                    0.91,
                    f"{param_long_name}: Rignot Basin {args.basin}",
                    {
                        "boxstyle": "round",  # Style of the box (e.g.,'round','square')
                        "facecolor": "aliceblue",  # Background color of the box
                        "alpha": 1.0,  # Transparency of the box (0-1)
                        "edgecolor": "lightgrey",  # Color of the box edge
                    },
                    15,
                    fontweight="bold",
                )
                annotation_list = [annot]

                xpos = 0.02
                ypos = 0.98
                ysep = 0.03

                annotation_list.append(
                    Annotation(
                        xpos,
                        -0.01 + ypos,
                        "Period start of:",
                        None,
                        9,
                    )
                )
                annotation_list.append(
                    Annotation(
                        xpos,
                        ypos - ysep,
                        f"{start_month_str} {start_year}",
                        None,
                        14,
                        fontweight="bold",
                    )
                )

                annotation_list.append(
                    Annotation(
                        xpos,
                        ypos - ysep * 2,
                        "Period end of:",
                        None,
                        9,
                    )
                )

                annotation_list.append(
                    Annotation(
                        xpos,
                        0.01 + ypos - ysep * 3,
                        f"{end_month_str} {end_year}",
                        None,
                        14,
                        fontweight="bold",
                    )
                )
            else:
                annot = Annotation(
                    0.344,
                    0.958,
                    f"{param_long_name}",
                    {
                        "boxstyle": "round",  # Style of the box (e.g.,'round','square')
                        "facecolor": "aliceblue",  # Background color of the box
                        "alpha": 1.0,  # Transparency of the box (0-1)
                        "edgecolor": "lightgrey",  # Color of the box edge
                    },
                    15,
                    fontweight="bold",
                )
                annotation_list = [annot]

                annotation_list.append(
                    Annotation(
                        xpos,
                        -0.01 + ypos,
                        "Period start of:",
                        None,
                        10,
                    )
                )
                annotation_list.append(
                    Annotation(
                        xpos,
                        ypos - ysep,
                        f"{start_month_str} {start_year}",
                        None,
                        16,
                        fontweight="bold",
                    )
                )

                annotation_list.append(
                    Annotation(
                        xpos,
                        ypos - ysep * 2,
                        "Period end of:",
                        None,
                        10,
                    )
                )

                annotation_list.append(
                    Annotation(
                        xpos,
                        0.01 + ypos - ysep * 3,
                        f"{end_month_str} {end_year}",
                        None,
                        16,
                        fontweight="bold",
                    )
                )

                annotation_list.append(
                    Annotation(
                        0.246,
                        0.08,
                        f"{prod_name}",
                        None,
                        10,
                        fontweight="normal",
                    )
                )

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

            thisarea = args.area
            if args.basin:
                thisarea = f"rignot_basin_{args.basin}"

            Polarplot(thisarea, area_overrides).plot_points(
                dataset,
                map_only=map_only,
                output_file=out_file,
                use_default_annotation=False,
                annotation_list=annotation_list,
                image_format="webp",
                use_cmap_in_hist=(args.parameter != "dMdt"),
                dpi=150,
                webp_settings=(95, 6),
                figure_width=figure_width,
                figure_height=figure_height,
                facecolor="#f0f0f0",
            )

            # -----------------------------------------------------------------------
            # Redo plots with hillshade
            # -----------------------------------------------------------------------

            area_overrides["apply_hillshade_to_vals"] = True

            Polarplot(thisarea, area_overrides).plot_points(
                dataset,
                map_only=map_only,
                output_file=f"{out_file}-HS",
                use_default_annotation=False,
                annotation_list=annotation_list,
                image_format="webp",
                use_cmap_in_hist=(args.parameter != "dMdt"),
                dpi=150,
                webp_settings=(95, 6),
                figure_width=figure_width,
                figure_height=figure_height,
                facecolor="#f0f0f0",
            )

            print(f"Output: {out_file}")


if __name__ == "__main__":
    main()

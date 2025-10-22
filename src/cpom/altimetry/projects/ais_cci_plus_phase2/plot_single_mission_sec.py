"""
cpom.altimetry.projects.ais_cci_plus_phase2.plot_single_mission_sec.py

# Purpose

Plot dh/dt from AIS CCI+ phase-2 single mission dh/dt products

# Example

plot_single_mission_sec.py -f \
    ~/Downloads/\
        ESACCI-AIS-L3C-SEC-CS2-5KM-20100927-20241203-fv2.nc \
          -o /tmp

"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from cpom.areas.area_plot import Annotation, Polarplot


def main():
    """main function for tool"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prod_filename", "-f", help="path of input multi-mission dhdt.npz file", required=True
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
        default="antarctica",
    )

    parser.add_argument(
        "--parameter",
        "-p",
        help=("parameter to plot: sec, sec_uncertainty, basin_id, surface_type"),
        default="sec",
    )

    if len(sys.argv) == 1:
        print("no args provided")
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    # Extract CCI netcdf file: example: ESACCI-AIS-L3C-SEC-CS2-5KM-20100927-20250909-fv2.nc

    output_dir = args.outdir
    if not output_dir:
        output_dir = os.path.dirname(args.prod_filename)

    prod_name = os.path.basename(args.prod_filename)

    out_file = f"{output_dir}/{prod_name.replace('.nc',f'-{args.parameter}.png')}"

    if args.parameter == "sec":
        param_long_name = "Surface Elevation Change"
    elif args.parameter == "sec_uncertainty":
        param_long_name = "Uncertainty of SEC"
    elif args.parameter == "basin_id":
        param_long_name = "Glaciological Basin ID (Rignot 2016)"
    elif args.parameter == "surface_type":
        param_long_name = "Ice Surface Type"
    else:
        param_long_name = args.parameter

    start_year = prod_name[27:31]
    start_month = prod_name[31:33]
    print(f"{start_month} {start_year}")
    end_year = prod_name[36:40]
    end_month = prod_name[40:42]
    print(f"{end_month} {end_year}")
    mission_str = prod_name[19:22]
    print(f"mission_str {mission_str}")

    if mission_str == "CS2":
        mission_name = "CryoSat-2"
    elif mission_str == "S3A":
        mission_name = "Sentinel-3A"
    elif mission_str == "S3B":
        mission_name = "Sentinel-3B"
    elif mission_str == "ER1":
        mission_name = "ERS-1"
    elif mission_str == "ER2":
        mission_name = "ERS-2"
    elif mission_str == "ENV":
        mission_name = "ENVISAT"
    elif mission_str == "IS2":
        mission_name = "ICESat-2"
    else:
        mission_name = f"{mission_str}"

    with Dataset(args.prod_filename) as nc:
        # start_date = str(nc.data_start_time)
        # start_month = int(start_date[5:7])
        # start_year = int(start_date[0:4])
        # end_date = str(nc.data_end_time)
        # end_month = int(end_date[5:7])
        # end_year = int(end_date[0:4])
        sec_period_length = str(nc.sec_period_length).replace("yrs", "")

        lats = np.ma.filled(nc.variables["lat"][:], np.nan)
        lons = np.ma.filled(nc.variables["lon"][:], np.nan)

        plot_var = np.ma.filled(nc.variables[args.parameter][:], np.nan)

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
            "cmap_name": "RdYlBu",  # Optional: Colormap name
            "cmap_over_color": "#150685",  # Optional: Over color for colormap
            "cmap_under_color": "#9E0005",  # Optional: Under color for colormap
            "cmap_extend": "both",  # Optional: Extend colormap
        }

        if args.parameter == "sec_uncertainty":
            dataset["cmap_name"] = "RdYlBu_r"
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

        logo_image = plt.imread("ais_cci_phase2_logo.png")
        logo_width = 0.23  # in axis coordinates
        logo_height = 0.23
        logo_position = (
            -0.0,
            1 - logo_height + 0.03,
            logo_width,
            logo_height,
        )  # [left, bottom, width, height]

        xpos = 0.74
        ypos = 0.89
        ysep = 0.032

        annot = Annotation(
            xpos,
            ypos,
            "Period start of:",
            None,
            10,
        )
        annotation_list = [annot]

        annotation_list.append(
            Annotation(
                xpos,
                ypos + ysep + 0.03,
                "Mission: ",
                None,
                12,
                fontweight="normal",
            )
        )
        annotation_list.append(
            Annotation(
                xpos,
                ypos + ysep + 0.001,
                f"{mission_name}",
                None,
                18,
                fontweight="bold",
            )
        )

        annotation_list.append(
            Annotation(
                xpos,
                ypos - ysep,
                f"{start_month} {start_year}",
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
                ypos - ysep * 4,
                "Duration (yrs):",
                None,
                10,
                fontweight="normal",
            )
        )
        annotation_list.append(
            Annotation(
                xpos,
                ypos - ysep * 5,
                f"{sec_period_length}",
                None,
                16,
                fontweight="bold",
            )
        )

        annotation_list.append(
            Annotation(
                xpos,
                ypos - ysep * 3,
                f"{end_month} {end_year}",
                None,
                16,
                fontweight="bold",
            )
        )

        annotation_list.append(
            Annotation(
                0.28,
                0.94,
                param_long_name,
                {
                    "boxstyle": "round",  # Style of the box (e.g.,'round','square')
                    "facecolor": "aliceblue",  # Background color of the box
                    "alpha": 1.0,  # Transparency of the box (0-1)
                    "edgecolor": "lightgrey",  # Color of the box edge
                },
                18,
                fontweight="bold",
            )
        )

        annotation_list.append(
            Annotation(
                0.275,
                0.9,
                f"Product: {prod_name}",
                None,
                10,
                fontweight="normal",
            )
        )
        annotation_list.append(
            Annotation(
                0.275,
                0.87,
                f"NetCDF parameter: {args.parameter}",
                None,
                10,
                fontweight="normal",
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

        Polarplot(args.area, area_overrides).plot_points(
            dataset,
            # map_only=True,
            output_file=out_file,
            use_default_annotation=False,
            annotation_list=annotation_list,
            logo_image=logo_image,
            logo_position=logo_position,
        )

        area_overrides["apply_hillshade_to_vals"] = True

        Polarplot(args.area, area_overrides).plot_points(
            dataset,
            # map_only=True,
            output_file=out_file.replace(".png", "-hs.png"),
            use_default_annotation=False,
            annotation_list=annotation_list,
            logo_image=logo_image,
            logo_position=logo_position,
        )

        print(f"Output: {out_file}")


if __name__ == "__main__":
    main()

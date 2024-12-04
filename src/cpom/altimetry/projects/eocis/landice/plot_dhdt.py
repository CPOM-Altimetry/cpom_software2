"""
cpom.altimetry.projects.eocis.landice.plot_dhdt.py

# Purpose

Plot EOCIS dh/dt products

# Example

plot_dhdt.py -f \
    ~/Sites/landice_portal/test_data/\
        EOCIS-AIS-L3C-SEC-MULTIMISSION-5KM-5YEAR-MEANS-199101-199601-fv1.nc \
          -o /tmp

"""

import argparse
import sys

import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from cpom.areas.area_plot import Annotation, Polarplot


def main():
    """main function for tool"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dhdt_filename", "-f", help="path of input multi-mission dhdt.npz file", required=True
    )

    parser.add_argument(
        "--outfile",
        "-of",
        help=("file name in output directory for output file."),
        required=True,
    )
    parser.add_argument(
        "--area",
        "-a",
        help="Area of interest, greenland or antarctica. Default is antarctica.",
        default="antarctica",
    )

    parser.add_argument("--parameter", "-p", help="parameter to plot", default="sec")

    parser.add_argument(
        "--hillshade",
        "-hs",
        help="apply hillshade to data",
        required=False,
        action="store_true",
    )

    if len(sys.argv) == 1:
        print("no args provided")
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    # Read dhdt file

    with Dataset(args.dhdt_filename) as nc:
        start_date = str(nc.data_start_time)
        start_month = int(start_date[5:7])
        start_year = int(start_date[0:4])
        end_date = str(nc.data_end_time)
        end_month = int(end_date[5:7])
        end_year = int(end_date[0:4])

        print(f"{start_month} {start_year}")
        print(f"{end_month} {end_year}")

        lats = nc.variables["lat"][:].data
        lons = nc.variables["lon"][:].data

        finite_mask = np.isfinite(lats)
        lats = lats[finite_mask]
        lons = lons[finite_mask]

        plot_var = nc.variables[args.parameter][:].data[0, :, :]  # has extra dim of time
        plot_var = plot_var[finite_mask]

        long_name = nc[args.parameter].long_name
        try:
            units = nc[args.parameter].units
        except KeyError:
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
        }

        area_overrides = {
            "apply_hillshade_to_vals": args.hillshade,
        }

        if "antarctic" in args.area:
            xpos = 0.02
            ypos = 0.85
            ysep = 0.03
        else:
            xpos = 0.05
            ypos = 0.8
            ysep = 0.03

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
                ypos - ysep,
                f"{start_month:02d} {start_year}",
                None,
                18,
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
                ypos - ysep * 3,
                f"{end_month:02d} {end_year}",
                None,
                18,
                fontweight="bold",
            )
        )

        annotation_list.append(
            Annotation(
                0.33,
                0.96,
                "surface elevation change (m/yr)",
                {
                    "boxstyle": "round",  # Style of the box (e.g.,'round','square')
                    "facecolor": "aliceblue",  # Background color of the box
                    "alpha": 1.0,  # Transparency of the box (0-1)
                    "edgecolor": "lightgrey",  # Color of the box edge
                },
                14,
                fontweight="normal",
            )
        )

        Polarplot(args.area, area_overrides).plot_points(
            dataset,
            map_only=True,
            output_file=args.outfile,
            use_default_annotation=False,
            annotation_list=annotation_list,
        )

        print(f"Output: {args.outfile}")


if __name__ == "__main__":
    main()

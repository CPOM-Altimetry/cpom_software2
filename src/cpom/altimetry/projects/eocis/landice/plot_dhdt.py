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

from cpom.areas.area_plot import Polarplot


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
        # date = str(nc.data_start_time)
        # month = int(date[5:7])
        # year = int(date[0:4])
        # title = "Elevation change gridded 5-year mean"

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

        Polarplot(args.area, area_overrides).plot_points(
            dataset, map_only=True, output_file=args.outfile
        )


if __name__ == "__main__":
    main()

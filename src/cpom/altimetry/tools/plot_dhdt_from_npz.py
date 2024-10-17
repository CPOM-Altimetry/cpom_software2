#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" plot dhdt.npz files from CPOM sec processing (not a finished tool)

    using black background
    to highlight missing areas of the grid

    Note, just supports greenland and antarctica grid areas at the moment.
    Doesn't read the dhdt.info file

    Need to edit dataset name string, plot range
"""
import argparse

import numpy as np

from cpom.areas.area_plot import Polarplot
from cpom.gridding.gridareas import GridArea

# pylint: disable=too-many-locals


def main():
    """main function of tool

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="plot dhdt.npz files from sec processing.")
    parser.add_argument("--file", "-f", help=("file name of dhdt.npz"), required=False)
    parser.add_argument(
        "--greenland",
        "-g",
        help=("set for greenland grid area and plot instead of antarctica"),
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--plot_name",
        "-n",
        default="dh/dt",
        help=("plot name to use in annotation"),
        required=False,
    )

    parser.add_argument(
        "--plot_range",
        "-pr",
        default=0.5,
        help=("absolute value of plot range to use for dh/dt plots"),
        required=False,
        type=float,
    )

    args = parser.parse_args()

    dhdt_filename = args.file

    g_area = "antarctica"
    if args.greenland:
        g_area = "greenland"
    thisgridarea = GridArea(g_area, 5000)

    npdata = np.load(dhdt_filename, allow_pickle=True)

    dhdt_grid = npdata.get("dhdt")

    ncols = np.shape(dhdt_grid)[1]
    nrows = np.shape(dhdt_grid)[0]

    print("Forming lat,lon,dhdt lists as plot input...")

    # form lists of lat[],lon[],dhdt[]
    # First form x[],y[],dhdt[], and then transform x,y to lat,lon
    dhdt = []
    x = []
    y = []
    lats = []
    lons = []
    for col in range(ncols):
        for row in range(nrows):
            if np.isfinite(dhdt_grid[row, col]):
                dhdt.append(dhdt_grid[row, col])
                thisx, thisy = thisgridarea.get_cellcentre_x_y_from_col_row(col, row)
                x.append(thisx)
                y.append(thisy)

    if len(x) > 0:
        lats, lons = thisgridarea.transform_x_y_to_lat_lon(x, y)

    data_set = {
        "name": args.plot_name,
        "lats": lats,
        "lons": lons,
        "vals": dhdt,
        "units": "m/yr",
        "min_plot_range": args.plot_range * -1,
        "max_plot_range": args.plot_range,
        "cmap_name": "RdYlBu",  # colormap name to use for this dataset
        "cmap_under_color": "#A85754",
        "cmap_over_color": "#3E4371",
        "plot_size_scale_factor": 0.1,
    }

    plot_area = "antarctica_is"
    if args.greenland:
        plot_area = "greenland_is"
    Polarplot(
        plot_area,
        area_overrides={
            "background_image": "basic_land_black",
        },
    ).plot_points(data_set)


if __name__ == "__main__":
    main()

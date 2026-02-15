#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for irregular data gridding functions optimised for typical satellite data uses

Author: Alan Muir (CPOM/UCL)
Date: 2019
Copyright: UCL/MSSL/CPOM. Not to be used outside CPOM/MSSL without permission of author

"""

import sys
from math import floor

# Package Imports
import numpy as np

# pylint: disable=too-many-branches,too-many-locals,too-many-arguments
# ----------------------------------------------------------------------------------------------
# 	griddata
# 		bin irregular data (x,y,vals) in to a 2d rectangular grid of cell size: binsize in meters.
# 		grid coordinates defined by x,y extrema : origin xmin-binsize/2., ymin - binsize/2.
# 		ie xmin,ymin is in center of 1st bin (not edge).
# 		Calculates median (or mean or %failure) of binned values. Nans are allowed but excluded
#       from calculation.
#
# 		returns: 2d grid of calculated (median, mean or %failure)
# ----------------------------------------------------------------------------------------------


def griddata(
    x,
    y,
    z,
    binsize=10000,
    calc_median=False,
    calc_mean=False,
    calc_percent_of_value=False,
    return_density=False,
):
    """
    bin irregular data (x,y,vals) in to a 2d rectangular grid of cell size: binsize in meters.
    grid coordinates defined by x,y extrema : origin xmin-binsize/2., ymin - binsize/2.
    ie xmin,ymin is in center of 1st bin (not edge).
    Calculates median (or mean or %value) of binned values.
    Nans are allowed but excluded from calculation.
    Empty bins are set to np.nan

    Parameters
    ----------
    x : ndarray (1D)
        float, x values (in meters).
    y : ndarray (1D)
        float, y values (in meters).
    z : ndarray (1D)
        corresponding data values.
    binsize : scalar, optional, units meters
        The full width and height of each bin on the grid.  If each
        bin is a cube, then this is the x and y dimension.  This is
        the step in both directions, x and y. Defaults to 10000 (ie 10km)
    calc_median : boolean, default=False. Calculate the median value in each bin
    as return grid value
    calc_mean : boolean, default=False. Calculate the mean value in each bin
    as return grid value
    calc_percent_of_value : a scalar value or boolean False, default=False.
    Calculate the % of this value in each bin as return grid value

    Returns
    -------
    grid : ndarray (2D)
        The evenly gridded data.  The value of each cell is the median,
        mean or %failure value of
        the contents of the bin.
    density_grid : ndarray (2D) of int, optional
        xi, yi: the mesh of x,y coordinates for new grid
         grid of measurement density (number of measurements in each cell)
         only returned if return_density==True

    Revisions
    ---------
    2019-11-04  Initial version

    """

    if not calc_mean and not calc_median and not calc_percent_of_value:
        sys.exit("griddata: Must set either calc_mean, calc_median, calc_percent_of_value to True")

        # get extrema values of input x,y coords
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # make x,y coordinate arrays.
    xi = np.arange(xmin - binsize / 2.0, xmax + binsize / 2.0, binsize)
    yi = np.arange(ymin - binsize / 2.0, ymax + binsize / 2.0, binsize)
    ximin = xmin - binsize / 2.0
    yimin = ymin - binsize / 2.0
    xi, yi = np.meshgrid(xi, yi)

    # create the output grid.
    grid = np.full(xi.shape, np.nan, dtype=x.dtype)
    if return_density:
        density_grid = np.full(xi.shape, np.nan, dtype=x.dtype)
    nrow, ncol = grid.shape

    # Go through the data and fill in the grid

    # Create a grid of empty lists (ie each index contains [None])
    bins = [[[None] for x in range(ncol)] for y in range(nrow)]

    if calc_median:
        print("filling grid with median values")
    if calc_mean:
        print("filling grid with mean values")
    if calc_percent_of_value:
        print("filling grid with % of ", calc_percent_of_value)

    # Fill the grid with input data points z
    for i in range(z.size):
        xtmp = x[i]
        ytmp = y[i]
        grid_x_pos = floor((xtmp - ximin) / binsize)
        grid_y_pos = floor((ytmp - yimin) / binsize)

        if not bins[grid_y_pos][grid_x_pos][0]:
            bins[grid_y_pos][grid_x_pos][0] = z[i]
        else:
            bins[grid_y_pos][grid_x_pos].append(z[i])

    # Calculate the grid statistic to be output (ie mean. median, % of value)
    for row in range(nrow):
        for col in range(ncol):
            if bins[row][col][0] is None:
                grid[row, col] = np.nan
                continue
            if calc_median:
                grid[row, col] = np.nanmedian(bins[row][col])
            if calc_mean:
                grid[row, col] = np.nanmean(bins[row][col])
            if calc_percent_of_value:
                vals = np.array(bins[row][col])
                w = np.where(vals == calc_percent_of_value)[0]
                grid[row, col] = 100.0 * w.size / vals.size
            if return_density:
                density_grid[row, col] = len(bins[row][col])
                # print(row, col, density_grid[row,col])
    if return_density:
        return grid, xi, yi, density_grid
    return grid, xi, yi

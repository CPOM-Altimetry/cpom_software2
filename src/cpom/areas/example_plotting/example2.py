"""cpom.areas.example_plotting.example1.py

# Purpose

Basic example of plotting using Polarplot

We will plot 2 separate data sets on a hill-shaded map
of Greenland which includes a grounded ice dataset mask

For the second dataset we will specify a different
colormap to plot with ('viridis'), and turn off the
dataset ground ice masking

"""

import numpy as np

from cpom.areas.area_plot import Polarplot

# Create 1D coordinate vectors

# Build the 2D grid
lon_grid, lat_grid = np.meshgrid(
    np.arange(-75.0, -10.0 + 0.1, 0.1), np.arange(59.5, 83.9 + 0.1, 0.1)
)

# Prepare 1st dataset. It uses the default colourmap
dataset1 = {
    "lats": lat_grid.ravel(),  # latitude values in deg N
    "lons": lon_grid.ravel(),  # longitude values in deg E
    "vals": np.linspace(start=0, stop=100, num=len(lat_grid.ravel())),  # data values
    "name": "dataset1",  # optional name of data set, appears as a plot label
}

# Plot both datasets over Greenland using area 'greenland_hs_is'
# This area includes area masking from bedmachine
# area names can be any string, but the convention used here
# is that 'hs' stands for use hill shading of the underlying basemap, and
# 'is' stands for ice sheet (ie grounded ice only) masking of dataset locations
Polarplot("greenland_hs_is").plot_points(dataset1, map_only=True, output_file="/tmp/example2.png")

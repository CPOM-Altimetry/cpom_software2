"""cpom.areas.example_plotting.example1.py

# Purpose

Basic example of plotting using Polarplot

We will plot 2 separate data sets on a hill-shaded map
of Greenland. For the second dataset we will specify a different
colormap to plot with ('viridis')

"""

import numpy as np

from cpom.areas.area_plot import Polarplot

# Prepare dataset.
dataset = {
    "lats": np.arange(-85, -74, 0.1),  # latitude values in deg N
    "lons": np.linspace(start=170, stop=180, num=110),  # longitude values in deg E
    "vals": np.linspace(start=0, stop=100, num=110),  # data values
}

# Plot both datasets over the Ross Ice Shelf using area 'ross_iceshelf_hs_fi'
# This area includes area masking from bedmachine
# area names can be any string, but the convention used here
# is that 'hs' stands for use hill shading of the underlying basemap, and
# 'fi' stands for floating ice (ie ice shelves only) masking of dataset locations
Polarplot("ross_iceshelf_hs_fi").plot_points(dataset, output_file="/tmp/example1.png")

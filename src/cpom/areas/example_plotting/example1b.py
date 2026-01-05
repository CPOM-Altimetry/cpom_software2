"""cpom.areas.example_plotting.example1.py

# Purpose

Basic example of plotting using Polarplot

We will plot 2 separate data sets on a hill-shaded map
of Greenland. For the second dataset we will specify a different
colormap to plot with ('viridis')

"""

import numpy as np

from cpom.areas.area_plot import Polarplot

dataset1 = {
    "lats": np.arange(-85, -74, 0.1),  # latitude values in deg N
    "lons": np.linspace(start=180, stop=190, num=110),  # longitude values in deg E
    "vals": np.linspace(start=0, stop=100, num=110),  # data values
    "name": "dataset2",  # optional name of data set, appears as a plot label
    "apply_area_mask_to_data": False,  # Optional: Apply area mask to data
    "cmap_name": "viridis",  # Optional: Colormap name
    "valid_range": [0, 80],  # Optional: only plot values in this range
    "plot_size_scale_factor": 3.0,  # Optional: Marker size scale factor
}

# Prepare 1st dataset. It uses the default colourmap
dataset2 = {
    "lats": np.arange(-85, -74, 0.1),  # latitude values in deg N
    "lons": np.linspace(start=170, stop=180, num=110),  # longitude values in deg E
    "vals": np.linspace(start=0, stop=100, num=110),  # data values
    "name": "dataset1",  # optional name of data set, appears as a plot label
}

# Plot both datasets over the Ross Ice Shelf using area 'ross_iceshelf_hs_fi'
# This area includes area masking from bedmachine
# area names can be any string, but the convention used here
# is that 'hs' stands for use hill shading of the underlying basemap, and
# 'fi' stands for floating ice (ie ice shelves only) masking of dataset locations
Polarplot("ross_iceshelf_hs_fi").plot_points(
    *(dataset1, dataset2), output_file="/tmp/example1b.png"  # plot both datasets
)

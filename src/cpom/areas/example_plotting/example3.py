"""cpom.areas.example_plotting.example1.py

# Purpose

Basic example of plotting using Polarplot

We will plot 2 separate data sets on a hill-shaded map
of Greenland. For the second dataset we will specify a different
colormap to plot with ('viridis')

"""

import numpy as np

from cpom.areas.area_plot import Annotation, Polarplot

# Build the 2D grid
STEP = 0.5
lon_grid, lat_grid = np.meshgrid(
    np.arange(-180.0, 180.0 + STEP, STEP), np.arange(-90.0, -61.0 + STEP, STEP)
)

# Prepare 1st dataset. It uses the default colourmap
dataset = {
    "lats": lat_grid.ravel(),  # latitude values in deg N
    "lons": lon_grid.ravel(),  # longitude values in deg E
    "vals": np.linspace(start=0, stop=100, num=len(lat_grid.ravel())),  # data values
    "name": "dataset1",  # optional name of data set, appears as a plot label
}

# Add some custom text annotations
annotation_list = [
    Annotation(
        0.03,
        0.93,
        "My Custom Annotation",
        {
            "boxstyle": "round",  # Style of the box (e.g.,'round','square')
            "facecolor": "aliceblue",  # Background color of the box
            "alpha": 1.0,  # Transparency of the box (0-1)
            "edgecolor": "lightgrey",  # Color of the box edge
        },
        18,
        fontweight="bold",
    ),
    Annotation(
        0.03,
        0.86,
        "Another custom annotation",
        None,
        10,
    ),
]


# Plot both datasets over the Ross Ice Shelf using area 'ross_iceshelf_hs_fi'
# This area includes area masking from bedmachine
# area names can be any string, but the convention used here
# is that 'hs' stands for use hill shading of the underlying basemap, and
# 'fi' stands for floating ice (ie ice shelves only) masking of dataset locations
Polarplot("antarctica_hs_is").plot_points(
    dataset,  # plot this dataset
    use_default_annotation=False,
    annotation_list=annotation_list,
    output_file="/tmp/example3.png",
)

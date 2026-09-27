"""
# Area definition

## Summary:
Based on area: global
Global area used by the CryoSat-2 CSQA performance monitoring service

Layout (figure fractions) differs from the global area to give a larger map: the map uses 90%
of the figure width, with the annotations and bad data mini-map above it, and the colour bar,
statistics, histograms and latitude scatter plot (or flag percentages) below it.
"""

area_definition = {
    "use_definitions_from": "global",
    "long_name": "Global",
    # --------------------------------------------
    # Main map: full width (the 2:1 map is 0.9 x 0.54 of the 12 x 10 inch figure)
    # --------------------------------------------
    "axes": [  # define plot axis position
        0.05,  # left
        0.25,  # bottom
        0.90,  # width (axes fraction)
        0.54,  # height (axes fraction)
    ],
    # --------------------------------------------
    # Annotations above the map
    # --------------------------------------------
    "area_long_name_position": (0.05, 0.825),  # area name above the map's left end
    "mask_long_name_position": (0.14, 0.829),  # right of the area name
    # --------------------------------------------
    # Bad data mini-map: top right, above the map
    # --------------------------------------------
    "bad_data_minimap_axes": [  # define minimap axis position
        0.70,  # left
        0.835,  # bottom
        0.25,  # width (axes fraction)
        0.14,  # height (axes fraction)
    ],
    # legend to the left of the mini-map, below the plot title lines
    "bad_data_minimap_legend_pos": (-0.02, 0.5),
    # --------------------------------------------
    # Below the map: colour bar and statistics (left), histograms and latitude scatter (right)
    # --------------------------------------------
    "horizontal_colorbar_axes": [
        0.10,  # left
        0.165,  # bottom
        0.36,  # width
        0.02,  # height
    ],
    "histogram_plotrange_axes": [
        0.58,  # left
        0.05,  # bottom
        0.07,  # width (axes fraction)
        0.14,  # height (axes fraction)
    ],
    "histogram_fullrange_axes": [
        0.72,  # left
        0.05,  # bottom
        0.07,  # width (axes fraction)
        0.14,  # height (axes fraction)
    ],
    "latvals_axes": [
        0.85,  # left
        0.05,  # bottom
        0.12,  # width (axes fraction)
        0.14,  # height (axes fraction)
    ],
    # flag percentages below the map ([left, bottom, width], height is set by the flag count)
    "flag_perc_axis": [0.42, 0.07, 0.2],
}

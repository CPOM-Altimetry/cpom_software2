"""
# Area definition

## Summary:
Based on area: antarctica
**Data mask: grounded ice from bedmachine v2**

"""

area_definition = {
    "long_name": "Antarctic Ocean",
    "area_summary": "Antarctic Ocean",
    "use_definitions_from": "antarctica",
    # --------------------------------------------
    #    mask from cpom.masks.Mask
    # --------------------------------------------
    "apply_area_mask_to_data": True,  # filter data using areas cpom.masks.Mask
    "maskname": "antarctica_bedmachine_v2_grid_mask",  # from  cpom.masks.Mask
    "masktype": "grid",
    "basin_numbers": [1],  # [n1,n2,..] if mask allows basin numbers
    # for bedmachine v2, 2=grounded ice, 3=floating, 4=vostok
    "show_polygon_mask": False,  # show mask polygon
    "polygon_mask_color": "red",  # color to draw mask polygon
    "show_bad_data_map": False,
    # --------------------------------------------
    # Data filtering using lat/lon extent (used as a quick data pre-filter before masking)
    # --------------------------------------------
    #   Area min/max lat/lon for initial data filtering
    "minlat": -86.0,  # minimum latitude to initially filter records for area
    "maxlat": -54.0,  # maximum latitude to initially filter records for area
    "bounding_lat": -54.0,  # limiting latitude for round areas or None
    # --------------------------------------------
    # Plot parameters for this area
    # --------------------------------------------
    "axes": [  # define plot axis position
        0.04,  # left
        0.12,  # bottom
        0.757,  # width (axes fraction)
        0.757,  # height (axes fraction)
    ],
    "histogram_plotrange_axes": [
        0.87,  # left
        0.12,  # bottom
        0.07,  # width (axes fraction)
        0.31,  # height (axes fraction)
    ],  # axis location of plot range histogram for Polarplot.plot_points()
    "histogram_fullrange_axes": [
        0.87,  # left
        0.47,  # bottom
        0.07,  # width (axes fraction)
        0.31,  # height (axes fraction)
    ],  # axis location of plot range histogram for Polarplot.plot_points()
    "show_latitude_scatter": False,  # Whether to show the latitude scatter plot
    "latitude_of_radial_labels": -49.4,  # latitude for radial grid line labels for circular plots
    "horizontal_colorbar_axes": [
        0.15,
        0.07,
        0.55,
        0.02,
    ],  # [ left, bottom, width, height (fractions of axes)]
    "stats_position_x_offset": -0.07,  # x offset to stats position
    "stats_position_y_offset": 0.02,  # y offset to stats position
    "mapscale": [
        -179.9,  # longitude to position scale bar
        -56.0,  # latitide to position scale bar
        0.0,  # longitude of true scale (ie centre of area)
        -90.0,  # latitude of true scale (ie centre of area)
        1000,  # width of scale bar (km)
        "black",  # color of scale bar
        70,  # size of scale bar
    ],
    "latitude_gridlines": [-60, -70, -80],  # deg N
}

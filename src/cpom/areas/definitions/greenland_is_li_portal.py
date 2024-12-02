"""
# Area definition

## Summary:
Based on area: greenland
**Data mask: grounded ice from bedmachine v3**

"""

area_definition = {
    "area_summary": "Greenland [grounded ice mask]",
    "use_definitions_from": "greenland",
    # --------------------------------------------
    #    mask from cpom.masks.Mask
    # --------------------------------------------
    "apply_area_mask_to_data": True,  # filter data using areas cpom.masks.Mask
    "maskname": "greenland_bedmachine_v3_grid_mask",  # from  cpom.masks.Mask
    "masktype": "grid",  # mask is a polar stereo grid of Nkm resolution
    "basin_numbers": [2],  # [n1,n2,..] if mask allows basin numbers
    # for bedmachine v3, 2=grounded ice, 3=floating
    "show_polygon_mask": False,  # show mask polygon
    "polygon_mask_color": "red",  # color to draw mask polygon
    "simple_axes": [  # define plot axis position in the map only plot
        0.05,  # left
        0.04,  # bottom
        0.8,  # width (axes fraction)
        0.90,  # height (axes fraction)
    ],
    "vertical_colorbar_axes_simple": [
        0.04,  # left
        0.05,  # bottom
        0.02,  # width
        0.55,  # height (fraction of axes 0-1)
    ],  # [ left, bottom, width, height (fractions of axes)]
    "stats_position_x_offset_simple": -20,  # x offset to stats when plotting map_only
    "stats_position_y_offset_simple": 0,  # y offset to stats when plotting map_only
    "varname_annotation_position_xy_simple": (
        0.3,
        0.96,
    ),  # normalized position of default varname annotation in map_only plot
    "area_long_name_position_simple": (
        0.55,
        0.12,
    ),  # for default annot. pos of area name (map_only)
    "mask_long_name_position_simple": (
        -20,
        0.95,
    ),  # for default annotation position of area long name
}

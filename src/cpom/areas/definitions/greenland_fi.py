"""
# Area definition

## Summary:
Based on area: greenland
**Data mask: floating ice from bedmachine v3**

"""

area_definition = {
    "area_summary": "Greenland [floating ice mask]",
    "use_definitions_from": "greenland",
    # --------------------------------------------
    #    mask from cpom.masks.Mask
    # --------------------------------------------
    "apply_area_mask_to_data": True,  # filter data using areas cpom.masks.Mask
    "maskname": "greenland_bedmachine_v3_grid_mask",  # from  cpom.masks.Mask
    "masktype": "grid",  # mask is a polar stereo grid of Nkm resolution
    "basin_numbers": [3],  # [n1,n2,..] if mask allows basin numbers
    # for bedmachine v2, 2=grounded ice, 3=floating,
    "show_polygon_mask": False,  # show mask polygon
    "polygon_mask_color": "red",  # color to draw mask polygon
}

"""
# Area definition

## Summary:
Based on area: greenland
**mask: greenland_bedmachine_v3_grid_mask[2] == grounded ice sheet**

"""

area_definition = {
    "area_summary": "Greenland [grounded ice mask]",
    "use_definitions_from": "greenland_hs",
    # --------------------------------------------
    #    mask from cpom.masks.Mask
    # --------------------------------------------
    "apply_area_mask_to_data": True,  # filter data using areas cpom.masks.Mask
    "maskname": "greenland_bedmachine_v3_grid_mask",  # from  cpom.masks.Mask
    "masktype": "grid",  # mask is a polar stereo grid of Nkm resolution
    "basin_numbers": [2],  # [n1,n2,..] if mask allows basin numbers
    # for bedmachine v2, 2=grounded ice, 3=floating,
    "show_polygon_mask": False,  # show mask polygon
    "polygon_mask_color": "red",  # color to draw mask polygon
}

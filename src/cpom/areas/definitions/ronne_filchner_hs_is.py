"""
# Area definition

## Summary: 
Based on area: ronne_filchner
**background_image: hillshade**
**mask: grounded ice**

"""

area_definition = {
    "use_definitions_from": "ronne_filchner_hs",
    "area_summary": "Ronne Filchner ice shelves [hillshade, grounded ice mask]",
    # --------------------------------------------
    #    mask from clev2er.utils.masks.Mask
    # --------------------------------------------
    "apply_area_mask_to_data": True,  # filter data using areas clev2er.utils.masks.Mask
    "maskname": "antarctica_bedmachine_v2_grid_mask",  # from  clev2er.utils.masks.Mask
    "masktype": "grid",
    "basin_numbers": [2, 4],  # [n1,n2,..] if mask allows basin numbers
    # for bedmachine v2, 2=grounded ice, 3=floating, 4=vostok
    "show_polygon_mask": False,  # show mask polygon
    "polygon_mask_color": "red",  # color to draw mask polygon
}

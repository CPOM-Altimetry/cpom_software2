"""cpom.areas.definitions.ross_iceshelf_hs_fi.py

Area defined for the Ross Ice Shelf, includes hillshaded basemap
and floating ice mask from Bedmachine v2
"""

area_definition = {
    "use_definitions_from": "ross_iceshelf_hs",
    "area_summary": "Ross Ice Shelf [hillshade, floating ice]",
    "apply_area_mask_to_data": True,  # filter data using areas clev2er.utils.masks.Mask
    "maskname": "antarctica_bedmachine_v2_grid_mask",  # from  clev2er.utils.masks.Mask
    "masktype": "grid",
    "basin_numbers": [
        3,
    ],  # for bedmachine v2, 2=grounded ice, 3=floating, 4=vostok
}

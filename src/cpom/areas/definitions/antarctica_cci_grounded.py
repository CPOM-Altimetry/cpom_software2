"""Area definition"""

area_definition = {
    "long_name": "Antarctica (grounded ice)",
    "use_definitions_from": "antarctica_cci",
    "apply_area_mask_to_data": True,  # filter data using areas cpom.masks.Mask
    "maskname": "antarctica_bedmachine_v2_grid_mask",  # from  cpom.masks.Mask
    "masktype": "grid",  # mask is a polar stereo grid of Nkm resolution
    "basin_numbers": [2, 4],  # [n1,n2,..] if mask allows basin number
}

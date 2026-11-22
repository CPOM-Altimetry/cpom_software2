"""Definition for ase_xylimits_mask mask."""

mask_definition = {
    # ASE rectangular mask for rapid masking
    "mask_type": "xylimits",  # 'xylimits', 'polygon', 'grid','latlimits'
    "xlimits": [
        -1996781 - 50000,
        -1196781 + 50000,
    ],  # [minx, maxx] in m, in current  coordinate system
    "ylimits": [
        -768646 - 50000,
        31353 + 50000,
    ],  # [miny, maxy] in m, in current  coordinate system
    "crs_epsg_code": "epsg:3031",  # Polar Stereo - South
}

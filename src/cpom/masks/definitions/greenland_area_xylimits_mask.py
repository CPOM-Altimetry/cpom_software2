"""Definition for greenland_area_xylimits_mask mask."""

mask_definition = {
    # Greenland rectangular mask for rapid masking
    "mask_type": "xylimits",  # 'xylimits', 'polygon', 'grid','latlimits'
    "xlimits": [
        -630000,
        904658,
    ],  # [minx, maxx] in m, in current  coordinate system
    "ylimits": [
        -3355844,
        -654853,
    ],  # [miny, maxy] in m, in current  coordinate system
    "crs_epsg_code": "epsg:3413",  # Polar Stereo - North -latitude of origin 70N, 45W
}

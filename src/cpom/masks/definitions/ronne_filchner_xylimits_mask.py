"""Definition for ronne_filchner_xylimits_mask mask."""

mask_definition = {
    # ASE rectangular mask for rapid masking
    "mask_type": "xylimits",  # 'xylimits', 'polygon', 'grid','latlimits'
    "xlimits": [
        -1685615,
        -390616,
    ],  # [minx, maxx] in m, in current  coordinate system
    "ylimits": [
        -48143,
        1246856,
    ],  # [miny, maxy] in m, in current  coordinate system
    "crs_epsg_code": "epsg:3031",  # Polar Stereo - South
    # -----------------------------------------------------------------------------
}

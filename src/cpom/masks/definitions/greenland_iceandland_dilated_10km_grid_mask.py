"""Definition for greenland_iceandland_dilated_10km_grid_mask mask."""

from os import environ

import numpy as np

mask_definition = {
    "mask_type": "grid",  # 'xylimits', 'polygon', 'grid','latlimits'
    "default_mask_file": (
        f"{environ['CPDATA_DIR']}/RESOURCES/surface_discrimination_masks"
        "/greenland/bedmachine_v3/grn_dilated_grid_mask.nc"
    ),
    "num_x": 10218,
    "num_y": 18346,
    "dtype": np.uint8,  # data type used for mask array.
    "bad_mask_value": 255,  # value in unknown grid cells in mask
    "binsize": 150,  # meters
    "minxm": -652925,
    "minym": -3384575,
    "crs_epsg_code": "epsg:3413",  # Polar Stereo - North (70N, 45W)
    "mask_grid_possible_values": [0, 1],  # values in the mask_grid
    "grid_value_names": ["outside", "inside Greenland dilated mask"],
    "grid_colors": ["blue", "darkgrey"],
    "mask_long_name": "Dilated by 10km in to  Ocean",
}

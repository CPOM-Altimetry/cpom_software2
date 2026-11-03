"""Definition for antarctica_iceandland_dilated_10km_grid_mask mask."""

from os import environ

import numpy as np

mask_definition = {
    "mask_type": "grid",  # 'xylimits', 'polygon', 'grid','latlimits'
    "default_mask_file": (
        f"{environ['CPDATA_DIR']}/RESOURCES/surface_discrimination_masks"
        "/antarctica/bedmachine_v2/ant_dilated_grid_mask.nc"
    ),
    "num_x": 13333,
    "num_y": 13333,
    "dtype": np.uint8,  # data type used for mask array.
    "bad_mask_value": 255,  # value in unknown grid cells in mask
    "minxm": -3333000,
    "minym": -3333000,
    "binsize": 500,  # meters
    "crs_epsg_code": "epsg:3031",  # Polar Stereo - South (71S, 0E)
    "mask_grid_possible_values": [0, 1],  # values in the mask_grid
    "grid_value_names": ["outside", "inside Antarctic dilated mask"],
    "grid_colors": ["blue", "darkgrey"],
    "nc_mask": True,
    "nc_mask_flip": False,
    "nc_mask_var": "mask",
    "mask_long_name": "Dilated by 10km in to  Ocean",
}

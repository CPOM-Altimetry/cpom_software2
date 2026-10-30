"""Definition for antarctic_grounded_and_floating_2km_grid_mask mask."""

from os import environ

import numpy as np

mask_definition = {
    "mask_type": "grid",  # 'xylimits', 'polygon', 'grid','latlimits',
    "default_mask_file": (
        f"{environ['CPOM_SOFTWARE_DIR']}/resources/drainage_basins/antarctica"
        "/zwally_2012_imbie1_ant_grounded_and_floating_icesheet_basins/"
        "basins/zwally_2012_imbie1_ant_grounded_and_floating_icesheet_basins_2km.nc"
    ),
    "num_x": 2820,
    "num_y": 2420,
    "dtype": np.uint8,
    "bad_mask_value": 255,  # value in unknown grid cells in mask
    "nc_mask": True,
    "nc_mask_flip": False,
    "nc_mask_var": "ANT_ZWALLY_BASINMASK_INCFLOATING_ICE",
    "minxm": -2820000,
    "minym": -2420000,
    "binsize": 2000,  # km
    "mask_grid_possible_values": list(range(28)),  # values in the mask_grid
    "grid_value_names": [f"Basin-{i}" for i in range(28)],
    "grid_value_names[0]": "Unknown",
    "crs_epsg_code": "epsg:3031",  # Polar Stereo - South -71S
    "mask_long_name": "Zwally grounded and floating ice 2km grid",
}

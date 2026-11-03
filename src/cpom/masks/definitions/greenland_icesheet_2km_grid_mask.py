"""Definition for greenland_icesheet_2km_grid_mask mask."""

from os import environ

import numpy as np

mask_definition = {
    "mask_type": "grid",  # 'xylimits', 'polygon', 'grid','latlimits'
    "default_mask_file": (
        f"{environ['CPOM_SOFTWARE_DIR']}/resources/drainage_basins/greenland/"
        "zwally_2012_grn_icesheet_basins/basins/Zwally_GIS_basins_2km.nc"
    ),
    "num_x": 1000,
    "num_y": 1550,
    "dtype": np.uint8,
    "bad_mask_value": 255,  # value in unknown grid cells in mask
    "minxm": -1000000,
    "minym": -3500000,
    "binsize": 2000,
    "nc_mask": True,
    "nc_mask_flip": False,
    "nc_mask_var": "gre_basin_mask",
    "crs_epsg_code": "epsg:3413",  # Polar Stereo - North -latitude of origin 70N, 45
    "grid_value_names": [
        "None",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "2.1",
        "2.2",
        "3.1",
        "3.2",
        "3.3",
        "4.1",
        "4.2",
        "4.3",
        "5.0",
        "6.1",
        "6.2",
        "7.1",
        "7.2",
        "8.1",
        "8.2",
    ],
    "mask_grid_possible_values": list(range(20)),  # values in the mask_grid
    "grid_colors": [
        "blue",
        "bisque",
        "darkorange",
        "moccasin",
        "gold",
        "greenyellow",
        "yellowgreen",
        "gray",
        "lightgray",
        "silver",
        "purple",
        "sandybrown",
        "peachpuff",
        "coral",
        "tomato",
        "navy",
        "lavender",
        "olivedrab",
        "lightyellow",
        "sienna",
    ],
    "mask_long_name": "Zwally grounded and floating ice 2km grid",
}

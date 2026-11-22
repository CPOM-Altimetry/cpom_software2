"""Definition for greenland_icesheet_2km_grid_mask_mouginot2019 mask."""

from os import environ

import numpy as np

mask_definition = {
    "mask_type": "grid",  # 'xylimits', 'polygon', 'grid','latlimits'
    "default_mask_file": (
        f"{environ['CPOM_SOFTWARE_DIR']}/resources/drainage_basins/greenland/"
        "Mouginot/"
        "mouginot_2019_grn_grounded_icesheet_basins_2km.nc"
    ),
    "num_x": 1000,
    "num_y": 1550,
    "minxm": -1000000,  # meters
    "minym": -3500000,  # meters
    "binsize": 2000,  # meters
    "dtype": np.uint8,
    "bad_mask_value": 0,  # value in unknown grid cells in mask
    "nc_mask": True,
    "nc_mask_flip": False,
    "nc_mask_var": "basinmask",
    "mask_grid_possible_values": list(range(7)),  # values in the mask_grid
    "grid_value_names": [
        "Unclassified [0]",
        "CE [1]",
        "CW [2]",
        "NO [3]",
        "NE [4]",
        "NW [5]",
        "SE [6]",
        "SW [7]",
    ],
    "crs_epsg_code": "epsg:3413",  # Polar Stereo - North -latitude of origin 70N, 45
    "shapefile_path": (
        f"{environ['CPOM_SOFTWARE_DIR']}/resources/drainage_basins/greenland/"
        "Mouginot/"
        "Greenland_Basins_PS_v1.4.2.shp"
    ),
    "shapefile_column_name": "SUBREGION1",
    "mask_long_name": "Mouginot (2019) 2km grid",
}

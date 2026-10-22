"""Definition for antarctic_icesheet_2km_grid_mask_rignot2016 mask."""

from os import environ

import numpy as np

mask_definition = {
    # basin mask values are : 0..18, or 255 (unknown)
    #
    "mask_type": "grid",  # 'xylimits', 'polygon', 'grid','latlimits'
    "default_mask_file": (
        f"{environ['CPOM_SOFTWARE_DIR']}/resources/drainage_basins/antarctica/"
        "rignot_2016_imbie2_ant_grounded_icesheet_basins/basins/"
        "rignot_2016_imbie2_ant_grounded_icesheet_basins_2km.nc"
    ),
    "num_x": 2820,
    "num_y": 2420,
    "dtype": np.uint8,
    "bad_mask_value": 255,  # value in unknown grid cells in mask
    "minxm": -2820000,  # meters
    "minym": -2420000,  # meters
    "binsize": 2000,  # meters
    "nc_mask": True,
    "nc_mask_flip": False,
    "nc_mask_var": "basinmask",
    "mask_grid_possible_values": list(range(19)),  # values in the mask_grid
    "grid_value_names": [
        "Islands",
        "West H-Hp",
        "West F-G",
        "East E-Ep",
        "East D-Dp",
        "East Cp-D",
        "East B-C",
        "East A-Ap",
        "East Jpp-K",
        "West G-H",
        "East Dp-E",
        "East Ap-B",
        "East C-Cp",
        "East K-A",
        "West J-Jpp",
        "Peninsula Ipp-J",
        "Peninsula I-Ipp",
        "Peninsula Hp-I",
        "West Ep-F",
    ],
    "crs_epsg_code": "epsg:3031",  # Polar Stereo - South -71S
    "shapefile_path": (
        f"{environ['CPOM_SOFTWARE_DIR']}/resources/drainage_basins/"
        "antarctica/rignot_2016_imbie2_ant_grounded_icesheet_basins/data/"
        "/ANT_Basins_IMBIE2_v1.6.shp"
    ),
    "shapefile_column_name": "SUBREGION1",
    "mask_long_name": "Rignot (2016) 2km grid",
}

"""Definition for greenland_icesheet_2km_grid_mask_rignot2016 mask."""

from os import environ

import numpy as np

mask_definition = {
    "mask_type": "grid",  # 'xylimits', 'polygon', 'grid','latlimits'
    "default_mask_file": (
        f"{environ['CPOM_SOFTWARE_DIR']}/resources/drainage_basins/greenland/"
        "GRE_Basins_IMBIE2_v1.3/basins/"
        "rignot_2016_imbie2_grn_grounded_icesheet_basins_2km.nc"
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
    "mask_grid_possible_values": list(range(57)),  # values in the mask_grid
    # 0 (unclassified), 1-50 (ice caps), 51 (NW), 52(CW), 53(SW), 54(SE), 55(NE), 56(NO)
    "grid_value_names": ["Ice cap -" + str(i) for i in range(57)],
    "grid_value_names[0]": "unclassified",
    "grid_value_names[51]": "NW",
    "grid_value_names[52]": "CW",
    "grid_value_names[53]": "SW",
    "grid_value_names[54]": "SE",
    "grid_value_names[55]": "NE",
    "grid_value_names[56]": "NO",
    "crs_epsg_code": "epsg:3413",  # Polar Stereo - North -latitude of origin 70N, 45
    "shapefile_path": (
        f"{environ['CPOM_SOFTWARE_DIR']}/resources/drainage_basins/greenland/"
        "GRE_Basins_IMBIE2_v1.3/shpfiles/GRE_IceSheet_IMBIE2_v1.3.shp"
    ),
    "shapefile_column_name": "SUBREGION1",
    "mask_long_name": "Rignot (2016) 2km grid",
}

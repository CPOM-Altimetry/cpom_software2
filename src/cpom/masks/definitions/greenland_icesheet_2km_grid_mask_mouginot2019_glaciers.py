"""Definition for greenland_icesheet_2km_grid_mask_mouginot2019_glaciers mask."""

from os import environ

import numpy as np

mask_definition = {
    "mask_type": "grid",  # 'xylimits', 'polygon', 'grid','latlimits'
    "default_mask_file": (
        f"{environ['CPOM_SOFTWARE_DIR']}/resources/drainage_basins/greenland/"
        "Mouginot/mouginot_subbasins/"
        "mouginot_2019_grn_grounded_icesheet_glacier_basins_2km.nc"
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
    "glacier_csv": (
        f"{environ['CPOM_SOFTWARE_DIR']}/resources/drainage_basins/greenland/"
        "Mouginot/mouginot_subbasins/glacier_names.csv"
    ),
    # Mask values are 1-260, CSV has 260 rows. Prepend entry for mask value 0.
    "shapefile_path": (
        f"{environ['CPOM_SOFTWARE_DIR']}/resources/drainage_basins/greenland/"
        "Mouginot/mouginot_subbasins/"
        "Greenland_Basins_PS_v1.4.2.shp"
    ),
    "shapefile_column_name": "NAME",
    "crs_epsg_code": "epsg:3413",  # Polar Stereo - North -latitude of origin 70N, 45
    "mask_long_name": "Mouginot (2019) 2km glacier grid",
}

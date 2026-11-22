"""Definition for greenland_bedmachine_v3_grid_mask mask."""

from os import environ

import numpy as np


def basin_longname_suffix(mask_long_name: str, basin_numbers: list[int]) -> str:
    """Adds suffixes to the long name depending on basin selection

    Args:
        mask_long_name (str): Long mask name
        basin_numbers (list[int]): Selected basins

    Returns:
        str: mask_long_name
    """
    if 0 in basin_numbers:
        mask_long_name += "Ocean"
    if 1 in basin_numbers:
        mask_long_name += " Ice-free land"
    if 2 in basin_numbers:
        mask_long_name += " Grounded ice"
    if 3 in basin_numbers:
        mask_long_name += " Floating ice"
    if 4 in basin_numbers:
        mask_long_name += "Non-Greenland land"
    mask_long_name += " (bedmachine v3)"

    return mask_long_name


mask_definition = {
    # Greenland surface type grid mask from BedMachine v3, 150m resolution
    #   - 0='Ocean', 1='Ice-free land',2='Grounded ice',3='Floating ice',
    #     4='non-Greenland land'
    "mask_type": "grid",  # 'xylimits', 'polygon', 'grid','latlimits'
    # read netcdf file
    "default_mask_file": (
        f"{environ['CPDATA_DIR']}/RESOURCES/surface_discrimination_masks"
        "/greenland/bedmachine_v3/BedMachineGreenland-2017-09-20.nc"
    ),
    "num_x": 10218,  # mask dimension in x direction
    "num_y": 18346,  # mask dimension in y direction
    "dtype": np.uint8,  # data type used for mask array.
    # values are 0..4, so using uint8 to reduce memory
    "bad_mask_value": 255,  # value in unknown grid cells in mask
    "nc_mask": True,
    "nc_mask_flip": True,
    "binsize": 150,
    "minxm": -652925,
    "minym": -632675 - (18346 * 150),
    "crs_epsg_code": "epsg:3413",  # Polar Stereo - South (70N, 45W)
    "mask_grid_possible_values": [0, 1, 2, 3, 4],  # values in the mask_grid
    "grid_value_names": [
        "Ocean",  # 0
        "Ice-free land",  # 1
        "Grounded ice",  # 2
        "Floating ice",  # 3
        "Non-Greenland land",  # 4
    ],
    "grid_colors": ["blue", "brown", "grey", "green", "white"],
    # -----------------------------------------------------------------------------
    # Antarctica surface type grid mask derived from BedMachine v2, 500m resolution
    #    - 0='Other', 1='Ice (grounded+floating)+ice-free land, dilated by 10km in to  ocean'
}

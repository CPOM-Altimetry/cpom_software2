"""Definition for antarctica_bedmachine_v2_grid_mask mask."""

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
    if 4 in basin_numbers and 2 not in basin_numbers:
        mask_long_name += " Lake Vostok"
    mask_long_name += " (bedmachine v2)"

    return mask_long_name


mask_definition = {
    # Antarctica surface type grid mask from BedMachine v2, 500m resolution
    #    - 0='Ocean', 1='Ice-free land',2='Grounded ice',3='Floating ice',
    #      4='Lake Vostok'
    # src=https://nsidc.org/data/nsidc-0756/versions/2
    "mask_type": "grid",  # 'xylimits', 'polygon', 'grid','latlimits'
    "default_mask_file": (
        f"{environ['CPDATA_DIR']}/RESOURCES/surface_discrimination_masks"
        "/antarctica/bedmachine_v2/BedMachineAntarctica_2020-07-15_v02.nc"
    ),
    "num_x": 13333,  # mask dimension in x direction
    "num_y": 13333,  # mask dimension in y direction
    "dtype": np.uint8,  # data type used for mask array.
    # values are 0..4, so using uint8 to reduce memory
    "bad_mask_value": 255,  # value in unknown grid cells in mask
    "nc_mask": True,
    "nc_mask_flip": True,
    "minxm": -3333000,
    "minym": -3333000,
    "binsize": 500,
    "crs_epsg_code": "epsg:3031",  # Polar Stereo - South (71S, 0E)
    "mask_grid_possible_values": [0, 1, 2, 3, 4],  # values in the mask_grid
    "grid_value_names": [
        "Ocean",
        "Ice-free land",
        "Grounded ice",
        "Floating ice",
        "Lake Vostok",
    ],
    "grid_colors": ["blue", "brown", "grey", "green", "red"],
    "basin_longname_suffixes": basin_longname_suffix,
}

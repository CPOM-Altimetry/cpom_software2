# pylint: disable=R0801
"""
Template for defining an altimetry dataset and it's parameters.
Used by AltDataset class to create a dataset object.
"""

import os

dataset_definition = {
    "mission": "mission_name",
    "long_name": "dataset description",
    "l2_dir": os.environ["CPDATA_DIR"] + "path to dataset dir ",
    "search_pattern": "search pattern to use when globbing",
    "yyyymm_str_fname_indices": [-83, -75],  # Indices of the date in the filename
    "time_epoch": "YYYY-MM-DDT00:00:00",  # Time epoch of the dataset
    "data_packed_in_blocks": False,  # True for E1, E2, False for others
    # -------------------#
    # --Core Parameters--#
    # -------------------#
    "lat_param": "latitude parameter name",
    "lon_param": "longitude parameter name ",
    "elevation_param": "elevation parameter name",
    "time_param": "time parameter name",
    "power_param": "power parameter name ",
    # -------------------------#
    # --Additional Parameters--#
    # -------------------------#
    "lat_nadir_param": "",  # Optional
    "lon_nadir_param": "",  # Optional
    # Add any other parameters here
}

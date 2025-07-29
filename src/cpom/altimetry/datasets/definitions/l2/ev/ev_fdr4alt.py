"""ENVISAT FDR4ALT Land Ice dataset definition"""

# pylint: disable=R0801

import os

dataset_definition = {
    "mission": "ev",
    "long_name": "ENVISAT FDR4ALT Land Ice",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/ENV/L2/FDR4ALT/TDP",
    # Search pattern for L2 file discovery."
    "search_pattern": "**/EN1_F4A_ALT_TDP_LI_*.nc",
    "yyyymm_str_fname_indices": [-23, -31],
    "time_epoch": "1950-01-01T00:00:00",
    # -------------------#
    # --Core Parameters--#
    # -------------------#
    "lat_param": "expert/ice_sheet_lat_poca",
    "lon_param": "expert/ice_sheet_lon_poca",
    "lat_nadir_param": "expert/latitude",
    "lon_nadir_param": "expert/longitude",
    "elevation_param": "expert/ice_sheet_elevation_ice1_roemer",
    "power_param": "expert/sigma0_ice1",
    "quality_param": "expert/retracking_ice1_qual",  # contains good=0, bad=1
    "time_param": "expert/time",
}

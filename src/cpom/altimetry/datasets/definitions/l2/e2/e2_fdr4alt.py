"""ERS-2 FDR4ALT Land Ice dataset definition"""

# pylint: disable=R0801

import os

dataset_definition = {
    "mission": "e2",
    "long_name": "ERS-2 FDR4ALT Land Ice",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/ERS2/L2/FDR4ALT/TDP",
    "search_pattern": "**/ER2_F4A_ALT_TDP_LI_*.nc",
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
    "quality_param": "expert/retracking_ice1_qual",
    "time_param": "expert/time",
}

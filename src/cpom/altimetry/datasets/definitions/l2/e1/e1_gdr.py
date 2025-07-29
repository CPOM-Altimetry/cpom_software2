"""ERS-1 REAPER dataset definition"""

# pylint: disable=R0801

import os
import numpy as np

dataset_definition = {
    "mission": "e1",
    "long_name": "ERS-1 REAPER",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/ERS1/REAPER/L2",
    "search_pattern": "**/E1_REAP_ERS_ALT_2__*.NC",
    "yyyymm_str_fname_indices": [-39, -31],
    "time_epoch": "1990-01-01T00:00:00",
    "data_packed_in_blocks": True,
    # -------------------#
    # --Core Parameters--#
    # -------------------#
    "lat_param": "offset_lat_20hz",
    "lon_param": "offset_lon_20hz",
    "lat_nadir_param": "lat_20hz",
    "lon_nadir_param": "lon_20hz",
    "elevation_param": "offset_elevation_20hz_filt",
    "power_param": "ice1_sig0_20hz",
    "time_param": "time_20hz",
}


def get_offset_elevation_20hz_filt(get_var, nc=None):
    elev = get_var("offset_elevation_20hz")
    qual = get_var("ice1_qual_flag_20hz")

    if elev.mask is np.ma.nomask:
        elev = np.ma.masked_where(qual > 0, elev)
    return elev


derived = {
    "offset_elevation_20hz_filt": get_offset_elevation_20hz_filt,
}

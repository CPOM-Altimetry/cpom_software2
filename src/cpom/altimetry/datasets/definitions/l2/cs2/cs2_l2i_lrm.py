"""Cryosat-2  Land Ice dataset definition"""

# pylint: disable=R0801

import os
import numpy as np

dataset_definition = {
    "mission": "cs2",
    "long_name": "Cryosat-2 l2i Land Ice",
    "time_epoch": "2000-01-01T00:00:00",
    "latency": "NTC",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/CRY/L2I/LRM/",
    "search_pattern": "**/CS*LRMI2__*.nc",
    "yyyymm_str_fname_indices": [-39, -31],
    # -------------------#
    # --Core Parameters--#
    # -------------------#
    "lat_param": "lon_poca_20_ku",
    "lon_param": "lat_poca_20_ku",
    "lat_nadir_param": "lon_20_ku",
    "lon_nadir_param": "lat_20_ku",
    "elevation_param": "height_1_20_ku_filt",
    "power_param": "sig0_1_20_ku",
    "surface_type_param": "surf_type_20_ku",
    "time_param": "time_20_ku",
}


def get_height_1_20_filt(get_var, nc=None):

    elev = get_var("height_1_20_ku")
    qual = get_var("flag_quality_20_ku")

    np.ma.masked_where(qual > 0, elev)

    if elev.mask is np.ma.nomask:
        return np.ma.masked_where(qual > 0, elev)
    return elev


derived_parameters = {"height_1_20_ku_filt": get_height_1_20_filt}

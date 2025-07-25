"""Sentinel-3A L2 dataset definition"""

# pylint: disable=R0801

import os
import numpy as np

dataset_definition = {
    "mission": "s3a",
    "long_name": "Sentinel-3A SR_2_LAN NT",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/S3A/L2/SR_2_LAN_NT",
    "search_pattern": "S3A_SR_2_LAN____*.SEN3/standard_measurement.nc",
    "yyyymm_str_fname_indices": [-83, -75],  # Comes from name.parent for s3a
    "time_epoch": "2000-01-01T00:00:00",
    # -------------------#
    # --Core Parameters--#
    # -------------------#
    "lat_param": "lat_20_ku",
    "lon_param": "lon_20_ku",
    "lat_nadir_param": "lat_cor_20_ku",
    "lon_nadir_param": "lon_cor_20_ku",
    "elevation_param": "elevation_ocog_20_ku_filt",
    "time_param": "time_20_ku",
    "power_param": "sig0_ocog_20_ku_base3",
    # -------------------------#
    # --Additional Parameters--#
    # -------------------------#
    # Sentinel-3A specific parameters for lat , lon
    "lat_cband_nadir_param": "lat_20_c",
    "lon_cband_nadir_param": "lon_20_c",
    "lat_cband_poca_param": "lat_cor_20_c",
    "lon_cband_poca_param": "lon_cor_20_c",
    "lat_01_param": "lat_01",  # 1Hz nadir latitude parameter name
    "lon_01_param": "lon_01",  # 1Hz nadir longitude parameter name
    # Mission L2 surface type parameter names
    # Surface Type parameter name to use for ocean==0 discimination
    "surf_type_20_param": "surf_class_20_ku",
    "surf_type_01_param": "surf_class_01",
    "surf_type_cband_param": "surf_class_20_c",
}


def get_elevation_ocog_20_ku_filt(get_var, nc=None):

    elev = get_var("elevation_ocog_20_ku")
    qual = get_var("waveform_qual_ice_20_ku")

    if elev.mask is np.ma.nomask:
        return np.ma.masked_where(qual > 0, elev)
    return elev


def get_sig0_ocog_20_ku_base3(get_var, nc=None):
    product_name = nc.getncattr("product_name")
    if product_name[91:94] == "004":
        var = get_var("sig0_ocog_20_ku") + (10.0 * np.log10(64))
    else:
        var = get_var("sig0_ocog_20_ku")
    return var


derived_parameters = {
    "elevation_ocog_20_ku_filt": get_elevation_ocog_20_ku_filt,
    "sig0_ocog_20_ku_base3": get_sig0_ocog_20_ku_base3,
}

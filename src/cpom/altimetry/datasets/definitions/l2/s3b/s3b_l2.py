"""Sentinel-3B L2 dataset definition"""

# pylint: disable=R0801

import os
import numpy as np

dataset_definition = {
    "mission": "s3b",
    "long_name": "Sentinel-3B SR_2_LAN_NT",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/S3B/L2/SR_2_LAN_NT",
    # Search pattern for L2 file discovery.
    "search_pattern": "S3B_SR_2_LAN____*.SEN3/standard_measurement.nc",
    "yyyymm_str_fname_indices": [-83, -75],  # Comes from name.parent for s3b
    "time_epoch": "2000-01-01T00:00:00",
    # -------------------#
    # --Core Parameters--#
    # -------------------#
    "lat_param": "lat_20_ku",
    "lon_param": "lon_20_ku",
    "lat_nadir_param": "lat_cor_20_ku",
    "lon_nadir_param": "lon_cor_20_ku",
    "elevation_param": "elevation_ocog_20_ku_filt",
    "power_param": "sig0_ocog_20_ku_base3",
    "time_param": "time_20_ku",
    # -------------------------#
    # --Additional Parameters--#
    # -------------------------#
    "lat_20_cband_nadir": "lat_20_c",
    "lon_20_cband_nadir": "lon_20_c",
    "lat_20_cband_poca": "lat_cor_20_c",
    "lon_20_cband_poca": "lon_cor_20_c",
    "lat_01_name": "lat_01",  # 1Hz nadir latitude parameter name
    "lon_01_name": "lon_01",  # 1Hz nadir longitude parameter name
    # Mission L2 surface type parameter names
    # 20Hz Surface Type parameter name to use for ocean==0 discimination
    "surf_type_name_20": "surf_class_20_ku",
    # 1Hz Surface Type parameter name to use for ocean==0 discimination
    "surf_type_name_01": "surf_class_01",
    # C-band 20Hz Surface Type parameter name to use for ocean==0 discimination
    "surf_type_name_cband": "surf_class_20_c",
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

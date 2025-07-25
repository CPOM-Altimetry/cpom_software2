"""Icesat-2 ATL06 dataset definition.`"""

# pylint: disable=R0801
import os
import numpy as np

dataset_definition = {
    "mission": "is2",
    "long_name": "Icesat-2 ATL06",
    "time_epoch": "2000-01-01T00:00:00",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/LASER/ICESAT-2/ATL-06/versions/006/",
    "search_pattern": "**/ATL06_*.h5",
    "yyyymm_str_fname_indices": [-33, -25],
    # Can be overridden if beam list passed to AltDataset
    "beams": ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"],
    # -------------------#
    # --Core Parameters--#
    # -------------------#
    "latitude_param": "{beam}/land_ice_segments/latitude",
    "longitude_param": "{beam}/land_ice_segments/longitude",
    "elevation_param": "{beam}/land_ice_segments/h_li_filt",
    "time_param": "{beam}/land_ice_segments/time",
}


def get_h_li_filt(get_var, beam, nc=None):
    """Get filtered land ice height."""
    h_li = get_var("{beam}/land_ice_segments/h_li")
    quality = get_var("{beam}/land_ice_segments/atl06_quality_summary")

    return np.ma.masked_where((quality != 0) | (h_li > 10e3), h_li)


derived_parameters = {"h_li_filt": "{beam}/land_ice_segments/h_li_filt"}

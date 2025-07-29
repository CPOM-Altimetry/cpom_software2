"""ENVISAT GDR dataset definition"""

# pylint: disable=R0801

import os
import numpy as np

dataset_definition = {
    "mission": "ev",
    "long_name": "ENVISAT GDR",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/ENV/L2/GDR_v3",
    # Search pattern for L2 file discovery."
    "search_pattern": "**/ENV_RA_2_GDR____*.nc",
    "yyyymm_str_fname_indices": [-80, -72],
    "time_epoch": "2000-01-01T00:00:00",
    # -------------------#
    # --Core Parameters--#
    # -------------------#
    "lat_param": "lat_cor_20",
    "lon_param": "lon_cor_20",
    "lat_nadir_param": "lon_20",
    "lon_nadir_param": "lon_cor_20_ku",
    "power_param": "sig0_ice1_20_ku",
    "time_param": "time_20",
    # Additional parameters
    "elevation_param": "fully_corrected_elevation",  # Derived
    # -------------------------#
    # --Additional Parameters--#
    # -------------------------#
    # 20Hz Surface Type parameter name to use for ocean==0 discimination
    "surf_type_name_20": "surf_type_20",
}

# ---------------------------------------#
# --   Derived Parameters Definitions  --#
# ---------------------------------------#


def get_fully_corrected_elevation(get_var, nc=None):
    elev = get_var("elevation_ice1_20_ku")
    surf = get_var("surf_class_20")
    ind_meas_1hz_20 = get_var("ind_meas_1hz_20")

    dry = get_var("mod_dry_tropo_cor_01")[ind_meas_1hz_20]
    wet = get_var("mod_wet_tropo_cor_01")[ind_meas_1hz_20]
    iono = get_var("iono_cor_gim_01_ku")[ind_meas_1hz_20]
    solid = get_var("solid_earth_tide_01")[ind_meas_1hz_20]
    pole = get_var("pole_tide_01")[ind_meas_1hz_20]
    load = get_var("load_tide_sol2_01")[ind_meas_1hz_20]
    ocean = get_var("ocean_tide_sol2_01")[ind_meas_1hz_20]
    ib = get_var("inv_bar_cor_01")[ind_meas_1hz_20]

    ocean = np.logical_or(surf == 0, surf == 5)  # Find ocean or floating ice shelf
    land = (
        ~ocean
    )  # Anything else is land (no aqua veg, salt basins, continental water in cryosphere)

    geocorr = np.zeros(elev.size)  # Array of geocorrection total, parallel to elev

    if sum(ocean) > 0:
        geocorr[ocean] = (
            dry[ocean]
            + wet[ocean]
            + iono[ocean]
            + solid[ocean]
            + pole[ocean]
            + ocean[ocean]
            + ib[ocean]
        )
    if sum(land) > 0:
        geocorr[land] = dry[land] + wet[land] + iono[land] + solid[land] + pole[land] + load[land]

    var = np.ma.masked_where(np.isnan(elev - geocorr), elev - geocorr)
    return var


derived_parameters = {"fully_corrected_elevation": get_fully_corrected_elevation}

import os

dataset_definition = {
    "mission": "ev",
    "long_name": "ENVISAT GDR",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/ENV/L2/GDR_v3",
    "search_pattern": "**/ENV_RA_2_GDR____*.nc",  # Search pattern for L2 file discovery."
    "yyyymm_str_fname_indices": [-80, -72],
    "time_epoch": "2000-01-01T00:00:00",
    "latency": "NTC",
    "data_packed_in_blocks": False,
    "latitude_param": "lat_cor_20",
    "longitude_param": "lon_cor_20",
    "latitude_nadir_param": "lon_20",
    "longitude_nadir_param": "lon_cor_20_ku",
    "lat_01_name": "lat_01",  # 1Hz nadir latitude parameter name
    "lon_01_name": "lon_01",  # 1Hz nadir longitude parameter name
    "lat_40_name": "ice_sheet_lat_poca",
    "lon_40_name": "ice_sheet_lon_poca",
    # Mission L2 surface type parameter names
    "surf_type_name_20": "surf_type_20",  # 20Hz Surface Type parameter name to use for ocean==0 discimination
    "surf_type_name_01": "surf_type_01",  # 1Hz Surface Type parameter name to use for ocean==0 discimination
    "elevation_param": "fully_corrected_elevation",
    "power_param": "sig0_ice1_20_ku",
    # "time_param": #TODO,
    # "mode_param": "instrument_mode",
}

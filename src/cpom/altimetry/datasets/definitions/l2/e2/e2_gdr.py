import os

dataset_definition = {
    "mission": "e2",
    "long_name": "ERS-2 REAPER",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/ERS2/REAPER/L2",
    "search_pattern": "**/E2_REAP_ERS_ALT_2__*.NC",
    "yyyymm_str_fname_indices": [-39, -31],
    "time_epoch": "1990-01-01T00:00:00",

    "latency": "NTC",
    "data_packed_in_blocks": True,
    "latitude_param": "offset_lat_20hz",
    "longitude_param": "offset_lon_20hz",
    "latitude_nadir_param": "lat_20hz",
    "longitude_nadir_param": "lon_20hz",
    "lat_01_name": "lat",  # 1Hz nadir latitude parameter name
    "lon_01_name": "lon",  # 1Hz nadir longitude parameter name
    # Mission L2 surface type parameter names
    "surf_type_name_01": "surface_type",  # 1Hz Surface Type parameter name to use for ocean==0 discimination
    "elevation_param": "offset_elevation_20hz_filt",
    "power_param": "ice1_sig0_20hz",
    # "time_param": #TODO,
    # "mode_param": "instrument_mode",
}

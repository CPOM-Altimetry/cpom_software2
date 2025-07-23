import os

dataset_definition = {
    "mission": "e1",
    "long_name": "ERS-1 REAPER",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/ERS1/REAPER/L2",
    "search_pattern": "**/E1_REAP_ERS_ALT_2__*.NC",  # Search pattern for L2 file discovery."
    "yyyymm_str_fname_indices": [-39, -31],
    "time_epoch": "1990-01-01T00:00:00",
    "data_packed_in_blocks": True,
    "latency": "NTC",
    "latitude_param": "offset_lat_20hz",
    "longitude_param": "offset_lon_20hz",
    "latitude_nadir_param": "lat_20hz",
    "longitude_nadir_param": "lon_20hz",
    "lat_01_param": "lat",  # 1Hz nadir latitude parameter name
    "lon_01_param": "lon",  # 1Hz nadir longitude parameter name
    # Mission L2 surface type parameter names
    "surf_type_param": "surface_type",  # 1Hz Surface Type parameter name to use for ocean==0 discimination
    "elevation_param": "offset_elevation_20hz_filt",
    "power_param": "ice1_sig0_20hz",
    # "time_param": #TODO,
}

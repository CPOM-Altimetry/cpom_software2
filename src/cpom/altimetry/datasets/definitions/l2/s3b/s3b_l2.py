import os

dataset_definition = {
    "mission": "s3b",
    "long_name": "Sentinel-3B SR_2_LAN_NT",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/S3B/L2/SR_2_LAN_NT",
    "search_pattern": "S3B_SR_2_LAN____*.SEN3/standard_measurement.nc",  # Search pattern for L2 file discovery."
    "yyyymm_str_fname_indices": [-83, -75],  # Comes from dir not filename for sentinel
    "data_packed_in_blocks": False,
    "time_epoch": "2000-01-01T00:00:00",
    "latency": "NTC",
    "latitude_param": "lat_20_ku",
    "longitude_param": "lon_20_ku",
    "latitude_nadir_param": "lat_cor_20_ku",
    "longitude_nadir_param": "lon_cor_20_ku",
    "latitude_20_cband_nadir": "lat_20_c",
    "longitude_20_cband_nadir": "lon_20_c",
    "latitude_20_cband_poca": "lat_cor_20_c",
    "longitude_20_cband_poca": "lon_cor_20_c",
    "lat_01_name": "lat_01",  # 1Hz nadir latitude parameter name
    "lon_01_name": "lon_01",  # 1Hz nadir longitude parameter name
    # Mission L2 surface type parameter names
    "surf_type_name_20": "surf_class_20_ku",  # 20Hz Surface Type parameter name to use for ocean==0 discimination
    "surf_type_name_01": "surf_class_01",  # 1Hz Surface Type parameter name to use for ocean==0 discimination
    "surf_type_name_cband": "surf_class_20_c",  # C-band 20Hz Surface Type parameter name to use for ocean==0 discimination
    "elevation_param": "elevation_ocog_20_ku_filt",
    "power_param": "sig0_ocog_20_ku_base3",
    # "time_param": #TODO,
    # "mode_param": "instrument_mode",
}

import os

dataset_definition = {
    "mission": "cs2",
    "long_name": "Cryotempo Land Ice",
    "data_packed_in_blocks": False,
    "time_epoch": "2000-01-01T00:00:00",
    "latency": "NTC",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/CRY/L2I/",
    "lrm": {
        "search_pattern": "**/CS*SINI2__*.nc",  # Search pattern for L2 file discovery."
        "yyyymm_str_fname_indices": [-39, -31],
        "latitude_param": "lat_poca_20_ku",
        "longitude_param": "lon_poca_20_ku",
        "latitude_nadir_param": "lat_20_ku",
        "longitude_nadir_param": "lon_20_ku",
        "elevation_param": "height_3_20_ku_filt",
        "power_param": "sig0_3_20_ku",
        "surface_type_param": "surf_type_20_ku",
        # "time_param": "lrm/time"
    },
    "sin": {
        "search_pattern": "**/CS*LRMI2__*.nc",  # Search pattern for L2 file discovery."
        "yyyymm_str_fname_indices": [-48, -42],
        "latitude_param": "lon_poca_20_ku",
        "longitude_param": "lat_poca_20_ku",
        "latitude_nadir_param": "lon_20_ku",
        "longitude_nadir_param": "lat_20_ku",
        "elevation_param": "height_1_20_ku_filt",
        "power_param": "sig0_1_20_ku",
        "surface_type_param": "surf_type_20_ku",
        # "time_param": "sin/time"
    },
}

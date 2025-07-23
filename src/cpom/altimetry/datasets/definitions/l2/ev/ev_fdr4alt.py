import os

dataset_definition = {
    "mission": "ev",
    "long_name": "ENVISAT FDR4ALT Land Ice",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/ENV/L2/FDR4ALT/TDP",
    "search_pattern": "**/EN1_F4A_ALT_TDP_LI_GREENL*.nc",  # Search pattern for L2 file discovery."
    "yyyymm_str_fname_indices": [-23, -31],
    "time_epoch": "1950-01-01T00:00:00",
    "latency": "NTC",
    "latitude_param": "ice_sheet_lat_poca",
    "longitude_param": "ice_sheet_lon_poca",
    "latitude_nadir_param": "latitude",
    "longitude_nadir_param": "longitude",
    "elevation_param": "ice_sheet_elevation_ice1_roemer",
    "power_param": "sigma0_ice1",
    "quality_param": "retracking_ice1_qual",  # contains good=0, bad=1
    # "time_param": #TODO,
    # "mode_param": "instrument_mode",
}

import os

dataset_definition = {
    "mission": "cs2",
    "long_name": "Cryotempo L1B",
    "data_packed_in_blocks": False,
    "time_epoch": "2000-01-01T00:00:00",
    "latency": "NTC",
    "l2_dir": os.environ["CPDATA_DIR"] + "/SATS/RA/CRY/L1B/",
    "lrm": {
        "search_pattern": "**/CS*SIN_1B__*.nc",  # Search pattern for L2 file discovery."
        "yyyymm_str_fname_indices": [-39, -31],
    },
    "sin": {
        "search_pattern": "**/CS*LRM_1B_*.nc",  # Search pattern for L2 file discovery."
        "yyyymm_str_fname_indices": [-48, -42],
    },
}

"""Cryotempo Land Ice dataset definition"""

# pylint: disable=R0801

dataset_definition = {
    "mission": "cs2",
    "long_name": "Cryotempo Land Ice",
    "time_epoch": "2000-01-01T00:00:00",
    "l2_dir": "${CPOM_DATA_DIR}/altimetry/cryotempo_li",
    "search_pattern": "**/CS_OFFL_SIR_TDP_LI*.nc",
    "yyyymm_str_fname_indices": [-48, -42],
    # -------------------#
    # --Core Parameters--#
    # -------------------#
    "latitude_param": "latitude",
    "longitude_param": "longitude",
    "elevation_param": "elevation",
    "power_param": "backscatter",
    "time_param": "time",
    "mode_param": "instrument_mode",
}

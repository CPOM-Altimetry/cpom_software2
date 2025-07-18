dataset_definition = {
    "long_name": "Cryotempo Land Ice",
    "time_epoch": "2000-01-01T00:00:00",
    "l2_dir": "${CPOM_DATA_DIR}/altimetry/cryotempo_li",
    
    # Search pattern for L2 file discovery."
    "search_pattern": "**/CS_OFFL_SIR_TDP_LI*.nc",
    # "int int : negative indices from end of L2 file name or path which point to the
    # YYYY start year in the name. For example with CryoTEMPO L2 files, this is: -48 -42"
    "yyyymm_str_fname_indices": [-48, -42],
    # Parameters to use in L2 netcdf files. If netcdf contains groups use use / to separate. example data/ku/backscatter
    "latitude_param": "latitude",
    "longitude_param": "longitude",
    "elevation_param": "elevation",
    "power_param": "backscatter",
    "time_param": "time",
    "mode_param": "instrument_mode",
}

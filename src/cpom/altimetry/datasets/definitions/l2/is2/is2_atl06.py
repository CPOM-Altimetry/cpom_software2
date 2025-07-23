# dataset_definition = {
#     "mission": "is2", 
#     "data_format": "hdf5",
#     "data_packed_in_blocks": False, 
#     "long_name": #TODO
#     "time_epoch": #TODO, 
#     "latency": "NTC",
#     # "l2_dir": "${CPOM_DATA_DIR}/altimetry/cryotempo_li",
#     # Search pattern for L2 file discovery."
#     "search_pattern": #TODO, 
#     # "int int : negative indices from end of L2 file name or path which point to the
#     "yyyymm_str_fname_indices": [-33,-25],
#     "latitude_param": "lat_mean",
#     "longitude_param": "lon_mean",
  
#     # Mission L2 surface type parameter names
#     "surf_type_param": "surf_type_20_ku"  # 1Hz Surface Type parameter name to use for ocean==0 discimination
    
#     "elevation_param": "h_li",
#     "uncertainty_param": "h_li_sigma",
#     "time_param": #TODO,
# }
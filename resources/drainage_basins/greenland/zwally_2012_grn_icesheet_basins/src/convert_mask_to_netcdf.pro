;Make a netCDF of the Zwally basin mask

PRO convert_mask_to_netcdf

RESTORE, '../basins/Zwally_GIS_basins_2km.sav'

gre_num_basins=N_ELEMENTS(gre_basin_names)

ncfile='../basins/Zwally_GIS_basins_2km.nc'

id=NCDF_CREATE(ncfile, /CLOBBER)

x_dim_id=NCDF_DIMDEF(id, 'gre_basin_nx', gre_basin_nx)
y_dim_id=NCDF_DIMDEF(id, 'gre_basin_ny', gre_basin_ny)
basin_dim_id=NCDF_DIMDEF(id, 'gre_num_basins', gre_num_basins)
strlen_dim_id=NCDF_DIMDEF(id, 'gre_basin_name_str_len', 4)

gre_basin_minxm_id=NCDF_VARDEF(id, 'gre_basin_minxm', /FLOAT)
gre_basin_minym_id=NCDF_VARDEF(id, 'gre_basin_minym', /FLOAT)
gre_basin_binsize_id=NCDF_VARDEF(id, 'gre_basin_binsize', /FLOAT)
gre_basin_names_id=NCDF_VARDEF(id, 'gre_basin_names', $
  [strlen_dim_id,basin_dim_id], /CHAR)
gre_basin_mask_id=NCDF_VARDEF(id, 'gre_basin_mask', [x_dim_id,y_dim_id], /FLOAT)

NCDF_CONTROL, id, /ENDEF

NCDF_VARPUT, id, gre_basin_minxm_id, gre_basin_minxm
NCDF_VARPUT, id, gre_basin_minym_id, gre_basin_minym
NCDF_VARPUT, id, gre_basin_binsize_id, gre_basin_binsize
NCDF_VARPUT, id, gre_basin_names_id, gre_basin_names
NCDF_VARPUT, id, gre_basin_mask_id, gre_basin_mask

NCDF_CLOSE, id

END

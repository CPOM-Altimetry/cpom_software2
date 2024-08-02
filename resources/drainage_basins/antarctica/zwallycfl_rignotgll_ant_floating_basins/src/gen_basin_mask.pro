;------------------------------------------------------------------------------------------
;+
; NAME: GEN_BASIN_MASK
;
; PURPOSE:
;		Generate mask grid for ice shelves, consisting of the Rignot GLL & islands and Zwally 2012 CFL + Zwally floating basin definitions.
; INPUTS:
;		Zwally 2012 Grounded+floating basins (for basins + Carving Front Location (CFL))
;		Rignot 2016 Grounded ice sheet basins (for GLL) + islands
;
; OUTPUTS:
;		


PRO GEN_BASIN_MASK, $
BINSIZE=binsize ; in m ie 5e3

; Load Zwally 2012 basin grid

zwally_gridfile='$RT_HOME/aux/drainage_basins/antarctica/zwally_2012_imbie1_ant_grounded_and_floating_icesheet_basins/basins/zwally_2012_imbie1_ant_grounded_and_floating_icesheet_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.sav'

restore,zwally_gridfile
 
; Load Rignot 2016 basin grid

rignot_gridfile='$RT_HOME/aux/drainage_basins/antarctica/rignot_2016_imbie2_ant_grounded_icesheet_basins/basins/rignot_2016_imbie2_ant_grounded_icesheet_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.sav'

restore,rignot_gridfile

nx = ant_zwally_basinmask_incfloating_ice_nx
ny = ant_zwally_basinmask_incfloating_ice_ny

ice_shelf_mask=make_array(nx,ny,/BYTE)
ice_shelf_mask[*,*]=0

FOR i=0, nx -1 DO BEGIN
	FOR j=0, ny -1 DO BEGIN
		IF ant_zwally_basinmask_incfloating_ice[i,j] gt 0 THEN ice_shelf_mask[i,j]=ant_zwally_basinmask_incfloating_ice[i,j]
		IF rignot_2016_mask[i,j] gt 0 THEN ice_shelf_mask[i,j]=0
	ENDFOR
ENDFOR

ice_shelf_mask_nx = nx
ice_shelf_mask_ny = ny
ice_shelf_minxm = ant_zwally_basinmask_incfloating_ice_minxm
ice_shelf_minym = ant_zwally_basinmask_incfloating_ice_minym
ice_shelf_binsize = ant_zwally_basinmask_incfloating_ice_binsize

save, filename='../basins/zwallycfl_rignotgll_ant_floating_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.sav', ice_shelf_mask, $
ice_shelf_mask_nx,  ice_shelf_mask_ny, ice_shelf_minxm,ice_shelf_minym, ice_shelf_binsize

print, 'saved as ../basins/zwallycfl_rignotgll_ant_floating_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.sav'

END

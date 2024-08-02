;------------------------------------------------------------------------------------------
;+
; NAME:
;      GEN_BASIN_MASK
; 
; PURPOSE:
;      generate a basin mask grid for the Zwally Antarctic Grounded + Floating basin masks 
;
; INPUTS:
;    Ant_Full_DrainageSystem_Polygons.txt from
;	http://icesat4.gsfc.nasa.gov/cryo_data/drainage_divides/Ant_Full_DrainageSystem_Polygons.txt
;
; OUTPUT:
;      IDL sav file containing the grid mask at specified posting in polar stereo grid
;      ../basins/ant_zwally_incfloating_basins_mask_<N>km.sav     
;

PRO GEN_BASIN_MASK,$
BINSIZE=binsize ; in m ie 5e3

; Setup polar stereo Antarctic projection 
this_proj = setup_projection('s')

restore,filename='.template'
d = read_ascii('../data/Ant_Full_DrainageSystem_Polygons.txt',template=template)
;   LAT             FLOAT     Array[901322]
;   LON             FLOAT     Array[901322]
;   BASIN           LONG      Array[901322]

; Define Antarctic PS Projection Grid
minxm=-2820e3
minym=-2420e3
grid_x_size=5640e3
grid_y_size=4840e3
	
nx=fix(grid_x_size/binsize)
ny=fix(grid_y_size/binsize)

outfile='../basins/zwally_2012_imbie1_ant_grounded_and_floating_icesheet_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.sav'	

print, 'nx=',nx, ' ny=',ny
print, outfile

ant_zwally_basinmask_incfloating_ice = make_array(nx,ny,/BYTE)
ant_zwally_basinmask_incfloating_ice[*,*]=0


FOR jj=1,27 DO BEGIN
	print, jj
	ok=where(d.basin eq jj)
	lats=d.lat[ok]
	lons=d.lon[ok]
	lons = (lons + 360.0) MOD 360.0
	
	map_latlon_xy, lats, lons, this_proj,px,py
		
   	pxmin=min(px)
     	pxmax=max(px)
     	pymin=min(py)
     	pymax=max(py)
     	
	FOR x=0,nx-1 DO BEGIN
	FOR y=0,ny-1 DO BEGIN
		IF ant_zwally_basinmask_incfloating_ice[x,y] gt 0 THEN continue
		
		xm = (x*binsize) + minxm
		ym = (y*binsize) + minym
		
   		IF xm gt pxmax THEN continue
		IF xm lt pxmin THEN continue
		IF ym gt pymax THEN continue
		IF ym lt pymin THEN continue		
			
		isinside = inside2([xm],[ym],px,py)
		IF isinside eq 1 THEN BEGIN
			ant_zwally_basinmask_incfloating_ice[x,y]=jj
		ENDIF
	ENDFOR
	ENDFOR
ENDFOR

;------------------------------------------------------------------------------------------------
; Save the mask file
;------------------------------------------------------------------------------------------------

ant_zwally_basinmask_incfloating_ice_nx=nx
ant_zwally_basinmask_incfloating_ice_ny=ny
ant_zwally_basinmask_incfloating_ice_minxm=minxm
ant_zwally_basinmask_incfloating_ice_minym=minym
ant_zwally_basinmask_incfloating_ice_binsize=binsize

print, 'Saved as ', outfile

save, filename=outfile,ant_zwally_basinmask_incfloating_ice,ant_zwally_basinmask_incfloating_ice_nx, ant_zwally_basinmask_incfloating_ice_ny, ant_zwally_basinmask_incfloating_ice_minxm, ant_zwally_basinmask_incfloating_ice_minym, ant_zwally_basinmask_incfloating_ice_binsize
	
END


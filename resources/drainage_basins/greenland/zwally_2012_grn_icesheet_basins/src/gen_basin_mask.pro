;------------------------------------------------------------------------------------------
;+
; NAME:
;      GEN_BASIN_MASK
; 
; PURPOSE:
;      generate a basin mask grid for the Greenland Zwally Basins 
;
; INPUTS:
;      ../data/Zwally_GIS_basins.txt file containing the basin polygons labelled
;       1.1 to 8.2
;
; OUTPUT:
;      IDL sav file containing the grid mask at specified posting in polar stereo grid
;      ../basins/Zwally_GIS_basins_<N>km.sav     
;      		each array contains a number 0..2
;

PRO GEN_BASIN_MASK, $
BINSIZE=binsize ; in m ie 5e3

IF N_ELEMENTS(binsize) lt 1 THEN BEGIN
	print,'usage, gen_basin_mask, binsize='
ENDIF

print,'binsize=', binsize

restore,'.data_template'
d=read_ascii('../data/Zwally_GIS_basins.txt',TEMPLATE=data_template)
names=d.field1
lats=d.field2
lons=d.field3

unique_names = names[uniq(names)]
print, unique_names
gre_basin_names=unique_names

n_basins = n_elements(unique_names)
n_points = n_elements(names)

; Define Greenland PS Projection Grid
minxm=-1000e3
minym=-3500e3
grid_x_size=2000e3
grid_y_size=3100e3
	
nx=fix(grid_x_size/binsize)
ny=fix(grid_y_size/binsize)
outfile='../basins/Zwally_GIS_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.sav'	

print, 'nx=',nx, ' ny=',ny
print, outfile

gre_basin_mask=make_array(nx,ny,/BYTE)
gre_basin_mask[*,*]=0


FOR x=0,nx-1 DO BEGIN
	print,x,nx
	FOR y=0,ny-1 DO BEGIN
		xm = (x*binsize) + minxm
		ym = (y*binsize) + minym
		
		FOR i=0, n_basins-1 DO BEGIN
			ok = where(names eq unique_names[i])
			this_lats=lats[ok]
			this_lons=lons[ok]
			
			mapll,this_lats, this_lons,'g',poly_x,poly_y
			IF xm lt min(poly_x) or xm gt max(poly_x) THEN continue
			IF ym lt min(poly_y) or ym gt max(poly_y) THEN continue
			pinside = inside2(xm,ym,poly_x,poly_y)
                        inside_ok = where(pinside gt 0,n_inside)

                        IF n_inside gt 0 THEN BEGIN
                              gre_basin_mask[x,y]=i+1
                        ENDIF			
		ENDFOR
	ENDFOR
ENDFOR

gre_basin_names=['None',gre_basin_names]

gre_basin_nx=nx
gre_basin_ny=ny
gre_basin_minxm=minxm
gre_basin_minym=minym
gre_basin_binsize=binsize

print, 'Saving mask as ', outfile

save, filename=outfile, gre_basin_mask, gre_basin_nx, gre_basin_ny, gre_basin_minxm, gre_basin_minym, gre_basin_binsize, gre_basin_names
	
END


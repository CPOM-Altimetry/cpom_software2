PRO CREATE_MASK

restore,filename='.template'
d = read_ascii('Ant_Full_DrainageSystem_Polygons.txt',template=template)

basinfilepath='$XOVER_HOME/aux/basins/imbie_ais_basins_r08_r11_z11.save'
restore,basinfilepath
basin_nx=nx  ; save these as nx,ny are overwritten by modelparfile restore
basin_ny=ny
basins = BASINS_Z11
mask_minx=-28025e2
mask_miny=-28025e2
mask_binsize = 5e3

basins[*,*]=0.0
help, basins

FOR jj=1,27 DO BEGIN
	print, jj
	ok=where(d.basin eq jj)
	nfound=0L
	
	; find basin polygon
	lats=d.lat[ok]
	lons=d.lon[ok]
	mapll,lats,lons,'s',px,py

	FOR i=0, 1120 DO BEGIN
		FOR j=0, 1120 DO BEGIN
			; if already found skip
			IF basins[i,j] gt 0.0 THEN CONTINUE
			
			; calculate lat, lon of grid point
		
			x =(i*mask_binsize)+mask_minx
			y =(j*mask_binsize)+mask_miny
			
			isinside = inside2([x],[y],px,py)
		
			IF isinside eq 1 THEN basins[i,j]=1.*jj
			IF isinside eq 1 THEN nfound++

		ENDFOR
	ENDFOR
	
	print, 'Number found for ',jj,nfound
	BASINS_Z11=basins
	save, filename='$XOVER_HOME/aux/basins/imbie_z_basins_including_floating_ice.sav',BASINS_Z11,mask_minx, mask_miny, mask_binsize, basin_nx,basin_ny

ENDFOR

BASINS_Z11=basins
save, filename='$XOVER_HOME/aux/basins/imbie_z_basins_including_floating_ice.sav',BASINS_Z11,mask_minx, mask_miny, mask_binsize, basin_nx,basin_ny

END

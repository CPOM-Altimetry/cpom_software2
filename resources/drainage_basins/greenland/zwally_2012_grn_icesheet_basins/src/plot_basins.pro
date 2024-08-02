PRO PLOT_BASINS, basinindex=basinindex, binsize=binsize

usage='plot_basins, binsize=5e3, [basinindex=n]'
IF N_ELEMENTS(binsize) eq 0 THEN BEGIN
	print, usage
	return
ENDIF


restore, '../basins/Zwally_GIS_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.sav'

window, xsize=gre_basin_nx, ysize=gre_basin_ny

FOR i=0, 20 DO BEGIN
	tvlct, 255, 10+10*i,0,i
ENDFOR
tvlct, 0, 0,0,0
tvlct, 255, 255,255,255
white=255

IF N_ELEMENTS(basinindex) gt 0 THEN BEGIN
	tvlct,0,255,0,basinindex
ENDIF 

cgimage, GRE_BASIN_MASK

xyouts,0.03,0.03,'Zwally_GIS_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km',color=white,charsize=1.4,/normal




print, GRE_BASIN_NAMES
print, 'N-basins ',n_elements(GRE_BASIN_NAMES)

outfile='../images/Zwally_GIS_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.gif'

saveimage, outfile

print, 'Saved as ', outfile

END

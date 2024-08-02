PRO PLOT_BASINS, basinindex=basinindex, binsize=binsize

usage='plot_basins, binsize=5e3, [basinindex=n]'
IF N_ELEMENTS(binsize) eq 0 THEN BEGIN
	print, usage
	return
ENDIF

restore, '../basins/GRE_IceSheet_IMBIE2_v1_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.sav'
window, xsize=gre_icesheet_nx, ysize=gre_icesheet_ny


tvlct, 0, 0,0,0
tvlct, 0, 0,255,1
tvlct, 0, 255,255,2
tvlct, 255, 255,255,255
white=255

IF N_ELEMENTS(basinindex) gt 0 THEN BEGIN
	tvlct,0,255,0,basinindex
ENDIF 

cgimage, GRE_ICESHEET_MASK

xyouts,0.05,0.05,'GRE_IceSheet_IMBIE2_v1_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km',color=white,charsize=1.4

print, GRE_ICESHEET_NAMES

outfile='../images/GRE_IceSheet_IMBIE2_v1_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.gif'

saveimage, outfile

print, 'Saved as ', outfile

END

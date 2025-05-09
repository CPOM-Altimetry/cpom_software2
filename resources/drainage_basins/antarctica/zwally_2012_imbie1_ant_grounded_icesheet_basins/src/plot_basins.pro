PRO PLOT_BASINS, basinindex=basinindex, binsize=binsize

usage='plot_basins, binsize=5e3, [basinindex=n]'
IF N_ELEMENTS(binsize) eq 0 THEN BEGIN
	print, usage
	return
ENDIF

restore, '../basins/zwally_2012_imbie1_ant_grounded_icesheet_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.sav'

window,xsize=ANT_ZWALLY_BASINMASK_GROUNDED_ICE_NX,ysize=ANT_ZWALLY_BASINMASK_GROUNDED_ICE_NY

FOR i=2, 27 DO BEGIN
	tvlct, 255, 10+10*i,0,i
ENDFOR
tvlct, 0, 0,255,1
tvlct, 0, 0,0,0
tvlct, 255, 255,255,255
white=255


IF N_ELEMENTS(basinindex) gt 0 THEN BEGIN
	tvlct,0,255,0,basinindex
ENDIF 
cgimage, ANT_ZWALLY_BASINMASK_GROUNDED_ICE


xyouts,0.05,0.05,'zwally_2012_imbie1_ant_grounded_icesheet_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km',color=white,charsize=1.4

outfile='../images/zwally_2012_imbie1_ant_grounded_icesheet_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.png'

saveimage, outfile

print, 'Saved as ', outfile

END


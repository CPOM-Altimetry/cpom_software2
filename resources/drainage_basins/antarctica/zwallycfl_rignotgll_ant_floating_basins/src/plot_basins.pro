PRO PLOT_BASINS, basinindex=basinindex, binsize=binsize

usage='plot_basins, binsize=5e3, [basinindex=n]'
IF N_ELEMENTS(binsize) eq 0 THEN BEGIN
	print, usage
	return
ENDIF

restore, '../basins/zwallycfl_rignotgll_ant_floating_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.sav'
help

window, xsize=ice_shelf_mask_nx,ysize=ice_shelf_mask_ny

FOR i=1, 9 DO BEGIN
	tvlct, 255, 0+10*i*20,0,i
ENDFOR
FOR i=10, 20 DO BEGIN
	tvlct, 0, 0+10*(i-10)*20,255,i
ENDFOR
FOR i=20, 27 DO BEGIN
	tvlct,  0+10*(i-20)*25,255,0,i
ENDFOR

tvlct, 0, 0,0,0
tvlct, 255, 255,255,255
white=255


IF N_ELEMENTS(basinindex) gt 0 THEN BEGIN
	tvlct,0,255,0,basinindex
ENDIF 
cgimage, ICE_SHELF_MASK


xyouts,0.03,0.03,'zwallycfl_rignotgll_ant_floating_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km',color=white,charsize=1.4

outfile='../images/zwallycfl_rignotgll_ant_floating_basins_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.png'

saveimage, outfile

print, 'Saved as ', outfile

END

;------------------------------------------------------------------------------------------
;+
; NAME:
;      GEN_BASIN_MASK
; 
; PURPOSE:
;      generate a basin mask grid for the Greenland GRE_IceSheet IMBIE2 mask 
;	mask shows either ICE_SHEET or ICE_CAP
;
; INPUTS:
;      ../shpfiles/GRE_IceSheet_IMBIE2_v1.shp   SHAPE file containing the icesheet polygons
;
; OUTPUT:
;      IDL sav file containing the grid mask at specified posting in polar stereo grid
;      ../basins/GRE_IceSheet_IMBE2_v1_<N>km.sav     
;      		each array contains a number 0..2
;

PRO GEN_BASIN_MASK, $
BINSIZE=binsize ; in m ie 5e3

IF N_ELEMENTS(binsize) lt 1 THEN BEGIN
	print,'usage, gen_basin_mask, binsize='
ENDIF


sfile=OBJ_NEW('IDLffshape','../shpfiles/GRE_IceSheet_IMBIE2_v1.shp')
sfile->IDLffshape::GetProperty, N_ENTITIES=num_ent, ENTITY_TYPE=ent_type
ents=sfile->IDLffshape::GetEntity(/ALL, /ATTRIBUTES)

;ENTS            STRUCT    = -> IDL_SHAPE_ENTITY Array[2]
;
;  SHAPE_TYPE      LONG                 5   (POLYGON)
;   ISHAPE          LONG                 0  (index 0..2)
;   BOUNDS          DOUBLE    Array[8]  (index 0=xmin, 1=ymin, 4=xmax, 5=ymax)
;   N_VERTICES      LONG             151698
;  VERTICES        POINTER   <PtrHeapVar2>
;   MEASURE         POINTER   <NullPointer>
;   N_PARTS         LONG               3334
;   PARTS           POINTER   <PtrHeapVar3>
;   PART_TYPES      POINTER   <NullPointer>
;   ATTRIBUTES      POINTER   <PtrHeapVar5>

;print, *ents[0].attributes   
;{ ICE_CAP}
;print, *ents[1].attributes   
;{ ICE_SHEET}


print, 'number of entities ', num_ent


; Define Greenland PS Projection Grid
minxm=-1000e3
minym=-3500e3
grid_x_size=2000e3
grid_y_size=3100e3
	
nx=fix(grid_x_size/binsize)
ny=fix(grid_y_size/binsize)

outfile='../basins/GRE_IceSheet_IMBIE2_v1_'+STRTRIM(STRING(fix(binsize/1e3)),2)+'km.sav'	

print, 'nx=',nx, ' ny=',ny
print, outfile


gre_icesheet_mask=make_array(nx,ny,/BYTE)
gre_icesheet_mask[*,*]=0

gre_icesheet_names=make_array(num_ent,/STRING)
FOR enum=0, num_ent-1 DO BEGIN
	a =  *ents[enum].attributes
	gre_icesheet_names[enum] =  a.attribute_0
ENDFOR

print, gre_icesheet_names
print, nx,ny

FOR x=0,nx-1 DO BEGIN
	print,x,nx
	FOR y=0,ny-1 DO BEGIN
		xm = (x*binsize) + minxm
		ym = (y*binsize) + minym
		
		; Check each entity in shapefile
		FOR enum=0, num_ent-1 DO BEGIN
		
			this_ent=ents[enum]
			
			all_points=*this_ent.vertices
			poly_indexes= *this_ent.parts
			
			nparts=this_ent.n_parts
			
			all_lon=all_points[0,*]
			all_lat=all_points[1,*]
			
			FOR np=0, nparts -1 DO BEGIN
				IF np lt (nparts-1) THEN BEGIN
					this_lon = all_lon[poly_indexes[np]:poly_indexes[np+1]-1]
					this_lat = all_lat[poly_indexes[np]:poly_indexes[np+1]-1]
				ENDIF ELSE BEGIN
					this_lon = all_lon[poly_indexes[np]:*]
					this_lat = all_lat[poly_indexes[np]:*]
				ENDELSE
		
				mapll,this_lat, this_lon,'g',poly_x,poly_y
				
				IF xm gt max(poly_x) THEN continue
				IF xm lt min(poly_x) THEN continue
				IF ym gt max(poly_y) THEN continue
				IF ym lt min(poly_y) THEN continue
				
				; Check if it is inside the actual polygon
				
				pinside = inside2(xm,ym,poly_x,poly_y)
				inside_ok = where(pinside gt 0,n_inside)
				
				IF n_inside gt 0 THEN BEGIN
					gre_icesheet_mask[x,y]=enum+1
					break
				ENDIF
			ENDFOR
		ENDFOR
	ENDFOR
ENDFOR

gre_icesheet_names = ['None',gre_icesheet_names]

gre_icesheet_nx=nx
gre_icesheet_ny=ny
gre_icesheet_minxm=minxm
gre_icesheet_minym=minym
gre_icesheet_binsize=binsize

print, 'Saving mask as ', outfile

save, filename=outfile, gre_icesheet_mask, gre_icesheet_nx, gre_icesheet_ny, gre_icesheet_minxm, gre_icesheet_minym, gre_icesheet_binsize, gre_icesheet_names
	
END


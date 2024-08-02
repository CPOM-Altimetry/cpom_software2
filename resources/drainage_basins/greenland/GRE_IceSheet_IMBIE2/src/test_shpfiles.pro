PRO TEST_SHPFILES

	sfile=OBJ_NEW('IDLffshape','../shpfiles/GRE_IceSheet_IMBIE2_v1.shp')
	sfile->IDLffshape::GetProperty, N_ENTITIES=num_ent, ENTITY_TYPE=ent_type
	ents=sfile->IDLffshape::GetEntity(/ALL, /ATTRIBUTES)
	
	help, ents, /str
	
	print, 'number of entities ', num_ent
	
	this_ent=ents[1]
	
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
						
		IF np eq 0 THEN plot, poly_x, poly_y 
		IF np gt 0 THEN oplot, poly_x, poly_y 
	ENDFOR	
	
	
END

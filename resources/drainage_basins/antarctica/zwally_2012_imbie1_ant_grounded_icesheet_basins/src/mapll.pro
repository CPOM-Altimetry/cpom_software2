;+
; NAME:
;    MAPLL
;
; PURPOSE:
;	map from latitude & longitude to x,y in polar stereo projection
;	uses a standard parallel - latitude with no distortion of 71S
;	for Antarctica and 70N for Arctic, 71N for Greenland (Bamber).
;
; AUTHOR:
;	??
;
; CALLING SEQUENCE:
;       MAPLL
;
; KEYWORD PARAMETERS:
;
;	ALAT:		 input latitude value in degrees
;
;	ALON:		 input longitude value in degrees (E?)
;
;	X:		 returned X value in (m?)
;
;	Y:		 returned Y value in (m?)
;
; COPYRIGHT:	University College London, 2010
;
;###########################################################################


pro mapll,alat,alon,hem,x,y
;Hughes Ellipsoid
re=6378.273e3
e2=0.006693883d0 
e=sqrt(e2)
;WGS 84
a = 6378137.0d0 ; WGS84
e2 = 0.00669437999015d0  ; WGS84
e = sqrt(e2)
re=a


IF (strupcase(hem) EQ 'S') THEN BEGIN	;southern hemisphere
slat=71.  ;Standard parallel - latitude with no distortion
sn=-1.0
xlam=0.
ENDIF

IF (strupcase(hem) EQ 'N') THEN BEGIN	;northern hemisphere
 xlam=45.
 sn=1.
 slat=70.
ENDIF

IF (strupcase(hem) EQ 'G') THEN  BEGIN	;Greenland (45W)
 xlam=45.
 sn=1.
 slat=70.
ENDIF

IF (strupcase(hem) EQ 'JLB') THEN  BEGIN ; Greenland (39W), Bamber
 xlam=39.
 sn=1.
 slat=71.
ENDIF

alat=sn*alat
alon=sn*alon
rlat=alat
t1=tan(!pi/4.-rlat*!dtor/2.)/((1.0-e*sin(rlat*!dtor))/$
(1.0+e*sin(rlat*!dtor)))^(e/2.)
t2=tan(!pi/4.-slat*!dtor/2.)/((1.0-e*sin(slat*!dtor))/$
(1.0+e*sin(slat*!dtor)))^(e/2.)
cm=cos(slat*!dtor)/sqrt(1.0-e2*(sin(slat*!dtor)^2))
rho=re*cm*t1/t2
x=float( rho*sn*sin((alon+xlam)*!dtor))
y=float(-rho*sn*cos((alon+xlam)*!dtor))
alat=sn*alat
alon=sn*alon
end

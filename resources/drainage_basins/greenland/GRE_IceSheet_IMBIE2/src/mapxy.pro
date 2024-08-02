; This code provided by Eric Rignot
;
; Example call : mapxy,x,y,'s',lat,lon

Pro mapxy,x,y,hem,alat,alon

re = 6378137.0d0 ; WGS84
e2 = 0.00669437999015d0  ; WGS84

e=sqrt(e2)

; Antarctic Projection
IF (hem EQ 's') THEN BEGIN
	slat=71. ;Standard parallel = latitude with no distortion
	sn=-1.0
	xlam=0.
ENDIF

; Arctic Projection
IF (hem EQ 'n') THEN BEGIN
	slat=70.
	sn=1.
	xlam=45.
ENDIF

; Greenland Projection (45W)
IF (hem EQ 'g') THEN BEGIN
	slat=70.
	sn=1.
	xlam=45.
ENDIF

; Greenland Projection (Bamber, 39W)
IF (hem EQ 'jlb') THEN BEGIN
	slat=71.
	sn=1.
	xlam=39.
ENDIF

rho=sqrt(x^2+y^2)
cm=cos(slat*!dtor)/sqrt(1.0-e2*(sin(slat*!dtor)^2))
t=tan((!pi/4)-(slat*!dtor/2.))/((1.0-e*sin(slat*!dtor))/(1.0+e*sin(slat*!dtor)))^(e/2.)
t=rho*t/(re*cm)
chi=(!pi/2.)-2.*atan(t)
alat=chi+((e2/2.)+(5.0*e2^2/24.)+(e2^3/12.))*sin(2*chi)+$
((7.0*e2^2/48.)+(29.*e2^3/240.))*sin(4.0*chi)+$
(7.0*e2^3/120.)*sin(6.0*chi)
alat=float(sn*alat/!Dtor)
xpr=sn*x
ypr=sn*y
alon=atan(xpr,-ypr)/!dtor-sn*xlam
alon=sn*alon

indice=where(alon LT 0)
if (indice(0) NE (-1)) then alon(indice)=alon(indice)+360.
indice=where(alon GT 360)
if (indice(0) NE (-1)) then alon(indice)=alon(indice)-360.

end

FUNCTION Inside, x, y, px, py, Index=index

; Purpose: see if point is inside polygon
; Category: maths
; Input: x, y - [vector of] points
;        px,py - points defining polygon (will be closed automatically)
; Output: vector of 1's and 0's
;         OR
;         indicies of points inside (if /Index is set)
; Author: "Bård Krane" <bard.krane@fys.uio.no>
; Mods: wmc - make it work with x, y as vectors
; More-info: posted to comp.lang.idl-pvwave on Wed, 01 Apr 1998 12:26:38 +0200
; See-also: http://www.ecse.rpi.edu/Homepages/wrf/geom/pnpoly.html for another method, possibly better
; This is better than my routine "is_inside"
; Note: reduce test from 1e-8 to 1e-4, since we are usually in single precision. In fact, "0.1"
;       would do as well.

    On_Error, 1

    sx = Size(px)
    sy = Size(py)
    IF (sx[0] EQ 1) THEN NX = sx[1] ELSE Message,'Variable px is not a vector'
    IF (sy[0] EQ 1) THEN NY = sy[1] ELSE Message,'Varialbe py is not a vector'
    IF (NX EQ NY)   THEN N = NX ELSE Message,'Incompatible vector dimensions'

    tmp_px =  [px, px[0]]                           ; Close Polygon in x
    tmp_py =  [py, py[0]]                           ; Close Polygon in y

    i  = Indgen(N,/Long)                            ; indices 0...N-1
    ip = Indgen(N,/Long) + 1                        ; indices 1...N

    nn = N_Elements(x)
    X1 = tmp_px(i)  # Replicate(1,nn) - Replicate(1,n) # Reform([x],nn)
    Y1 = tmp_py(i)  # Replicate(1,nn) - Replicate(1,n) # Reform([y],nn)
    X2 = tmp_px(ip) # Replicate(1,nn) - Replicate(1,n) # Reform([x],nn)
    Y2 = tmp_py(ip) # Replicate(1,nn) - Replicate(1,n) # Reform([y],nn)

    dp = X2*X1 + Y1*Y2                               ; Dot-product
    cp = X1*Y2 - Y1*X2                               ; Cross-product
    theta = Atan(cp,dp)

    ret = Replicate(0L, N_Elements(x))
    i = Where(Abs(Total(theta,1)) GT 0.01,count)
    IF (count GT 0) THEN ret(i)=1
    IF (N_Elements(ret) EQ 1) THEN ret=ret[0]

    IF (Keyword_Set(index)) THEN ret=(Indgen(/Long, N_Elements(x)))(Where(ret eq 1))

    RETURN, ret

END


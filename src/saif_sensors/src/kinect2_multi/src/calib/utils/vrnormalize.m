function vec_n = vrnormalize(vec, maxzero)
%VRNORMALIZE Normalize a vector.
%   Y = VRNORMALIZE(X,MAXZERO) returns a unit vector Y parallel to the 
%   input vector X. Input X can be vector of any size. If the modulus of
%   the input vector is <= MAXZERO, the output is set to zeros(size(X)).
%
%   Not to be called directly.

%   Copyright 1998-2007 HUMUSOFT s.r.o. and The MathWorks, Inc.
%   $Revision: 1.1.6.1 $ $Date: 2007/06/04 21:16:04 $ $Author: batserve $

norm_vec = norm(vec);
if (norm_vec <= maxzero)
  vec_n = zeros(size(vec));
else
  vec_n = vec ./ norm_vec;
end
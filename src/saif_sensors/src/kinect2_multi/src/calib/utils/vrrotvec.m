function r = vrrotvec(a, b, options)
%VRROTVEC Calculate a rotation between two vectors.
%   R = VRROTVEC(A, B) calculates a rotation needed to transform 
%   a 3d vector A to a 3d vector B.
%
%   R = VRROTVEC(A, B, OPTIONS) calculates the rotation with the default 
%   algorithm parameters replaced by values defined in the structure
%   OPTIONS.
%
%   The OPTIONS structure contains the following parameters:
%
%     'epsilon'
%        Minimum value to treat a number as zero. 
%        Default value of 'epsilon' is 1e-12.
%
%   The result R is a 4-element axis-angle rotation row vector.
%   First three elements specify the rotation axis, the last element
%   defines the angle of rotation.
%
%   See also VRROTVEC2MAT, VRROTMAT2VEC, VRORI2DIR, VRDIR2ORI.

%   Copyright 1998-2007 HUMUSOFT s.r.o. and The MathWorks, Inc.
%   $Revision: 1.1.6.2 $ $Date: 2007/06/04 21:15:52 $ $Author: batserve $

% test input arguments
error(nargchk(2, 3, nargin, 'struct'));

if any(~isreal(a) || ~isnumeric(a))
  error('VR:argnotreal','Input argument contains non-real elements.');
end

if (length(a) ~= 3)
  error('VR:argwrongdim','Wrong dimension of input argument.');
end

if any(~isreal(b) || ~isnumeric(b))
  error('VR:argnotreal','Input argument contains non-real elements.');
end

if (length(b) ~= 3)
  error('VR:argwrongdim','Wrong dimension of input argument.');
end

if nargin == 2
  % default options values
  epsilon = 1e-12;
else
  if ~isstruct(options)
     error('VR:optsnotstruct','OPTIONS is not a structure.');
  else
    % check / read the 'epsilon' option
    if ~isfield(options,'epsilon') 
      error('VR:optsfieldnameinvalid','Invalid OPTIONS field name(s).'); 
    elseif (~isreal(options.epsilon) || ~isnumeric(options.epsilon) || options.epsilon < 0)
      error('VR:optsfieldvalueinvalid','Invalid OPTIONS field(s).');   
    else
      epsilon = options.epsilon;
    end
  end
end

% compute the rotation, vectors must be normalized
an = vrnormalize(a, epsilon);
bn = vrnormalize(b, epsilon);
axb = vrnormalize(cross(an, bn), epsilon);
ac = acos(dot(an, bn));

% Be tolerant to column vector arguments, produce a row vector
r = [axb(:)' ac];






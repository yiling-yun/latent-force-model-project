function R = rot2D(a, b)
% ROT2D  Compute 2×2 rotation matrix that rotates vector a to vector b
%
%   R = rot2D(a, b)
%
%   a, b : 2×1 vectors (need not be unit length)

    % Normalize
    a = a(:) / norm(a);
    b = b(:) / norm(b);

    % Compute angle between them
    cosTheta = dot(a, b);
    sinTheta = a(1)*b(2) - a(2)*b(1);   % 2D cross product (scalar)

    % Rotation matrix
    R = [cosTheta, -sinTheta;
         sinTheta,  cosTheta];


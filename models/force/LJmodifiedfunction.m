% modified LJ function.
function fnew = LJmodifiedfunction(f)

Fmax = 2;%default:2 (r: 0.3903); % 1 (0.4049)

signf = f>Fmax; % repulsive force
fnew = signf.* (Fmax*tanh(f/Fmax)) + (1-signf).*f;  

% fnew = f; 


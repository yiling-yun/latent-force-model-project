% modified LJ function.
function fnew = LJmodifiedfunction(f)

Fmax = 2; 

signf = f>Fmax; % repulsive force
fnew = signf.* (Fmax*tanh(f/Fmax)) + (1-signf).*f;  




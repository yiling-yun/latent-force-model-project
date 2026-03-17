% composite force from two sources, interactive and self combined, to predict the
% trajectory
function [apos,bpos,f_list,fs_list] = LJfuncCompforceGen1(Parameters,aposgentransf,aposprevf,aposobsf,bposobsf,aposprev,aposobs,bposobs,scaler)

% single traj self force
epsilon1 = Parameters(1);   %% strength in L-J potential
sigma1 = Parameters(2); %% repul dist in L-J potential
bcoef1 = Parameters(3);  %% attractive coefficient in L-J potential

% interactive force
epsilon2 = Parameters(4);   %% strength in L-J potential
sigma2 = Parameters(5); %% repul dist in L-J potential
bcoef2 = Parameters(6);  %% attractive coefficient in L-J potential

% scale to the parameter esimation space
if nargin < 4
    scaler = 100; 
end;
% agent B traj
aposobs = aposobs/scaler;
bposobs = bposobs/scaler;
% force video
aposobsf = aposobsf/scaler;
bposobsf = bposobsf/scaler;

% List of forces 
n = size(aposobs,1); dt = 1;

f_list = zeros(n,1);
fs_list = zeros(n,1);

apos = []; bpos = [];

vvec(1,:) = [0 0];
avec(1,:) = [0 0];
avec(2,:) = [0 0];

bpos = (bposobs);

unit_vectorint(1,1:2) = [0 0]; 
unit_vectorself(1,1:2) = [0 0]; 


for k = 1:n
    %% interactive force
    ba = aposgentransf(k,:)-bpos(k,:); % repulsive force, postive; attractive force, negative
    r = norm(ba); 

    r_list(k,:) = r; 
    if r<eps
        epsilon2 = 0;
        force2 = [0 0];
        f2 = 0; 
    else
        unit_vectorint = ba /r; 
       
        % force as negative gradient of lennard jones potential
        kk = 12;
        ww = 6;
        f2 = 48*epsilon2*(((sigma2^kk)/r^(kk+1))-(bcoef2*(sigma2^ww)/r^(ww+1)));
        f2 = LJmodifiedfunction(f2);  
        force2 = unit_vectorint * f2;
    end;
    f_list2(k,:)= force2;

   
    %% combine
    force = force2;
    f_list(k) = f2;
    fs_list(k) = 0;

    avec(k,:) = force*0.1; 
    vvec(k,:) = [0 0] + avec(k,:)*dt;
    apos(k,:) = aposgentransf(k,:) + vvec(k,:)*dt;
end

% scale to the parameter esimation space
apos = apos*scaler;
bpos = bpos*scaler;


% composite force from two sources, interactive and self combined, to predict the
% trajectory
function [apos,bpos,f_list,fs_list] = LJfuncCompforce(Parameters,aposprev,aposobs,bposobs,scaler)

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
aposobs = aposobs/scaler;
bposobs = bposobs/scaler;

distobsself = sqrt((aposobs(2:end,1)-aposobs(1:(end-1),1)).^2+(aposobs(2:end,2)-aposobs(1:(end-1),2)).^2);

% List of forces 
n = size(aposobs,1); dt = 1;

f_list = zeros(n,2);
fs_list = zeros(n,1);

apos = []; bpos = [];

apos(1,:) = aposprev(round(n/2),:);   % generate postion of "a" from prev predited
apos(2,:) = aposprev(round(n/2)+1,:);   % generate postion of "a" from prev predited
velocity1 = apos(2,:)-apos(1,:);
aorig = aposprev(round(n/2),:);   % generate postion of "a" from prev predited

vvec(1,:) = [0 0];
if mean(distobsself)<0.01   % no movements
    vvec(2,:) = [0 0];
else
    vvec(2,:) = velocity1(1, :);
end;
avec(1,:) = [0 0];
avec(2,:) = [0 0];

bpos = (bposobs);

unit_vectorint(1,1:2) = [0 0]; 
unit_vectorself(1,1:2) = [0 0]; 
for k = 2:size(aposobs,1)
    %% interactive force
    ba = aposobs(k,:)-bposobs(k,:); % repulsive force, postive; attractive force, negative
    r = norm(ba); %$$
    unit_vectorint(k,:) = ba /r; 

    %% self force
    ba = aposobs(k,:)-aposobs(k-1,:); % repulsive force, postive; attractive force, negative
    r = norm(ba); %$$
    if r == 0
        rotmat{k} = [0 0; 0 0];
    else
        unit_vectorself(k,:) = ba /r;     
        rotmat{k} = rot2D(unit_vectorint(k,:)', unit_vectorself(k,:)');  
    end;
end;


for k = 3:n
    %% interactive force
    % unit vector
    ba = apos(k-1,:)-bpos(k-1,:); % repulsive force, postive; attractive force, negative
    r = norm(ba); 
    r_list(k,:) = r; 
    if r<eps
        epsilon2 = 0;
        force2 = [0 0];
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

    %% single traj force
    % unit vector
    ba = apos(k-1,:)-apos(k-2,:);% aorig; %   
    r = norm(ba); %    

    r_list1(k,:) = r; 
    if r<eps
        epsilon1 = 0;
        force1 = [0 0];
    else
        unit_vector = transpose(rotmat{k}*unit_vectorint');

        % force as negative gradient of lennard jones potential
        kk = 12;
        ww = 6;
        f1 = 48*epsilon1*(((sigma1^kk)/r^(kk+1))-(bcoef1*(sigma1^ww)/r^(ww+1)));
        f1 = LJmodifiedselffunction(f1);  
        force1 = unit_vector * f1;
    end;
    f_list1(k,:)= force1;

    %% combine
    force = force2 + force1; 
    f_list(k,:) = force;
    fs_list(k) = norm(force);

    avec(k,:) = force*0.1; 
    vvec(k,:) = vvec(k-1,:) + avec(k,:)*dt;
    apos(k,:) = apos(k-1,:) + vvec(k,:)*dt;
end


% scale to the parameter esimation space
apos = apos*scaler;
bpos = bpos*scaler;


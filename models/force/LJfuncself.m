function [apos,f_list,fs_list] = LJfuncself(Parameters,aposprev,aposobs,scaler)

mode = 'self';
epsilon = Parameters(1);   %% strength in L-J potential
sigma = Parameters(2); %% repul dist in L-J potential
bcoef = Parameters(3);  %% attractive coefficient in L-J potential

% scale to the parameter esimation space
aposobs = aposobs/scaler;
aposprev = aposprev/scaler;

distobsself = sqrt((aposobs(2:end,1)-aposobs(1:(end-1),1)).^2+(aposobs(2:end,2)-aposobs(1:(end-1),2)).^2);

% List of forces 
n = size(aposobs,1); dt = 1;

f_list = zeros(n,2);
fs_list = zeros(n,1);

apos = []; 

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


for k = 3:n 
    % unit vector
    bpos = apos(k-2,:);
    ba = apos(k-1,:) - bpos;  
    baobs = aposobs(k-1,:) - aposobs(k-2,:); 

    r = norm(ba); 
    if r<eps && norm(baobs)<eps
        unit_vector = [0 0];
        f = 0;
    else
        if r<eps && norm(baobs)>eps   % the boundary case when stationary in first few frames, then move within a temporal window
            unit_vector = baobs/norm(baobs); 
            r = norm(baobs);
        else
            unit_vector = ba /r; 
        end;
        r_list(k,:) = r;         

        % force as negative gradient of lennard jones potential
        kk = 12;
        ww = 6;
        f = 48*epsilon*(((sigma^kk)/r^(kk+1))-(bcoef*(sigma^ww)/r^(ww+1)));
        f = LJmodifiedselffunction(f);  
    end
    
    force = unit_vector * f;
    f_list(k,:)= unit_vector*f;
    fs_list(k) = f;

    avec(k,:) = force*0.1;  % a = f/m;
    vvec(k,:) = vvec(k-1,:) + avec(k,:)*dt;
    apos(k,:) = apos(k-1,:) + vvec(k,:)*dt;
end

% scale to the parameter esimation space
apos = apos*scaler;



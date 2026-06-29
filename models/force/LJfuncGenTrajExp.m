function [apos,bpos,f_list,fs_list] = LJfuncGenTrajExp(Parameters,is_first_window,aposprev,aposobs_force, bposobs_traj, mode, scaler)

% single traj self force
epsilon_selfA = Parameters(1);   %% strength in L-J potential
sigma_selfA = Parameters(2); %% repul dist in L-J potential
bcoef_selfA = Parameters(3);  %% attractive coefficient in L-J potential

% interactive force
epsilon_inter = Parameters(4);   %% strength in L-J potential
sigma_inter = Parameters(5); %% repul dist in L-J potential
bcoef_inter = Parameters(6);  %% attractive coefficient in L-J potential

% scale to the parameter esimation space
if nargin < 7 %YY: 10/06/2025 changed from 6 to 7 when I added the mode argument
    scaler = 100; 
end
% aposobs_traj = aposobs_traj/scaler; % YY: video B agent
bposobs_traj = bposobs_traj/scaler; % YY: trajectory video, agent B
aposobs_force = aposobs_force/scaler; % YY: force video, agent A
aposprev = aposprev/scaler; % YY: video B agent previous

% List of forces 
n = size(aposobs_force,1); dt = 1;

f_list = zeros(n,2);
fs_list = zeros(n,1);

apos = []; bpos = [];

if is_first_window
    apos(1,:) = aposobs_force(1,:); % generate postion of "a"
    velocity_force_list = diff(aposobs_force,1,1); % YY: the second step of the first window is based on force video, agent A
    velocity_force = velocity_force_list(1,:);
else
    apos(1,:) = aposprev(round(n/2),:); % generate postion of "a"
    velocity_force_list = diff(aposprev,1,1); % YY: the second step of other windows is based on previously predicted position, agent A
    velocity_force = velocity_force_list(round(n/2),:);
    % velocity_force_list = diff(aposobs_force,1,1); % YY: the second step of the first window is based on force video, agent A
    % velocity_force = velocity_force_list(1,:); 
end
% apos(2,:) = aposprev(round(n/2) + 1,:); %YY
apos(2,:) = apos(1,:)+ velocity_force;   % generate postion of "a"

vvec(1,:) = [0 0];
vvec(2,:) = apos(2,:)  - apos(1,:) ;%YY
% vvec(2,:) = velocity_force;
avec(1,:) = [0 0];
avec(2,:) = [0 0];

bpos = (bposobs_traj);

for k = 3:n
    %% interactive force
    % unit vector
    vector_inter = apos(k-1,:)-bpos(k-1,:); % repulsive force, postive; attractive force, negative. The force direction must match estimation.
    r_inter = norm(vector_inter); 
    % r_list(k,:) = r; 
    % if r_inter<eps
    %     epsilon_inter = 0;
    %     force_inter = [0 0];
    % else
        unit_vector_inter = vector_inter /r_inter; 
    
        % force as negative gradient of lennard jones potential
        kk = 12;
        ww = 6;
        f_inter = 48*epsilon_inter*(((sigma_inter^kk)/r_inter^(kk+1))-(bcoef_inter*(sigma_inter^ww)/r_inter^(ww+1)));
    
        force_inter = unit_vector_inter * f_inter;
    % end
    % f_list2(k,:)= force2;

    %% single traj force
    % unit vector
    % aorig_force = aposobs_force(1,:);
    % ba = aposobs_force(k-1,:)-aorig_force; % repulsive force, postive; attractive
    % force, negative 
    vector_selfA = aposobs_force(k-1,:) - aposobs_force(1,:);  %%%$$$$$$
    % vector_selfA = apos(k-1,:) - apos(1,:);%YY: 10/21/2025
    % vector_selfA = aposobs_force(k-1,:) - [2.35 1.50];%YY: 10/21/2025
    % %YY: start with the first frame of first window? encircle will have
    % issue
    % vector_selfA = apos(k-1,:) - apos(1,:);
    % vector_selfA = apos(k-1,:) - aposobs_force(1,:);
    r_selfA = norm(vector_selfA); %

    % r_list1(k,:) = r; 
    % if r_selfA<eps
    %     epsilon_selfA = 0;
    %     force_selfA = [0 0];
    % else
    unit_vector_selfA = vector_selfA /r_selfA; 

    % force as negative gradient of lennard jones potential
    kk = 12;
    ww = 6;
    f_selfA = 48*epsilon_selfA*(((sigma_selfA^kk)/r_selfA^(kk+1))-(bcoef_selfA*(sigma_selfA^ww)/r_selfA^(ww+1)));

    force_selfA = unit_vector_selfA * f_selfA;
    % end
    % f_list1(k,:)= force1;

    %% combine
    if strcmp(mode, "interactiveOnly")
        force = force_inter;
    elseif strcmp(mode, "selfOnly")
        force = force_selfA;
    else
        force = force_inter + force_selfA; 
    end
    f_list(k,:) = force;
    fs_list(k) = norm(force);

    if fs_list(k)>10
        force = 0;
    end      

    avec(k,:) = force*0.1; 
    vvec(k,:) = vvec(k-1,:) + avec(k,:)*dt;
    apos(k,:) = apos(k-1,:) + vvec(k,:)*dt;
end

% scale to the parameter esimation space
apos = apos*scaler;
bpos = bpos*scaler;

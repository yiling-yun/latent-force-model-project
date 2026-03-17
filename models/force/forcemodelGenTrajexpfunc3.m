% force model: use the Leonard Jones Potential
% r is the distance between two interacting particles, 
% epsilon  governs the strength of the interaction, a measure of how strongly two entities attract each other
% sigma  is the distance at which the particle-particle potential energy V is zero (often referred to as 'size of the particle'). 
% The Lennard-Jones potential has its minimum at a distance of r=2^{1/6}*sigma 
% where the potential energy has the value {\displaystyle V=-eps.

function [aposgen,bposgen,aposobs_traj,bposobs_traj,force] = forcemodelGenTrajexpfunc3(dataraw_traj,dataraw_force,estpara, mode)
% YY: trajectory video, force video, force field 

framenum = size(dataraw_traj,1); 
intv = 5;%  % temporal window +/- intv.

% countfi = 0; 
framesall = 1+intv:intv:framenum-intv;
for i = 1:size(estpara,1)
    framerange = framesall(i)-intv:framesall(i)+intv;
    aposobs_traj{i} = dataraw_traj(framerange,1:2); % YY: trajectory video, agent A 
    aposobs_force{i} = dataraw_force(framerange,1:2); % YY: force video, agent A
    if i>1      
        is_first_window = false;
        aposprev{i} = aposgen{i-1};
    else
        is_first_window = true;

        %YY
        % new_start = aposobs_force{1}(1,:) + [1000, 0];
        % aposprev{i} = aposobs_force{1}*0 + new_start;
        aposprev{i} = aposobs_force{1}*0+aposobs_force{1}(1,:); % YY: for the first window, all use the first frame of trajectory video, agent A (should be the same as the force video, agent A)
    end
    bposobs_traj{i} = dataraw_traj(framerange,3:4); % YY: trajectory video, agent B
    [aposgen{i},bposgen{i},force{i}] = LJfuncGenTrajExp(estpara(i,:), is_first_window, aposprev{i},aposobs_force{i},bposobs_traj{i}, mode);
end


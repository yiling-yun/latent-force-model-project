function ep1 = process_one_v1_video(data_i)
% Per-video V1 estimation, factored out so parfor / parfeval workers can call it.
% Mirrors the V1 fitting in main_force_model_est.m: two-stage estimate, first
% using observed previous positions, then cascading with generated positions.

[eA]  = forcemodelprevobsA(data_i);
[eB]  = forcemodelprevobsB(data_i);
eObs  = [eA, eB];
[eA2] = forcemodelgenprevA(eObs, data_i);
[eB2] = forcemodelgenprevB(eObs, data_i);
ep1   = [eA2, eB2];
end

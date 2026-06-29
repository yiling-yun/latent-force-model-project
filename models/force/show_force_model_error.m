% show_force_model_error.m
% Report per-video error between OBSERVED and force-model GENERATED
% trajectories for the V2 force feature (estpara_v2).
%
% Loads the saved V2 estimate (estpart_forcemodel.v2_improved.mat) produced
% by main_force_model_est_par_v2.m, rebuilds the same padded trajectory
% matrices, and computes the obs-vs-generated error with compute_v1_v2_errors
% (the SAME cascade convention the V2 fit minimizes). No re-estimation.
%
% Prints a table sorted worst-first and writes:
%   rst/exp2/force_model_v2_error.csv
%
% Columns (pixel units):
%   posA  posB  : Agent A / B position error (obs vs generated)
%   velA  velB  : Agent A / B per-frame velocity error
%   posMean     : (posA + posB)/2, the sort key

clear all;
close all;

%% ============== SETTINGS (match the V2 estimator) ============
input   = 'stim';   % exp2 = 54 selected videos
reverseflag = 0;    % must match how estpara_v2 was generated
scaler  = 100;
intv    = 5;

%% ============== PATHS ========================================
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir); script_dir = pwd; end
addpath(script_dir);

xlsx_file = fullfile(script_dir, 'charades_traj_summary.xlsx');
savedir   = fullfile(script_dir, 'rst', 'exp2');
v2_file   = fullfile(savedir, 'estpart_forcemodel.v2_improved.mat');

%% ============== LOAD SAVED V2 ESTIMATE =======================
if ~exist(v2_file, 'file')
    error('V2 estimate not found: %s\nRun main_force_model_est_par_v2.m first.', v2_file);
end
s = load(v2_file, 'estpara_v2');
if ~isfield(s, 'estpara_v2')
    error('%s has no variable ''estpara_v2''.', v2_file);
end
estpara_v2 = s.estpara_v2;
fprintf('Loaded estpara_v2 (%d videos) from %s\n', numel(estpara_v2), v2_file);

%% ============== LOAD TRAJECTORY DATA (stim/exp2) =============
[num, txt, ~] = xlsread(xlsx_file, 'selected_1133');
[numS, ~, ~]  = xlsread(xlsx_file, 'selected_exp2');
[~, txtS, ~]  = xlsread(xlsx_file, 'selected_exp2');
vidnum   = length(numS);
semlabel = cell(vidnum, 1);
cordcell = cell(vidnum, 6);
for i = 1:vidnum
    rowi = find(numS(i) == num(:,2));
    semlabel{i} = txtS{i};
    for j = 1:6
        temp  = txt{rowi+1, 3+j};
        temp2 = str2double(strsplit(temp(3:end-2), ''', '''));
        cordcell{i,j} = temp2;
    end
end

%% ============== BUILD PADDED TRAJECTORY MATRICES =============
dataraw_all = cell(vidnum, 1);
for i = 1:vidnum
    if reverseflag == 0
        x1 = cordcell{i,1}; y1 = cordcell{i,2};
        x2 = cordcell{i,4}; y2 = cordcell{i,5};
    else
        x1 = fliplr(cordcell{i,1}); y1 = fliplr(cordcell{i,2});
        x2 = fliplr(cordcell{i,4}); y2 = fliplr(cordcell{i,5});
    end
    dataraw = [x1' y1' x2' y2'];
    datapad = repmat([x1(1) y1(1) x2(1) y2(1)], 11, 1);
    dataraw_all{i} = [datapad; dataraw];
end

%% ============== COMPUTE OBS-vs-GENERATED ERROR (V2) ==========
n = min(vidnum, numel(estpara_v2));
posA = zeros(n,1); velA = zeros(n,1);
posB = zeros(n,1); velB = zeros(n,1);
for i = 1:n
    ep = estpara_v2{i};
    if isempty(ep)
        posA(i)=NaN; velA(i)=NaN; posB(i)=NaN; velB(i)=NaN;
        continue;
    end
    % Pass V2 as both args; take the V2 half of the error vector.
    % errs = [posA_v1 velA_v1 posB_v1 velB_v1 posA_v2 velA_v2 posB_v2 velB_v2]
    errs = compute_v1_v2_errors(ep, ep, dataraw_all{i}, intv, scaler);
    posA(i) = errs(5); velA(i) = errs(6);
    posB(i) = errs(7); velB(i) = errs(8);
end
posMean = (posA + posB) / 2;

%% ============== TABLE (sorted worst-first) ===================
vidID = numS(1:n);
T = table((1:n).', vidID(:), string(semlabel(1:n)), ...
          posA, posB, velA, velB, posMean, ...
          'VariableNames', {'idx','videoID','label','posA','posB','velA','velB','posMean'});
T = sortrows(T, 'posMean', 'descend');

fprintf('\nForce model V2 — observed vs generated error (pixels), worst-first:\n\n');
disp(T);

fprintf('Overall mean:  posA=%.1f  posB=%.1f  velA=%.2f  velB=%.2f  posMean=%.1f\n', ...
        mean(posA,'omitnan'), mean(posB,'omitnan'), ...
        mean(velA,'omitnan'), mean(velB,'omitnan'), mean(posMean,'omitnan'));

%% ============== SAVE CSV =====================================
out_csv = fullfile(savedir, 'force_model_v2_error.csv');
writetable(T, out_csv);
fprintf('\nSaved table to %s\n', out_csv);

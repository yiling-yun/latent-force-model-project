% main_force_model_est_par_v2.m
% Parallel V2 estimation runner — V2 counterpart to main_force_model_est_par.m.
% Calls the *_v2 force-model fitters and saves under the V2 file/variable names
% used by compare_est_v1_v2.m and main_forceExpdissimmatExp2.m:
%   - estpart_forcemodel.v2_improved.mat        (variable: estpara_v2)
%   - estpart_forcemodel.v2_improved.prevobs.mat (variable: estparaObsprev_v2)
%   - estpart_forcemodel.v2_all.mat     (variable: estpara)
%   - per-video plot V2_trialtrial-<i>.png       (when plotflagtraj=1)
%
% Drop-in: produces the same file compare_est_v1_v2.m loads as v2_file, so
% you can run this once and then let compare_est_v1_v2 skip V2 estimation
% (it sees (loaded) for every video).

clear all;
close all;

%% ============== USER SETTINGS ===============================
input        = 'stim'; % 'all' (1133 videos) or 'stim' (54 selected)
reverseflag  = 0;      % 1: reverse frame order (Exp 3); 0: Exp 1
plotflagtraj = 0;      % 1: save per-video trajectory PNG; 0: skip
use_parfor   = true;
nworkers     = 32;     % 0 = MATLAB default pool size
force_recompute = false; % true: always re-estimate even if outputs already exist

rng(1);
scaler = 100;
intv   = 5;

%% ============== PATHS ========================================
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir); script_dir = pwd; end
addpath(script_dir);

xlsx_file = fullfile(script_dir, 'charades_traj_summary.xlsx');

%% ============== LOAD TRAJECTORY DATA ========================
[num, txt, ~] = xlsread(xlsx_file,'selected_1133'); %'all'

if strcmp(input, 'stim')
    exlab = 'exp2';
    [numS, txtS, ~] = xlsread(xlsx_file, 'selected_exp2');
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
    out_main    = 'estpart_forcemodel.v2_improved.mat';
    out_prevobs = 'estpart_forcemodel.v2_improved.prevobs.mat';
else
    exlab = 'all_1133';
    vidnum   = length(num);
    semlabel = cell(vidnum, 1);
    cordcell = cell(vidnum, 6);
    for i = 1:vidnum
        semlabel{i} = txt{i+1, 2};
        for j = 1:6
            temp  = txt{i+1, 3+j};
            temp2 = str2double(strsplit(temp(3:end-2), ''', '''));
            cordcell{i,j} = temp2;
        end
    end
    out_main    = 'estpart_forcemodel.v2_improved_all.mat';
    out_prevobs = 'estpart_forcemodel.v2_improved_all.prevobs.mat';
end

savedir = fullfile(script_dir, 'rst', exlab);
if ~exist(savedir, 'dir'), mkdir(savedir); end

%% ============== SKIP-IF-DONE GUARD ===========================
% If both V2 outputs already exist, load them into estpara_v2 /
% estparaObsprev_v2 and skip straight to SAVE. Set force_recompute=true
% (above) to ignore the cache and always re-run the parfor estimation.
out_main_path    = fullfile(savedir, out_main);
out_prevobs_path = fullfile(savedir, out_prevobs);
need_estimate    = true;
if ~force_recompute && exist(out_main_path, 'file') && exist(out_prevobs_path, 'file')
    sM = load(out_main_path,    'estpara_v2');
    sP = load(out_prevobs_path, 'estparaObsprev_v2');
    if isfield(sM, 'estpara_v2') && isfield(sP, 'estparaObsprev_v2')
        estpara_v2        = sM.estpara_v2;
        estparaObsprev_v2 = sP.estparaObsprev_v2;
        need_estimate     = false;
        fprintf(['Outputs found — loaded saved estimate (%d videos) from:\n' ...
                 '  %s\n  %s\nSkipping estimation. Set force_recompute=true to override.\n'], ...
                numel(estpara_v2), out_main_path, out_prevobs_path);
    else
        fprintf('Output files exist but lack expected variables — recomputing.\n');
    end
end

if need_estimate
%% ============== OPTIONAL V1 WARM-START =======================
% If a V1 estpara file exists alongside, pass each video's V1 6-tuple
% per window to forcemodelgenprevA_v2 as an extra fminsearch seed.
% This prevents V2 from regressing on non-convex clips where V1 sits
% in a strictly better basin. If no V1 file, V2 runs without warm-start.
if strcmp(input, 'stim')
    v1_file = fullfile(savedir, 'estpart_forcemodel.v3.mat');
else
    v1_file = fullfile(savedir, 'estpart_forcemodel.v3_all.mat');
end
estpara_v1 = cell(vidnum, 1);
if exist(v1_file, 'file')
    s_v1 = load(v1_file, 'estpara');
    if isfield(s_v1, 'estpara')
        n_v1 = min(numel(s_v1.estpara), vidnum);
        estpara_v1(1:n_v1) = s_v1.estpara(1:n_v1);
        n_have = sum(~cellfun(@isempty, estpara_v1));
        fprintf('V1 warm-start: loaded %d/%d entries from %s\n', n_have, vidnum, v1_file);
    else
        fprintf('V1 warm-start: %s has no ''estpara'' variable; running without warm-start.\n', v1_file);
    end
else
    fprintf('V1 warm-start: %s not found; running without warm-start.\n', v1_file);
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

%% ============== PARALLEL POOL ================================
pool = [];
if use_parfor
    try
        p = gcp('nocreate');
        if ~isempty(p) && nworkers > 0 && p.NumWorkers ~= nworkers
            fprintf('Existing pool has %d workers, want %d — recreating.\n', p.NumWorkers, nworkers);
            delete(p); p = [];
        end
        if isempty(p)
            if nworkers > 0; parpool(nworkers); else; parpool(); end
        end
        parfevalOnAll(gcp, @addpath, 0, script_dir);   % workers need forcemodel*_v2, LJfunc*, plot_one_video
        pool = gcp;
    catch ME
        warning('main_force_model_est_par_v2:poolFailed', ...
            'Parallel pool failed — falling back to sequential. Reason: %s', ME.message);
    end
end
nw = 1; if ~isempty(pool); nw = pool.NumWorkers; end
fprintf('Estimating V2 on %d videos with %d worker(s)...\n', vidnum, nw);

%% ============== ESTIMATION + (optional) PLOT =================
estpara_v2        = cell(vidnum, 1);
estparaObsprev_v2 = cell(vidnum, 1);

% progress: print finished-video count every ~50, LIVE during the parfor.
% DataQueue/afterEach can't be used here: its callback runs on the client,
% but the client is busy inside parfor, so those prints would only flush
% once the loop ends. Instead each worker appends one byte to a shared
% counter file and reports the running byte-count (= videos finished).
counter_file = fullfile(savedir, '.progress_v2.count');
if exist(counter_file, 'file'); delete(counter_file); end

t0 = tic;
parfor i = 1:vidnum
    data_i  = dataraw_all{i};
    ep_v1_i = estpara_v1{i};   % may be empty; gen*_v2 falls back to no warm-start

    % stage 1: use OBS positions for previous time window (V2 fitters)
    [eA]  = forcemodelprevobsA_v2(data_i);
    [eB]  = forcemodelprevobsB_v2(data_i);
    eObs  = [eA, eB];
    estparaObsprev_v2{i} = eObs;

    % stage 2: use GEN positions for previous time window (V2 fitters),
    %          warm-started from V1's per-window params when available
    [eA2] = forcemodelgenprevA_v2(eObs, data_i, ep_v1_i);
    [eB2] = forcemodelgenprevB_v2(eObs, data_i);
    ep_i  = [eA2, eB2];
    estpara_v2{i} = ep_i;

    if plotflagtraj == 1
        plot_one_video(ep_i, data_i, i, semlabel{i}, ...
                       savedir, 2, intv, scaler);    % vermodel=2 -> filename "V2_trialtrial-<i>.png"
    end

    fprintf('  done video %d/%d  (%s)\n', i, vidnum, semlabel{i});

    % live global progress: atomic 1-byte append, then report total count
    fid = fopen(counter_file, 'a');
    if fid > 0
        fwrite(fid, '.'); fclose(fid);
        d = dir(counter_file);
        if mod(d.bytes, 50) == 0
            fprintf('  >>> %d/%d videos finished\n', d.bytes, vidnum);
        end
    end
end
fprintf('All %d videos done in %.1f min.\n', vidnum, toc(t0)/60);
if exist(counter_file, 'file'); delete(counter_file); end

end  % if need_estimate

%% ============== SAVE =========================================
save(fullfile(savedir, out_main),    'estpara_v2');
save(fullfile(savedir, out_prevobs), 'estparaObsprev_v2');
fprintf('Saved estpara_v2 to %s\n', fullfile(savedir, out_main));
fprintf('Saved estparaObsprev_v2 to %s\n', fullfile(savedir, out_prevobs));

estpara = estpara_v2;
save(fullfile(savedir, 'estpart_forcemodel.v2_all.mat'),    'estpara');
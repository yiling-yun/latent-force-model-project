% main_force_model_est_par.m
% Parallel version of main_force_model_est.m. Same I/O, same outputs:
%   - estpart_forcemodel.v<vermodel>.mat        (variable: estpara)
%   - estpart_forcemodel.v<vermodel>.prevobs.mat (variable: estparaObsprev)
%   - per-video plot V<vermodel>_trialtrial-<i>.png  (when plotflagtraj=1)
%
% Differences from main_force_model_est.m:
%   - parfor loop over videos (set use_parfor / nworkers below)
%   - plotting done by plot_one_video.m using an invisible figure per
%     worker, so saveas is safe inside parfor
%   - paths resolved via mfilename('fullpath'), so script works from any pwd

clear all;
close all;

%% ============== USER SETTINGS ===============================
input        = 'all'; % 'all' (1133 videos) or 'stim' (54 selected)
vermodel     = 3;
reverseflag  = 0;      % 1: reverse frame order (Exp 3); 0: Exp 1
plotflagtraj = 0;      % 1: save per-video trajectory PNG; 0: skip
use_parfor   = true;
nworkers     = 32;     % 0 = MATLAB default pool size

rng(1);
scaler = 100;
intv   = 5;

%% ============== PATHS ========================================
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir); script_dir = pwd; end
addpath(script_dir);

xlsx_file = fullfile(script_dir, 'charades_traj_summary.xlsx'); 


%% ============== LOAD TRAJECTORY DATA ========================
[num, txt, ~] = xlsread(xlsx_file, 'selected_1133'); %'all'

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
end

savedir = fullfile(script_dir, 'rst', exlab);
if ~exist(savedir, 'dir'), mkdir(savedir); end

%% ============== BUILD PADDED TRAJECTORY MATRICES =============
% Done once on the client so workers receive ready-to-use data slices.
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
    datapad = repmat([x1(1) y1(1) x2(1) y2(1)], 11, 1);   % match main_force_model_est.m
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
        parfevalOnAll(gcp, @addpath, 0, script_dir);   % workers need forcemodel*, LJfunc*, plot_one_video
        pool = gcp;
    catch ME
        warning('main_force_model_est_par:poolFailed', ...
            'Parallel pool failed — falling back to sequential. Reason: %s', ME.message);
    end
end
nw = 1; if ~isempty(pool); nw = pool.NumWorkers; end
fprintf('Estimating V%d on %d videos with %d worker(s)...\n', vermodel, vidnum, nw);

%% ============== ESTIMATION + (optional) PLOT =================
estpara        = cell(vidnum, 1);
estparaObsprev = cell(vidnum, 1);

% progress: print finished-video count every ~50, LIVE during the parfor.
% DataQueue/afterEach can't be used here: its callback runs on the client,
% but the client is busy inside parfor, so those prints would only flush
% once the loop ends. Instead each worker appends one byte to a shared
% counter file and reports the running byte-count (= videos finished).
counter_file = fullfile(savedir, sprintf('.progress_v%d.count', vermodel));
if exist(counter_file, 'file'); delete(counter_file); end

t0 = tic;
parfor i = 1:vidnum
    data_i = dataraw_all{i};

    % use OBS positions for previous time window
    [eA]  = forcemodelprevobsA(data_i);
    [eB]  = forcemodelprevobsB(data_i);
    eObs  = [eA, eB];
    estparaObsprev{i} = eObs;

    % use GEN positions for previous time window
    [eA2] = forcemodelgenprevA(eObs, data_i);
    [eB2] = forcemodelgenprevB(eObs, data_i);
    ep_i  = [eA2, eB2];
    estpara{i} = ep_i;

    if plotflagtraj == 1
        plot_one_video(ep_i, data_i, i, semlabel{i}, ...
                       savedir, vermodel, intv, scaler);
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

%% ============== SAVE =========================================
if strcmp(input, 'all')
    save(fullfile(savedir, ['estpart_forcemodel.v' num2str(vermodel) '_all.mat']),         'estpara');
    save(fullfile(savedir, ['estpart_forcemodel.v' num2str(vermodel) '_all.prevobs.mat']), 'estparaObsprev');
else
    save(fullfile(savedir, ['estpart_forcemodel.v' num2str(vermodel) '.mat']),         'estpara');
    save(fullfile(savedir, ['estpart_forcemodel.v' num2str(vermodel) '.prevobs.mat']), 'estparaObsprev');
end
fprintf('Saved estpara and estparaObsprev to %s\n', savedir);

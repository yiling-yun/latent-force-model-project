% examine_videos.m
% Drill into V1 vs V2 fits for a list of videos. For each window of each
% video, prints two errors side by side:
%   - sample @ intv+1     : metric reported by compute_v1_v2_errors / table
%                           / per-trajectory subtitle
%   - cascade window MEAN : the shape minimized by the fitter objective
%                           (mean over all 11 frames of the window)
%
% The warm-start guarantees V2 cascade-MEAN <= V1 cascade-MEAN per window.
% It does NOT guarantee V2 sample-at-intv+1 <= V1 sample-at-intv+1, because
% those are different samples of the same cascade. This script makes the
% distinction visible per window, so you can tell when a "V2 worse"
% headline is a real fit regression (mean worse too) versus a metric
% disagreement (mean better, single sample worse).

clear; close all;

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir); script_dir = pwd; end
addpath(script_dir);

target_video_indices = [11, 44, 54];   % talk, throw, encircle
intv   = 5;
scaler = 100;

%% load inputs
xlsx_file = fullfile(script_dir, 'charades_traj_summary.xlsx');
[num,  txt,  ~] = xlsread(xlsx_file, 'all');
[numS, txtS, ~] = xlsread(xlsx_file, 'selected_exp2');

v1_file = fullfile(script_dir, 'rst/exp2/estpart_forcemodel.v3.mat');
v2_file = fullfile(script_dir, 'rst/exp2/estpart_forcemodel.v2_improved.mat');
s1 = load(v1_file, 'estpara');
s2 = load(v2_file, 'estpara_v2');

for k = 1:length(target_video_indices)
    target = target_video_indices(k);
    label  = txtS{target};
    fprintf('\n%s\n  Video %d  (%s)\n%s\n', repmat('=',1,80), target, label, repmat('=',1,80));

    %% build padded dataraw for this video
    rowi = find(numS(target) == num(:,2));
    cc = cell(1,6);
    for j = 1:6
        temp  = txt{rowi+1, 3+j};
        cc{j} = str2double(strsplit(temp(3:end-2), ''', '''));
    end
    x1c = cc{1}; y1c = cc{2}; x2c = cc{4}; y2c = cc{5};
    dataraw = [x1c' y1c' x2c' y2c'];
    datapad = repmat(dataraw(1,:), 2*intv+1, 1);
    dataraw = [datapad; dataraw];

    ep1 = s1.estpara{target};
    ep2 = s2.estpara_v2{target};

    framenum = size(dataraw, 1);
    data     = dataraw / scaler;
    nwin     = size(ep1, 1);
    sampleIdx = intv + 1;

    err_sample_v1 = zeros(nwin,1); err_sample_v2 = zeros(nwin,1);
    err_mean_v1   = zeros(nwin,1); err_mean_v2   = zeros(nwin,1);

    aposprev_v1 = []; aposprev_v2 = [];

    countfi = 0;
    for fi = 1+intv : intv : framenum-intv
        if countfi >= nwin, break; end
        countfi = countfi + 1;

        frameintv = max(1,fi-intv) : min(fi+intv, framenum);
        aposobs = data(frameintv, 1:2);
        bposobs = data(frameintv, 3:4);

        if countfi == 1
            aposprev_v1 = aposobs*0;
            aposprev_v1(intv+1,:) = aposobs(1,:);
            aposprev_v1(intv+2,:) = aposobs(2,:);
            aposprev_v2 = aposprev_v1;
        end

        [apos1, ~, ~, ~] = LJfuncCompforce(ep1(countfi,1:6), aposprev_v1, aposobs, bposobs, 1);
        aposprev_v1 = apos1;
        [apos2, ~, ~, ~] = LJfuncCompforce(ep2(countfi,1:6), aposprev_v2, aposobs, bposobs, 1);
        aposprev_v2 = apos2;

        err_sample_v1(countfi) = sqrt(sum((apos1(sampleIdx,:) - aposobs(sampleIdx,:)).^2, 2)) * scaler;
        err_sample_v2(countfi) = sqrt(sum((apos2(sampleIdx,:) - aposobs(sampleIdx,:)).^2, 2)) * scaler;
        err_mean_v1(countfi)   = mean(sqrt(sum((apos1 - aposobs).^2, 2))) * scaler;
        err_mean_v2(countfi)   = mean(sqrt(sum((apos2 - aposobs).^2, 2))) * scaler;
    end

    %% per-window table
    fprintf('\n%-3s | %10s %10s | %10s %10s | %s\n', ...
        'win', 'sample V1','sample V2', 'meanwin V1','meanwin V2', 'note');
    fprintf('%s\n', repmat('-',1,80));
    for fi = 1:nwin
        note = '';
        if err_sample_v2(fi) > err_sample_v1(fi) + 1
            note = [note 'sampleV2>V1 '];
        end
        if err_mean_v2(fi) > err_mean_v1(fi) + 0.5
            note = [note 'MEAN-V2>V1(unexpected!)'];   %#ok<AGROW>
        end
        fprintf('%-3d | %10.2f %10.2f | %10.2f %10.2f | %s\n', ...
            fi, err_sample_v1(fi), err_sample_v2(fi), err_mean_v1(fi), err_mean_v2(fi), note);
    end

    %% per-video summary
    n_mean_worse   = sum(err_mean_v2   > err_mean_v1   + 0.5);
    n_sample_worse = sum(err_sample_v2 > err_sample_v1 + 1);

    fprintf('\n--- video %d (%s) summary ---\n', target, label);
    fprintf('  sample @ intv+1 mean : V1 = %7.2f,  V2 = %7.2f   (compute_v1_v2_errors metric)\n', ...
        mean(err_sample_v1), mean(err_sample_v2));
    fprintf('  cascade-MEAN    mean : V1 = %7.2f,  V2 = %7.2f   (fitter objective shape)\n', ...
        mean(err_mean_v1), mean(err_mean_v2));
    fprintf('  windows where V2 sample > V1 sample : %d / %d\n', n_sample_worse, nwin);
    fprintf('  windows where V2 mean   > V1 mean   : %d / %d   <-- warm-start guarantee violations\n', ...
        n_mean_worse, nwin);

    if n_mean_worse == 0
        fprintf('  ==> warm-start guarantee held on every window. The "V2 worse" in the table\n');
        fprintf('      is a sample-vs-mean disagreement, not a fitting regression.\n');
    else
        fprintf('  ==> warm-start FAILED to bring V2 mean down to V1''s on %d windows.\n', n_mean_worse);
        fprintf('      Likely cause: V1 params were out of V2''s bounds (bcoef>2), so fminsearch\n');
        fprintf('      from the V1 seed bounced into a different (worse) basin.\n');
    end

    %% per-window plot
    figure('Name', sprintf('vid %d %s', target, label), 'Position', [200 200 1000 400]);
    subplot(1,2,1);
    plot(1:nwin, err_sample_v1, '-o', 'DisplayName','V1'); hold on;
    plot(1:nwin, err_sample_v2, '-+', 'DisplayName','V2');
    xlabel('window'); ylabel('sample @ intv+1 err (px)');
    title(sprintf('%s — sample-at-intv+1', label)); legend('Location','best'); grid on;

    subplot(1,2,2);
    plot(1:nwin, err_mean_v1, '-o', 'DisplayName','V1'); hold on;
    plot(1:nwin, err_mean_v2, '-+', 'DisplayName','V2');
    xlabel('window'); ylabel('cascade-MEAN err (px)');
    title(sprintf('%s — window-MEAN (fitter objective)', label)); legend('Location','best'); grid on;
end

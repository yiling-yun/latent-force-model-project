% plot_selected_animations.m
% Run simulation and plot observed vs predicted trajectories for selected
% animations. Loads pre-computed estpara from the V1 .mat file.
%
% Set selected_idx to the animation indices you want to inspect.
% Figures are saved as JPG to rst/exp2/.

clear; close all;

%% =========================================================
%  USER SETTINGS
%% =========================================================
selected_idx = [1, 2, 3,6,9,10,11,12];   % <-- change to indices of interest (1-based into selected_exp2)

v1_file  = './rst/exp2/estpart_forcemodel.v3.mat';
savedir  = './rst/exp2';
scaler   = 100;
intv     = 5;

%% =========================================================
%  LOAD TRAJECTORY DATA
%% =========================================================
[num, txt, ~] = xlsread('charades_traj_summary.xlsx', 'all');
[numS, txtS, ~] = xlsread('charades_traj_summary.xlsx', 'selected_exp2');
total_vids = length(numS);

cordcell = cell(total_vids, 6);
for i = 1:total_vids
    rowi = find(numS(i) == num(:,2));
    for j = 1:6
        temp  = txt{rowi+1, 3+j};
        temp2 = str2double(strsplit(temp(3:end-2), ''', '''));
        cordcell{i,j} = temp2;
    end
end
semlabel = txtS;
fprintf('Loaded %d trajectories.\n', total_vids);

%% =========================================================
%  LOAD V1 ESTPARA
%% =========================================================
if ~exist(v1_file, 'file')
    error('V1 results not found: %s\nRun main_force_model_est.m first.', v1_file);
end
load(v1_file, 'estpara');
fprintf('Loaded estpara from %s\n', v1_file);

if ~exist(savedir, 'dir'), mkdir(savedir); end

%% =========================================================
%  SIMULATE AND PLOT SELECTED ANIMATIONS
%% =========================================================
for ii = 1:length(selected_idx)
    i = selected_idx(ii);

    if i < 1 || i > total_vids
        fprintf('Index %d out of range (1-%d), skipping.\n', i, total_vids);
        continue;
    end

    label = semlabel{i};
    fprintf('[%d/%d] Animation %d: %s\n', ii, length(selected_idx), i, label);

    % Build dataraw with 11-frame padding
    x1 = cordcell{i,1}; y1 = cordcell{i,2};
    x2 = cordcell{i,4}; y2 = cordcell{i,5};
    dataraw  = [x1' y1' x2' y2'];
    datapad  = repmat(dataraw(1,:), 2*intv+1, 1);
    dataraw  = [datapad; dataraw];
    framenum = size(dataraw, 1);

    estparavid = estpara{i};
    nwin       = size(estparavid, 1);
    framesall  = 1+intv : intv : framenum-intv;

    % Cascade through windows
    aposobs_all = cell(nwin, 1);
    bposobs_all = cell(nwin, 1);
    aposgen_all = cell(nwin, 1);
    bposgen_all = cell(nwin, 1);
    aposprev = []; bposprev = [];

    for fi = 1:nwin
        turnframe  = framesall(fi);
        framerange = max(1, turnframe-intv) : min(turnframe+intv, framenum);

        aposobs_all{fi} = dataraw(framerange, 1:2) / scaler;
        bposobs_all{fi} = dataraw(framerange, 3:4) / scaler;

        if fi == 1
            aposprev = aposobs_all{fi} * 0;
            aposprev(intv+1, :) = aposobs_all{fi}(1, :);
            aposprev(intv+2, :) = aposobs_all{fi}(2, :);
            bposprev = bposobs_all{fi} * 0;
            bposprev(intv+1, :) = bposobs_all{fi}(1, :);
            bposprev(intv+2, :) = bposobs_all{fi}(2, :);
        end

        [aposgen_all{fi}, bposgen_all{fi}, ~] = LJfuncCompforce( ...
            estparavid(fi, :), aposprev, aposobs_all{fi}, bposobs_all{fi}, 1);
        aposprev = aposgen_all{fi};

        [bposgen_all{fi}, ~] = LJfuncself( ...
            estparavid(fi, 8:10), bposprev, bposobs_all{fi}, 1);
        bposprev = bposobs_all{fi};
    end

    % Collect window-centre points
    dataall = zeros(nwin, 8);
    for fi = 1:nwin
        dataall(fi, :) = [ ...
            aposobs_all{fi}(intv+1, 1), aposobs_all{fi}(intv+1, 2), ...
            bposobs_all{fi}(intv+1, 1), bposobs_all{fi}(intv+1, 2), ...
            aposgen_all{fi}(intv+1, 1), aposgen_all{fi}(intv+1, 2), ...
            bposgen_all{fi}(intv+1, 1), bposgen_all{fi}(intv+1, 2)] * scaler;
    end

    % --- Plot ---
    hfig = figure('Visible', 'off', 'Position', [100 100 700 600]);

    % Trajectories
    plot(dataall(:,1), dataall(:,2), '-o', 'Color', [0.85 0.15 0.15], ...
        'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Obs A'); hold on;
    plot(dataall(:,3), dataall(:,4), '-o', 'Color', [0.15 0.35 0.85], ...
        'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Obs B');
    plot(dataall(:,5), dataall(:,6), '--+', 'Color', [0.1 0.7 0.2], ...
        'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Pred A');
    plot(dataall(:,7), dataall(:,8), '--x', 'Color', [0.2 0.2 0.2], ...
        'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Pred B');

    % Start markers (filled circles)
    plot(dataall(1,1), dataall(1,2), 'o', 'MarkerFaceColor', [0.85 0.15 0.15], ...
        'MarkerEdgeColor', [0.85 0.15 0.15], 'MarkerSize', 10, 'HandleVisibility', 'off');
    plot(dataall(1,3), dataall(1,4), 'o', 'MarkerFaceColor', [0.15 0.35 0.85], ...
        'MarkerEdgeColor', [0.15 0.35 0.85], 'MarkerSize', 10, 'HandleVisibility', 'off');

    xlim([0 4000]); ylim([0 3800]);
    xlabel('X (px)'); ylabel('Y (px)');
    title(sprintf('[%d] %s', i, label), 'Interpreter', 'none');
    legend('Location', 'best');
    grid on; box on;

    % Compute and display position errors in title
    errA = mean(sqrt((dataall(:,1)-dataall(:,5)).^2 + (dataall(:,2)-dataall(:,6)).^2));
    errB = mean(sqrt((dataall(:,3)-dataall(:,7)).^2 + (dataall(:,4)-dataall(:,8)).^2));
    subtitle(sprintf('Mean pos error — A: %.1f px   B: %.1f px', errA, errB));

    % Save
    fname = fullfile(savedir, sprintf('traj_%03d_%s.jpg', i, label));
    exportgraphics(hfig, fname, 'Resolution', 150);
    close(hfig);
    fprintf('   Saved: %s\n', fname);
end

fprintf('\nDone. Figures saved to %s\n', savedir);

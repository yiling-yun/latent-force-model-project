% examine_hit.m
% Drill into the V1 vs V2 fit for the "hit" video (index 10 in selected_exp2).
% Shows per-window:
%   - sampled-frame position error V1 vs V2 (matches compare_est_v1_v2.m table)
%   - all six A-force params V1 vs V2, side-by-side
% Then highlights the windows where V2 is worst vs V1, and prints the
% per-parameter mean & max-abs difference across windows.

clear; close all;

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir); script_dir = pwd; end

target_video_index = 10;          % "hit" in selected_exp2 (also: 37 = second 'hit')
intv   = 5;
scaler = 100;

%% load
xlsx_file = fullfile(script_dir, 'charades_traj_summary.xlsx');
[num, txt, ~] = xlsread(xlsx_file, 'all');
[numS, txtS, ~] = xlsread(xlsx_file, 'selected_exp2');

label = txtS{target_video_index};
rowi  = find(numS(target_video_index) == num(:,2));
cordcell = cell(1,6);
for j = 1:6
    temp  = txt{rowi+1, 3+j};
    cordcell{j} = str2double(strsplit(temp(3:end-2), ''', '''));
end
x1 = cordcell{1}; y1 = cordcell{2};
x2 = cordcell{4}; y2 = cordcell{5};
dataraw = [x1' y1' x2' y2'];
datapad = repmat(dataraw(1,:), 2*intv+1, 1);
dataraw = [datapad; dataraw];

v1_file = fullfile(script_dir, 'rst/exp2/estpart_forcemodel.v3.mat');
v2_file = fullfile(script_dir, 'rst/exp2/estpart_forcemodel.v2_improved.mat');
s1 = load(v1_file, 'estpara');     ep1 = s1.estpara{target_video_index};
s2 = load(v2_file, 'estpara_v2');  ep2 = s2.estpara_v2{target_video_index};

fprintf('Video %d (%s):  %d windows,  obs frames %d\n', ...
    target_video_index, label, size(ep1,1), size(dataraw,1));

%% per-window cascade — re-using compute_v1_v2_errors helper
[errs, dataall_v1, dataall_v2] = compute_v1_v2_errors(ep1, ep2, dataraw, intv, scaler);

% Per-window |pred - obs| at the sample point (sample frame intv+1).
% dataall_*(:,1:2) = obsA;  (:,5:6) = predA  (in pixel units already).
pe_A_v1 = sqrt(sum((dataall_v1(:,5:6) - dataall_v1(:,1:2)).^2, 2));
pe_A_v2 = sqrt(sum((dataall_v2(:,5:6) - dataall_v2(:,1:2)).^2, 2));

%% per-window param table (Agent A: cols 1-6 = eps1, sig1, bcoef1, eps2, sig2, bcoef2)
nwin = size(ep1,1);
fprintf('\n%-3s %-9s | %7s %7s %7s %7s %7s %7s | %7s %7s %7s %7s %7s %7s | %s\n', ...
    'win', 'errV1/V2', 'eps1_1','sig1_1','b1_1','eps2_1','sig2_1','b2_1', ...
                       'eps1_2','sig1_2','b1_2','eps2_2','sig2_2','b2_2', 'flag');
fprintf('%s\n', repmat('-',1,170));
for fi = 1:nwin
    p1 = ep1(fi,1:6); p2 = ep2(fi,1:6);
    flag = '';
    if pe_A_v2(fi) > pe_A_v1(fi) * 2 && pe_A_v2(fi) > 20
        flag = '*** V2 much worse';
    end
    fprintf('%-3d %4.1f/%4.1f  | %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f | %s\n', ...
        fi, pe_A_v1(fi), pe_A_v2(fi), p1, p2, flag);
end

%% summary
fprintf('\n--- summary ---\n');
fprintf('mean PosErr A:  V1 = %.2f,  V2 = %.2f\n', mean(pe_A_v1), mean(pe_A_v2));
fprintf('max  PosErr A:  V1 = %.2f (win %d),  V2 = %.2f (win %d)\n', ...
    max(pe_A_v1), argmax(pe_A_v1), max(pe_A_v2), argmax(pe_A_v2));
fprintf('windows where V2 is more than 2x V1 (and >20 px): %d / %d\n', ...
    sum(pe_A_v2 > 2*pe_A_v1 & pe_A_v2 > 20), nwin);

dparam = ep2(:,1:6) - ep1(:,1:6);
names = {'eps1','sig1','bcoef1','eps2','sig2','bcoef2'};
fprintf('\nParam-by-param V2 - V1 over the %d windows:\n', nwin);
fprintf('  %-7s %10s %10s %10s\n', 'name', 'mean', 'max-abs', 'V2 bound');
fprintf('  %s\n', repmat('-',1,42));
bounds = [40 40 2 40 40 2];
for k = 1:6
    fprintf('  %-7s %10.3f %10.3f %10.0f\n', names{k}, mean(dparam(:,k)), max(abs(dparam(:,k))), bounds(k));
end

% How often does V1 sit above the V2 cap of 2 on bcoef? (3rd and 6th cols)
fprintf('\nbcoef behavior:\n');
fprintf('  V1 bcoef1 > 2 in %d / %d windows (max %.2f)\n', sum(ep1(:,3) > 2), nwin, max(ep1(:,3)));
fprintf('  V1 bcoef2 > 2 in %d / %d windows (max %.2f)\n', sum(ep1(:,6) > 2), nwin, max(ep1(:,6)));
fprintf('  V2 bcoef1 max %.4f (cap = 2)\n', max(ep2(:,3)));
fprintf('  V2 bcoef2 max %.4f (cap = 2)\n', max(ep2(:,6)));

%% per-window scatter
figure('Name', sprintf('hit per-window errors (vid %d)', target_video_index), ...
       'Position', [200 200 900 400]);
subplot(1,2,1);
plot(1:nwin, pe_A_v1, '-o', 'LineWidth',1.2, 'DisplayName','V1'); hold on;
plot(1:nwin, pe_A_v2, '-+', 'LineWidth',1.2, 'DisplayName','V2');
xlabel('window'); ylabel('|pred - obs| at sample frame (px)');
title(sprintf('%s — per-window PosErr A', label));
legend('Location','best'); grid on;

subplot(1,2,2);
scatter(pe_A_v1, pe_A_v2, 40, 'filled'); hold on;
mx = max([pe_A_v1; pe_A_v2]) * 1.05;
plot([0 mx],[0 mx],'k--');
xlabel('V1 PosErr A'); ylabel('V2 PosErr A');
title('per-window V1 vs V2  (above diag = V2 worse)');
axis equal tight; grid on;

% helper used in fprintf above
function k = argmax(v)
    [~, k] = max(v);
end

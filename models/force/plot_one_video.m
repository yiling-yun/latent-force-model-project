function plot_one_video(estparavid, dataraw, vidIdx, label, ...
                        savedir, vermodel, intv, scaler)
% plot_one_video  Per-video trajectory plot used by main_force_model_est_par.m
%
% Replicates the plot block in main_force_model_est.m: cascade-rollout
% prediction with LJfuncCompforce (Agent A) + LJfuncself (Agent B),
% plot observed + predicted points at each window centre, save PNG.
%
% Uses an invisible figure so it is safe inside parfor.
%
% Filename matches main_force_model_est.m exactly:
%   <savedir>/V<vermodel>_trialtrial-<i>.png

framenum  = size(dataraw, 1);
framesall = 1+intv : intv : framenum-intv;

nwin    = size(estparavid, 1);
aposobs = cell(nwin, 1);
bposobs = cell(nwin, 1);
aposgen = cell(nwin, 1);
bposgen = cell(nwin, 1);

aposprev = []; bposprev = [];

for fi = 1:nwin
    if fi > length(framesall), break; end   % defensive: don't exceed available windows
    turnframe  = framesall(fi);
    framerange = max(1,turnframe-intv) : min(turnframe+intv, framenum);

    aposobs{fi} = dataraw(framerange,1:2) / scaler;
    bposobs{fi} = dataraw(framerange,3:4) / scaler;

    if fi == 1
        aposprev = aposobs{fi}*0;
        aposprev(intv+1,:) = aposobs{fi}(1,:);
        aposprev(intv+2,:) = aposobs{fi}(2,:);
        bposprev = bposobs{fi}*0;
        bposprev(intv+1,:) = bposobs{fi}(1,:);
        bposprev(intv+2,:) = bposobs{fi}(2,:);
    end

    [aposgen{fi}, bposgen{fi}, ~] = LJfuncCompforce(estparavid(fi,:), aposprev, aposobs{fi}, bposobs{fi}, 1);
    aposprev = aposgen{fi};
    [bposgen{fi}, ~]              = LJfuncself   (estparavid(fi,8:10), bposprev, bposobs{fi}, 1);
    bposprev = bposobs{fi};
end

dataall = zeros(nwin, 8);
for j = 1:nwin
    if isempty(aposgen{j}); continue; end
    dataall(j,:) = [aposobs{j}(intv+1,1) aposobs{j}(intv+1,2) ...
                    bposobs{j}(intv+1,1) bposobs{j}(intv+1,2) ...
                    aposgen{j}(intv+1,1) aposgen{j}(intv+1,2) ...
                    bposgen{j}(intv+1,1) bposgen{j}(intv+1,2)] * scaler;
end

hfig = figure('Visible','off');
plot(dataall(:,1), dataall(:,2), '-or'); hold on;   % obs a
plot(dataall(:,3), dataall(:,4), '-ob');             % obs b
plot(dataall(:,5), dataall(:,6), '-+g');             % pred a
plot(dataall(:,7), dataall(:,8), '-+k');             % pred b
plot(dataall(1,1), dataall(1,2), 'o', 'MarkerFaceColor','r', 'MarkerEdgeColor','r');
plot(dataall(1,3), dataall(1,4), 'o', 'MarkerFaceColor','b', 'MarkerEdgeColor','b');
xlim([50 4000]); ylim([50 3800]);
trialnum = ['trial-' num2str(vidIdx)];
title(sprintf('%s (%s)', trialnum, label), 'Interpreter','none');

% Match main_force_model_est.m's filename convention exactly (the word
% "trial" appears twice: <savedir>/V<vermodel>_trial<trialnum>.png).
fname = fullfile(savedir, ['V' num2str(vermodel) '_trial' trialnum '.png']);
saveas(hfig, fname);
close(hfig);
end

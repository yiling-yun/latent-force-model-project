function [estpara] = forcemodelgenprevA_v2(estpara, dataraw, ep_v1_init)
% Pass 2 estimation for Agent A (v2).
% Identical to forcemodelgenprevA (v1) except uses LJfuncParaCompforceEst_v2
% as the objective function.
%
% Optional 3rd argument ep_v1_init is a per-video V1 estpara matrix
% (nwin x >=6). When supplied, V1's per-window 6-tuple is added as an
% extra warm-start seed in V2's grid search, so V2 cannot regress on
% windows where V1 sat in a strictly better basin: fminsearch from V1
% is monotonically downhill in V1's basin, so the resulting V2 fit's
% objective is at worst equal to V1's.
%
% If ep_v1_init is omitted or empty, behavior is identical to the
% previous version (V2 picks its own grid seeds only).

if nargin < 3
    ep_v1_init = [];
end

estparainit = estpara;
clear estpara;

scaler = 100;
data = dataraw/scaler;

framenum = size(data,1);
intv = 5;
if framenum < 2*intv+1
    intv = 2;
end
distobjall = sqrt((data(:,3)-data(:,1)).^2+(data(:,4)-data(:,2)).^2);

xx = data(:,1);
yy = data(:,2);
vecx = diff(xx);
vecy = diff(yy);
sp = sqrt(vecx.^2 +vecy.^2);

accx = diff(vecx);
accy = diff(vecy);
absacc = sqrt(accx.^2 + accy.^2);

countfi = 0;
initflag = 1;

% Animation-level budget (Agent A mean speed, computed once)
if mean(sp) < 0.10; budget_a = 1000; else; budget_a = 3000; end

for fi = 1+intv:intv:framenum-intv
    countfi = countfi+1;
    turnframe = fi;
    turndist = distobjall(turnframe);
    frameintv = [max(1,turnframe-intv):min(turnframe+intv, size(data,1))];

    aposobs = data(frameintv,1:2);
    bposobs = data(frameintv,3:4);
    distobs = sqrt((bposobs(:,1)-aposobs(:,1)).^2+(bposobs(:,2)-aposobs(:,2)).^2);
    distobsself = sqrt((aposobs(2:end,1)-aposobs(1:(end-1),1)).^2+(aposobs(2:end,2)-aposobs(1:(end-1),2)).^2);

    if countfi == 1
       aposprev = aposobs*0;
       aposprev(1+intv,:) = aposobs(1,:);
       aposprev(2+intv,:) = aposobs(2,:);
    end;

    x = data(frameintv,:);
    x = [x aposprev];

    aorig = aposprev(round(size(aposobs,1)/2),:);
    distpred = distobs;
    distselfpred = distobsself;
    distobsselfprev = sqrt((aposobs(1:end,1)-aorig(1)).^2+(aposobs(1:end,2)-aorig(2)).^2);

    if mean(distobsself)<eps
        eall1 = 0; sall1 = 0; ball1 = 0;
        eall2 = 0; sall2 = 0; ball2 = 0;
    else
        eall1 = [estparainit(countfi,1) 0];
        turndistA= max([max(distselfpred) max(distobsself)]);
        sall1 = [estparainit(countfi,2) turndistA];
        ball1 = [estparainit(countfi,3) linspace(max([0,estparainit(countfi,3)-0.2]),estparainit(countfi,3)+0.2,3)];

        eall2 = [estparainit(countfi,4) 0];
        turndist = max([max(distpred) max(distobs)]);
        sall2 = [estparainit(countfi,5) turndist];
        ball2 = [estparainit(countfi,6) linspace(max([0,estparainit(countfi,6)-0.2]),estparainit(countfi,6)+0.2,3)];
    end;

    paravec = [];
    count = 0;
    for bi1 = 1:length(ball1)
        for ei1 = 1:length(eall1)
            for si1 = 1:length(sall1)
                for bi2 = 1:length(ball2)
                    for ei2 = 1:length(eall2)
                        for si2 = 1:length(sall2)
                            count = count+1;
                            paravec(count,:) = [eall1(ei1) sall1(si1) ball1(bi1) eall2(ei2) sall2(si2) ball2(bi2)];
                        end;
                    end;
                end;
            end;
        end;
    end;

    % Warm-start: append V1's per-window 6-tuple as an extra grid seed.
    % fminsearch from this seed is monotonically downhill, so the
    % best-of-all selection below guarantees V2 won't pick a basin
    % strictly worse than V1's for this window — PROVIDED the seed is
    % already inside V2's feasible region. V1 had no upper bound on
    % bcoef (params 3 & 6), so V1 sometimes returns bcoef > 2 (V2's
    % cap). If we passed such a seed verbatim, V2's penalty
    % 10^6*(bcoef>2) would fire at iteration 0 and the simplex would
    % flee V1's basin entirely. So clamp V1's seed into V2's bounds
    % first; fminsearch can then refine starting from the closest
    % feasible point.
    if ~isempty(ep_v1_init) && size(ep_v1_init,1) >= countfi
        v1_seed = ep_v1_init(countfi, 1:6);
        v1_seed(1) = min(max(v1_seed(1), 0), 40);   % eps1   in [0, 40]
        v1_seed(2) = min(max(v1_seed(2), 0), 40);   % sig1   in [0, 40]
        v1_seed(3) = min(max(v1_seed(3), 0),  2);   % bcoef1 in [0,  2]
        v1_seed(4) = min(max(v1_seed(4), 0), 40);   % eps2   in [0, 40]
        v1_seed(5) = min(max(v1_seed(5), 0), 40);   % sig2   in [0, 40]
        v1_seed(6) = min(max(v1_seed(6), 0),  2);   % bcoef2 in [0,  2]
        count = count + 1;
        paravec(count,:) = v1_seed;
    end

    initialParams = estparainit(countfi,1:6);

    devvalinit = LJfuncParaCompforceEst_v2(initialParams,x);     % <-- v2 objective
    objectiveFunc = @(parameters) LJfuncParaCompforceEst_v2(parameters,x);  % <-- v2 objective

    options = optimset('MaxFunEvals', budget_a,'MaxIter',budget_a,'Display','off');
    devval = 10^6;
    for ci = 1:count
        initialParams = paravec(ci,:);
        fittedParamsij = fminsearch(objectiveFunc, initialParams, options);
        devvalij = LJfuncParaCompforceEst_v2(fittedParamsij, x);
        if devvalij < devval
            fittedParams = fittedParamsij;
            devval = devvalij;
        end
    end

    if devval<devvalinit
        estpara(countfi,:) = [fittedParams devval];
    else
        % FIX: fall back to V2 Pass-1 (estparainit), not the loop's last
        % `initialParams` (which is just whatever seed was tried last —
        % a V2 grid point, or V1's warm-start seed when ep_v1_init is
        % supplied). Pairing the latter with devvalinit corrupts the
        % record and, with V1 warm-start, causes the cascade to blow up
        % on any window where no fminsearch chain beat devvalinit.
        estpara(countfi,:) = [estparainit(countfi,1:6) devvalinit];
    end;

    if mean(distobsself)<eps && mean(distobsselfprev)<eps
        estpara(countfi,1:6) = [0 0 0 0 0 0];
    end;

    [aposgen,bposgen,flist,fslist] = LJfuncCompforce(estpara(countfi,:),aposprev,aposobs,bposobs,1);
    aposprev = aposgen;
end;

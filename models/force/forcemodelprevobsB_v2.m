function [estpara] = forcemodelprevobsB_v2(dataraw)
% Pass 1 estimation for Agent B self-propelled force (v2).
% Identical to forcemodelprevobsB (v1) except uses LJfuncParaEstself3_v2
% as the objective function.

scaler = 100;
data = dataraw/scaler;

framenum = size(data,1);
intv = 5;

countfi = 0;

% Animation-level speed threshold (Agent B, computed once)
distobsself_all = sqrt(diff(data(:,3)).^2 + diff(data(:,4)).^2);
if mean(distobsself_all) < 0.10
    emax_a = 20; npts_a = 10; budget_a = 1000;
else
    emax_a = 40; npts_a = 15; budget_a = 3000;
end

for fi = 1+intv:intv:framenum-intv
    countfi = countfi+1;
    turnframe = fi;
    frameintv = [max(1,turnframe-intv):min(turnframe+intv, size(data,1))];

    aposobs = data(frameintv,3:4);
    distobsself = sqrt((aposobs(2:end,1)-aposobs(1:(end-1),1)).^2+(aposobs(2:end,2)-aposobs(1:(end-1),2)).^2);

    if countfi == 1
       aposprev = aposobs*0;
       aposprev(1+intv,:) = aposobs(1,:);
       aposprev(2+intv,:) = aposobs(2,:);
    end

    distself = sqrt((aposobs(2:end,1)-aposobs(1:(end-1),1)).^2+(aposobs(2:end,2)-aposobs(1:(end-1),2)).^2);

    eall1 = linspace(0.1,emax_a,npts_a);
    turndistA = max(distself);
    sall1 = linspace(0.02,(turndistA),20);
    ball1 = linspace(0,1,10);

    indx1 = []; dev1 = [];
    aorig = aposprev(round(size(aposobs,1)/2),:);
    count = 0;
    if mean(distobsself)<0.01
        indx1 = 1;
        count = count+1;
        paravec1(indx1,1:3) = [0,0,0];
        [apos,~,f_list,fs_list] = LJfunc(paravec1(1,:),aposprev,aposobs,aorig,1,"self");
        xdev2 = (apos(:,1) - aposobs(:,1)).^2;
        ydev2 = (apos(:,2) - aposobs(:,2)).^2;
        paravec1(1,4)  = sum(sqrt(xdev2+ydev2));
    else
        for bi = 1:length(ball1)
            for ei = 1:length(eall1)
                for si = 1:length(sall1)
                    epsilon = eall1(ei);
                    sigma = sall1(si);
                    bcoef = ball1(bi);
                    count = count+1;
                    paravec1(count,1:3) = [epsilon,sigma,bcoef];
                    [apos,~,f_list,fs_list] = LJfunc(paravec1(count,:),aposprev,aposobs,aorig,1,"self");
                    xdev2 = (apos(:,1) - aposobs(:,1)).^2;
                    ydev2 = (apos(:,2) - aposobs(:,2)).^2;
                    dev1(count) = sum(sqrt(xdev2+ydev2));
                    paravec1(count,4) = dev1(count);
                end;
            end;
        end;
        mindevval = min(dev1,[],'all');
        [indx1] = find(dev1==mindevval);
        indx1 = indx1(1);   % refine ONE tied-best seed; all ties share the same
                            % Pass-1 objective. Without this, a degenerate flat
                            % landscape (e.g. a near-static swapped agent) yields
                            % thousands of ties and the loop below runs for hours.
    end;

    initialParams = [paravec1(indx1(1),1:3)];

    x = [];
    x = aposobs;
    x = [x aposprev];

    devvalinit = LJfuncParaEstself3_v2(initialParams,x);
    objectiveFunc = @(parameters) LJfuncParaEstself3_v2(parameters,x);

    options = optimset('MaxFunEvals', budget_a,'MaxIter',budget_a,'Display','off');
    devval = 10^6;
    for indj = 1:length(indx1)
        initialParams = [paravec1(indx1(indj),1:3)];
        fittedParamsij = fminsearch(objectiveFunc, initialParams, options);
        devvalij = LJfuncParaEstself3_v2(fittedParamsij, x);
        if devvalij < devval
            fittedParams = fittedParamsij;
            devval = devvalij;
        end
    end

    if devval<devvalinit
        estpara(countfi,:) = [fittedParams devval];
    else
        estpara(countfi,:) = [initialParams devvalinit];
    end;

    [aposgen,~] = LJfuncself(estpara(countfi,:),aposprev,aposobs,1);
    aposprev = aposobs;
end;

estpara = [estpara];

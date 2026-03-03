function [estpara] = forcemodelgenprevB(estpara,dataraw)

% A is non-reference agent, estimate self-propelled force and interactive force parameters
estparainit = estpara;
clear estpara; 

mode = "selfB";

% B is reference, estimate self-propelled force parameter
scaler = 100; 
data = dataraw/scaler;   % scale down the position values for compuation precision 1~40

framenum = size(data,1); 
intv = 5;%  % temporal window +/- intv.
if framenum < 2*intv+1  % for very short videos
     intv = 2;
end;
% distobjallself = sqrt((data(1,3)-data(:,3)).^2+(data(1,4)-data(:,4)).^2);  % distance between objects

xx = data(:,3);  % change to B
yy = data(:,4); 

countfi = 0; 

for fi = 1+intv:intv:framenum-intv
    countfi = countfi+1; 
    %disp(countfi);
    turnframe = fi;
    frameintv = [max(1,turnframe-intv):min(turnframe+intv, size(data,1))];
    
    aposobs = data(frameintv,3:4);  % Get agent B position
    distobsself = sqrt((aposobs(2:end,1)-aposobs(1:(end-1),1)).^2+(aposobs(2:end,2)-aposobs(1:(end-1),2)).^2);

    if countfi == 1
       aposprev = aposobs*0;
       aposprev(1+intv,:) = aposobs(1,:);  % important for self-prop force estimate
       aposprev(2+intv,:) = aposobs(2,:);
    end;


    % extropolate b trajectory more to the future
    %% step 1: grid search

    if mean(distobsself)<0.01 
        eall1 = 0;
        sall1 = 0;
        ball1 = 0;    
    else
  
        eall1 = [estparainit(countfi,8) 0];%% strength in L-J potential
        turndistA= max(distobsself);
        sall1 = [estparainit(countfi,9) turndistA];%% repul dist in L-J potential
        ball1 = [estparainit(countfi,10) linspace(max([0,estparainit(countfi,10)-0.2]),estparainit(countfi,10)+0.2,3)]; % attractive coefficient in L-J potential

    end;

    paravec1 = [];
    count = 0;
    for bi1 = 1:length(ball1)
        for ei1 = 1:length(eall1)
            for si1 = 1:length(sall1)
                count = count +1;
                paravec1(count,:) = [eall1(ei1) sall1(si1) ball1(bi1)]; 
            end;
        end;
    end;

    indx1 = []; dev1 =[];
    for ci = 1:count
        [apos,~,~] = LJfuncself(paravec1(ci,:),aposprev,aposobs,1);
        xdev2 = (apos(:,1) - aposobs(:,1)).^2;
        ydev2 = (apos(:,2) - aposobs(:,2)).^2;
        dev1(ci) = sum(sqrt(xdev2+ydev2));  
        paravec1(ci,4) = dev1(ci); 
    end;
    mindevval = min(dev1,[],'all');
    [indx1] = find(dev1==mindevval);



    %% step2: use fminsearch to fine-tune

    % Initial guesses for [a, b, c]
    initialParams = [estparainit(countfi,8:10) ];

    x = [];
    x = aposobs;  % B agent traj (ref)
    x = [x aposprev];

    devvalinit = LJfuncParaEstself3(initialParams,x);      
    objectiveFunc = @(parameters) LJfuncParaEstself3(parameters,x);

    % Fit parameters using fminsearch
    options = optimset('MaxFunEvals', 1000,'MaxIter',1000,'Display','off');

    devval = 10^6;   
    for indj = 1:length(indx1)
        initialParams = [paravec1(indx1(indj),1:3) ];    
        % Fit parameters using fminsearch
        fittedParamsij = fminsearch(objectiveFunc, initialParams,options);
        
        % Display fitted parameters
        devvalij = LJfuncParaEstself3(fittedParamsij,x);  
        if devvalij<devval
            fittedParams = fittedParamsij;
            devval = devvalij;
        end;
    end;
       
    if devval<devvalinit
        estpara(countfi,:) = [fittedParams devval];
    else
        estpara(countfi,:) = [initialParams devvalinit];
    end;
 
    [aposgen,~,~] = LJfuncself(estpara(countfi,:),aposprev,aposobs,1);
    aposprev = aposobs; 

end;

estpara = [estpara];

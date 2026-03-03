function [estpara] = forcemodelprevobsB(dataraw)

% A is non-reference agent, estimate self-propelled force and interactive force parameters
% B is reference, estimate self-propelled force parameter
scaler = 100; 
data = dataraw/scaler;   % scale down the position values for compuation precision 1~40

framenum = size(data,1); 
intv = 5;%  % temporal window +/- intv.

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


    %% step 1: grid search

    distself = sqrt((aposobs(2:end,1)-aposobs(1:(end-1),1)).^2+(aposobs(2:end,2)-aposobs(1:(end-1),2)).^2);
    
    eall1 = linspace(0.1,20,10);   %% strength in L-J potential
    turndistA = max(distself);
    sall1 = linspace(0.02,(turndistA),20);%% repul dist in L-J potential
    ball1 = linspace(0,1,10); %% attractive coefficient in L-J potential


    indx1 = []; dev1 = [];
    % self-traj force
    aorig = aposprev(round(size(aposobs,1)/2),:);
    count = 0;
    if mean(distobsself)<0.01   % no movements
        indx1 = 1;
        count = count+1;
        paravec1(indx1,1:3) = [0,0,0];
        [apos,~,f_list,fs_list] = LJfunc(paravec1(1,:),aposprev,aposobs,aorig,1,"self");%YY
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
    
                    [apos,~,f_list,fs_list] = LJfunc(paravec1(count,:),aposprev,aposobs,aorig,1,"self");%YY
    
                    xdev2 = (apos(:,1) - aposobs(:,1)).^2;
                    ydev2 = (apos(:,2) - aposobs(:,2)).^2;
                    dev1(count) = sum(sqrt(xdev2+ydev2));  
                    paravec1(count,4) = dev1(count);
                end;
            end;
        end;
        mindevval = min(dev1,[],'all');
        [indx1] = find(dev1==mindevval);
    end;
   

    %% step2: use fminsearch to fine-tune
    % Initial guesses for [a, b, c]
    initialParams = [paravec1(indx1(1),1:3) ];

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
 
    [aposgen,~] = LJfuncself(estpara(countfi,:),aposprev,aposobs,1);
    aposprev = aposobs; 

end;

estpara = [estpara];

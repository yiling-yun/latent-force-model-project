function [estpara] = forcemodelgenprevA(estpara,dataraw)

estparainit = estpara;
clear estpara; 

% mode = "interactive_selfA";

scaler = 100; 
data = dataraw/scaler;   % scale down the position values for compuation precision 1~40

framenum = size(data,1); 
intv = 5;%  % temporal window +/- intv.
if framenum < 2*intv+1  % for very short videos
    intv = 2;
end
distobjall = sqrt((data(:,3)-data(:,1)).^2+(data(:,4)-data(:,2)).^2);  % distance between objects

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

for fi = 1+intv:intv:framenum-intv
    countfi = countfi+1; 
    turnframe = fi;
    turndist = distobjall(turnframe);
    frameintv = [max(1,turnframe-intv):min(turnframe+intv, size(data,1))];
    
    aposobs = data(frameintv,1:2);
    bposobs = data(frameintv,3:4);
    distobs = sqrt((bposobs(:,1)-aposobs(:,1)).^2+(bposobs(:,2)-aposobs(:,2)).^2);  % distance between objects
    distobsself = sqrt((aposobs(2:end,1)-aposobs(1:(end-1),1)).^2+(aposobs(2:end,2)-aposobs(1:(end-1),2)).^2);
    
    if countfi == 1
       aposprev = aposobs*0;
       aposprev(1+intv,:) = aposobs(1,:);  % important for self-prop force estimate
       aposprev(2+intv,:) = aposobs(2,:);
    end;

    x = data(frameintv,:);
    x = [x aposprev];


    % % corase grid, pred for prev
    aorig = aposprev(round(size(aposobs,1)/2),:);
    distpred = distobs;  % distance between objects
    distselfpred = distobsself;  % distance between objects
    distobsselfprev = sqrt((aposobs(1:end,1)-aorig(1)).^2+(aposobs(1:end,2)-aorig(2)).^2);

    if mean(distobsself)<eps %&& mean(distobsselfprev)<0.01
        eall1 = 0;
        sall1 = 0;
        ball1 = 0;
        eall2 = 0;
        sall2 = 0;
        ball2 = 0;        
    else
        eall1 = [estparainit(countfi,1) 0]; % strength in L-J potential
        turndistA= max([max(distselfpred) max(distobsself)]);
        sall1 = [estparainit(countfi,2) turndistA]; % repul dist in L-J potential
        ball1 = [estparainit(countfi,3) linspace(max([0,estparainit(countfi,3)-0.2]),estparainit(countfi,3)+0.2,3)]; % attractive coefficient in L-J potential
    
        eall2 = [estparainit(countfi,4) 0]; % strength in L-J potential
        turndist = max([max(distpred) max(distobs)]);
        sall2 = [estparainit(countfi,5) turndist]; % repul dist in L-J potential
        ball2 = [estparainit(countfi,6) linspace(max([0,estparainit(countfi,6)-0.2]),estparainit(countfi,6)+0.2,3)]; % attractive coefficient in L-J potential
    end;
    paravec = [];
    count = 0;
    for bi1 = 1:length(ball1)
        for ei1 = 1:length(eall1)
            for si1 = 1:length(sall1)
                 for bi2 = 1:length(ball2)
                    for ei2 = 1:length(eall2)
                        for si2 = 1:length(sall2)
                            count = count +1;
                            paravec(count,:) = [eall1(ei1) sall1(si1) ball1(bi1) eall2(ei2) sall2(si2) ball2(bi2)]; 
                        end;
                    end;
                 end;
            end;
        end;
    end;
        %% step2: use fminsearch to fine-tune

        % Initial guesses for [a, b, c]
        initialParams = estparainit(countfi,1:6);

        devvalinit = LJfuncParaCompforceEst(initialParams,x);  
        objectiveFunc = @(parameters) LJfuncParaCompforceEst(parameters,x);
    
        % Fit parameters using fminsearch
        options = optimset('MaxFunEvals', 1000,'MaxIter',1000,'Display','off');
        % options = optimoptions('fmincon','Algorithm','sqp');
        % options = optimoptions(options,'MaxIterations',1e4,'Display','off'); % Recommended
        % 
        % lb = [0,0.1,0,0,0.1,0];
        % ub = [40,40,10,40,40,10];
        devval = 10^6;  
        for ci = 1:count
            initialParams = paravec(ci,:);    
            %% Fit parameters using fminsearch
            fittedParamsij = fminsearch(objectiveFunc, initialParams,options);
            %% fittedParamsij = fmincon(objectiveFunc, initialParams,[],[],[],[],lb,ub,[],options);
            
            %% Display fitted parameters
            devvalij = LJfuncParaCompforceEst(fittedParamsij,x);  
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
   
  
    if mean(distobsself)<eps && mean(distobsselfprev)<eps
        estpara(countfi,1:6) = [0 0 0 0 0 0];
    end;

    [aposgen,bposgen,flist,fslist] = LJfuncCompforce(estpara(countfi,:),aposprev,aposobs,bposobs,1);
    aposprev = aposgen;
    

end;


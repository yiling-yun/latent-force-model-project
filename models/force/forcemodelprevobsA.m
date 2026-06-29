% force model: use the Leonard Jones Potential
% r is the distance between two interacting particles, 
% epsilon  governs the strength of the interaction, a measure of how strongly two entities attract each other
% sigma  is the distance at which the particle-particle potential energy V is zero (often referred to as 'size of the particle'). 
% The Lennard-Jones potential has its minimum at a distance of r=2^{1/6}*sigma 
% where the potential energy has the value {\displaystyle V=-eps.

function [estpara] = forcemodelprevobsA(dataraw)

mode = "interactive_selfA";

scaler = 100; 
data = dataraw/scaler;   % scale down the position values for compuation precision 1~40

framenum = size(data,1); 
intv = 5;%  % temporal window +/- intv.
% if framenum < 2*intv+1  % for very short videos
%     intv = 2;
% end
distobjall = sqrt((data(:,3)-data(:,1)).^2+(data(:,4)-data(:,2)).^2);  % distance between objects

xx = data(:,1);
yy = data(:,2); 

countfi = 0; 

for fi = 1+intv:intv:framenum-intv
    countfi = countfi+1; 
    %disp(fi);
    turnframe = fi;
    frameintv = [max(1,turnframe-intv):min(turnframe+intv, size(data,1))];
    
    aposobs = data(frameintv,1:2);
    bposobs = data(frameintv,3:4);
    distobs = sqrt((bposobs(:,1)-aposobs(:,1)).^2+(bposobs(:,2)-aposobs(:,2)).^2);  % distance between objects
    distobsself = sqrt((aposobs(2:end,1)-aposobs(1:(end-1),1)).^2+(aposobs(2:end,2)-aposobs(1:(end-1),2)).^2); % distance between frames for an agent
    if countfi == 1
       aposprev = aposobs*0;
       aposprev(1+intv,:) = aposobs(1,:);  % important for self-prop force estimate
       aposprev(2+intv,:) = aposobs(2,:);
    end;

    %% orig, obs for prev
    % eall1 = linspace(0.1,20,50);   %% strength in L-J potential
    % turndistA= distobjallself(turnframe);
    % sall1 = linspace(max(0,(turndistA-1)),(turndistA+1),50);%% repul dist in L-J potential
    % ball1 = linspace(0,0.5,50); %% attractive coefficient in L-J potential
    % 
    % eall2 = linspace(0.1,20,50);   %% strength in L-J potential
    % sall2 = linspace(max(0.3,(turndist-2)),(turndist+2),50);%% repul dist in L-J potential
    % ball2 = linspace(0,0.5,50); %% attractive coefficient in L-J potential

    %% coarse grid, obs for prev
    % eall1 = linspace(0.1,20,10);   %% strength in L-J potential
    % turndistA= distobjallself(turnframe);
    % sall1 = linspace(max(0,(turndistA-1)),(turndistA+1),20);%% repul dist in L-J potential
    % ball1 = linspace(0,0.5,10); %% attractive coefficient in L-J potential
    % 
    % eall2 = linspace(0.1,20,10);   %% strength in L-J potential
    % sall2 = linspace(max(0.3,(turndist-2)),(turndist+2),20);%% repul dist in L-J potential
    % ball2 = linspace(0,0.5,10); %% attractive coefficient in L-J potential

    % corase grid, pred for prev
    aorig = aposprev(round(size(aposobs,1)/2),:);
    
        % 0.4667
        eall1 = linspace(0.1,20,10);   %% strength in L-J potential
        turndistA= max(distobsself);
        sall1 = linspace(0.02,(turndistA),20);%% repul dist in L-J potential
        ball1 = linspace(0,1,10); %% attractive coefficient in L-J potential

        eall2 = linspace(0.1,20,10);   %% strength in L-J potential
        turndist = max(distobs);
        sall2 = linspace(0.3,(turndist)+2,20);%% repul dist in L-J potential
        ball2 = linspace(0,1,10); %% attractive coefficient in L-J potential


        % eall1 = linspace(0.1,20,20);   %% strength in L-J potential
        % turndistA= max(distobsself);
        % sall1 = linspace(0.02,(turndistA),40);%% repul dist in L-J potential
        % ball1 = linspace(0,1,20); %% attractive coefficient in L-J potential
        % 
        % eall2 = linspace(0.1,20,20);   %% strength in L-J potential
        % turndist = max(distobs);
        % sall2 = linspace(0.3,(turndist)+2,40);%% repul dist in L-J potential
        % ball2 = linspace(0,1,20); %% attractive coefficient in L-J potential

        % 
        indx1 = []; indx2 = []; dev1 =[]; dev=[];
        % self-traj force
   
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
        
                        [apos,~,f_list,fs_list] = LJfunc(paravec1(count,1:3),aposprev,aposobs,aorig,1,"self");%YY
        
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
    
        % interactive force
        count = 0;
        val = mean(sqrt((aposobs(:,1)-bposobs(:,1)).^2+(aposobs(:,2)-bposobs(:,2)).^2));    
        
        if val<0.01   % no movements
            indx2 = 1;
            count = count+1;
            paravec2(indx2,1:3) = [0,0,0];
            [apos,~] = LJfunc(paravec2(1,:),aposprev,aposobs,bposobs,1, "interactive");

            xdev2 = (apos(:,1) - aposobs(:,1)).^2;
            ydev2 = (apos(:,2) - aposobs(:,2)).^2;
            paravec2(1,4) = sum(sqrt(xdev2+ydev2));  

        else
            for bi = 1:length(ball2)
                for ei = 1:length(eall2)
                    for si = 1:length(sall2)
                        epsilon = eall2(ei); 
                        sigma = sall2(si); 
                        bcoef = ball2(bi);
                        count = count+1;
                        paravec2(count,1:3) = [epsilon,sigma,bcoef];
        
                        [apos,~] = LJfunc(paravec2(count,1:3),aposprev,aposobs,bposobs,1, "interactive");
        
                        xdev2 = (apos(:,1) - aposobs(:,1)).^2;
                        ydev2 = (apos(:,2) - aposobs(:,2)).^2;
                        dev(count) = sum(sqrt(xdev2+ydev2));  
                        paravec2(count,4) = dev(count);
                    end;
                end;
            end;
        
            mindevval = min(dev,[],'all');
            [indx2] = find(dev==mindevval);
        end;
    
        % if size(paravec1,1)>1
        %     [sorttemp1, indxr1]=sortrows(paravec1,4);
        %     if length(indx1) == 1
        %         indx1 = indxr1(1:1);
        %     end;
        % end; 
        % if size(paravec2,1)>1        
        %     [sorttemp2, indxr2]=sortrows(paravec2,4);
        %     if length(indx2) == 1
        %         indx2 = indxr2(1:1);
        %     end;
        % end;
        %% step2: use fminsearch to fine-tune
        if length(indx1)>1 ||length(indx2)>1
            if length(indx1)>500 ||length(indx2)>500
                warning(['frame ' num2str(fi) ', more min init:' num2str(length(indx1)) ', ' num2str(length(indx2))]);
            end
        end
    
        % if length(indx1)>1
        %     if sum(paravec1(indx1,2))==0 %||length(indx1)==125000 ||length(indx2)==125000  % estimated sigma is 0. 
        %         % warning(['sigma (grid search) = 0' ]);
        %         indx1 = 1;
        %         paravec1(indx1,:) = [0,0,0,0];
        %     end
        % end
        % Initial guesses for [a, b, c]
        initialParams = [paravec1(indx1(1),1:3) paravec2(indx2(1),1:3)];
    
        x = [];
        x = data(frameintv,:);
        x = [x aposprev];
        devvalinit = LJfuncParaEstobs(initialParams,x);      
        objectiveFunc = @(parameters) LJfuncParaEstobs(parameters,x);
    
        % Fit parameters using fminsearch
        options = optimset('MaxFunEvals', 1000,'MaxIter',1000,'Display','off');
        % options = optimoptions('fmincon','Algorithm','sqp');
        % options = optimoptions(options,'MaxIterations',1e4,'Display','off'); % Recommended
        % 
        % lb = [0,0.1,0,0,0.1,0];
        % ub = [40,40,10,40,40,10];
        % if length(indx1)==1 && length(indx2)==1
        %     initialParams = [paravec1(indx1(1),1:3) paravec2(indx2(1),1:3)];
        %     %fittedParams = fminsearch(objectiveFunc, initialParams,options);
        %     fittedParams = fmincon(objectiveFunc, initialParams,[],[],[],[],lb,ub,[],options);
        %     devval = LJfuncParaEst3(fittedParams,x);  
        % else
            devval = 10^6;   
            for indj = 1:length(indx1)
                for indk = 1:length(indx2)
                    initialParams = [paravec1(indx1(indj),1:3) paravec2(indx2(indk),1:3)];    
                    % Fit parameters using fminsearch
                    fittedParamsij = fminsearch(objectiveFunc, initialParams,options);
                    % fittedParamsij = fmincon(objectiveFunc, initialParams,[],[],[],[],lb,ub,[],options);
                    
                    % Display fitted parameters
                    devvalij = LJfuncParaEstobs(fittedParamsij,x);  
                    if devvalij<devval
                        fittedParams = fittedParamsij;
                        devval = devvalij;
                    end;
                end;
            end;
        % end;
    
    
        if devval<devvalinit
            estpara(countfi,:) = [fittedParams devval];
        else
            estpara(countfi,:) = [initialParams devvalinit];
        end;
    % end;
    [aposgen,bposgen,flist,fslist] = LJfuncCompforceobs(estpara(countfi,:),aposprev,aposobs,bposobs,1);
    aposprev = aposobs;
    

end;


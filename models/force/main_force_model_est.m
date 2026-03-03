% read in trajectory info for charade videos and compute similarity matrix
% main change: take the predicted first two frames, not observed two frames
% cross the neighbor temporal windows

clear all; 
close all; 

input = 'stim';% 'all';% %% 'all' has 1133 animations. 'stim' has 27 selected animations.

vermodel = 3; 

reverseflag = 0; % 1: reverse the frame order for Exp 3; 0: Exp 1

loadflag = 0; % 1: load the previous saved file; 0; compute the file
plotflagtraj = 1; % 1: plot traj and save in files; 0: no traj plot

rng(1); 
[num,txt,~] = xlsread('charades_traj_summary.xlsx','all');

if strcmp(input, 'stim')
    exlab = 'exp2';
    [numS,txtS,~] = xlsread('charades_traj_summary.xlsx','selected_exp2');
    vidnum = length(numS);
    for i = 1:vidnum
        rowi = find(numS(i)==num(:,2));
        idxsel(i,1) = rowi;
        semlabel{i,1} = txtS{i};
            for j = 1:6  % x1, y1, ori1, x2, y2, ori2
                temp = txt{idxsel(i)+1,3+j}; % x1
                temp2 = str2double((strsplit((temp(3:end-2)),''', ''')));
                cordcell{i,j} = temp2;        
            end
        frameN(i,1) = length(temp2);
    end
else
    exlab = 'all_1133';
    vidnum = length(num);
    for i = 1:vidnum
        idxsel(i,1) = i;
        allsemlabel{i,1} = txt{idxsel(i)+1,2};
        for j = 1:6  % x1, y1, ori1, x2, y2, ori2
            temp = txt{idxsel(i)+1,3+j}; % x1
            temp2 = str2double((strsplit((temp(3:end-2)),''', ''')));
            cordcell{i,j} = temp2;        
        end
        frameN(i,1) = length(temp2);
    end
end


%% estimate parameters using force model from force parameter
hfig = figure;

% Build directory path
savedir = ['./rst/' exlab];

% Create directory if it doesn't exist
if ~exist(savedir, 'dir')
    mkdir(savedir);
end

for i = 1:vidnum 
    disp(i);
    clf(hfig);
    tic;
    x=[]; y=[]; vx=[]; vy=[]; sp1=[]; sp2=[]; acx=[]; acy=[]; dataraw=[];
    if reverseflag == 0 % 1: reverse the frame order; 0: Exp 1
        x(1,:) = cordcell{i,1};
        y(1,:) = cordcell{i,2}; 
        x(2,:) = cordcell{i,4};
        y(2,:) = cordcell{i,5};         
    elseif reverseflag == 1
        x(1,:) = fliplr(cordcell{i,1});
        y(1,:) = fliplr(cordcell{i,2}); 
        x(2,:) = fliplr(cordcell{i,4});
        y(2,:) = fliplr(cordcell{i,5});    
    end 
    dataraw = [x(1,:)' y(1,:)' x(2,:)' y(2,:)'];

    % pad the first clip
    datapad = repmat([x(1,1) y(1,1) x(2,1) y(2,1)],11,1);
    dataraw = [datapad; dataraw];

    % use obs positions for previous time window
    [estparaA] = forcemodelprevobsA(dataraw);  % add B self force parameter
    [estparaB] = forcemodelprevobsB(dataraw);  % add B self force parameter
    estparaObsprev{i} = [estparaA estparaB];

    % use gen positions for previous time window
    [estparaA2] = forcemodelgenprevA(estparaObsprev{i},dataraw);
    [estparaB2] = forcemodelgenprevB(estparaObsprev{i},dataraw);
    estpara{i} = [estparaA2 estparaB2];

    toc; 
    if plotflagtraj ==1
        framenum = size(dataraw,1);
        scaler = 100; 

        intv = 5;
        framesall = 1+intv:intv:framenum-intv;
        estparavid = estpara{i};
        aposobs =[]; bposobs=[]; aposgen=[]; bposgen=[];
        for fi = 1:size(estparavid,1)
            turnframe = framesall(fi);
            framerange = [max(1,turnframe-intv):min(turnframe+intv, framenum)];
    
            aposobs{fi} = [dataraw(framerange,1) dataraw(framerange,2)]/scaler;
            bposobs{fi} = [dataraw(framerange,3) dataraw(framerange,4)]/scaler;


            if fi==1
                aposprev = aposobs{fi}*0;
                aposprev(intv+1,:) = aposobs{fi}(1,:);  % important for self-prop force estimate
                aposprev(intv+2,:) = aposobs{fi}(2,:);
                bposprev = bposobs{fi}*0;
                bposprev(intv+1,:) = bposobs{fi}(1,:);  % important for self-prop force estimate
                bposprev(intv+2,:) = bposobs{fi}(2,:);                              
            end;

            [aposgen{fi},bposgen{fi},force] = LJfuncCompforce(estparavid(fi,:),aposprev,aposobs{fi},bposobs{fi},1);
            aposprev = aposgen{fi}; 
            [bposgen{fi},~] = LJfuncself(estparavid(fi,8:10),bposprev,bposobs{fi},1);
            bposprev = bposobs{fi}; 
        end;
    
        dataall = [];
        for j = 1:size(aposgen,2)
            dataall(j,:) = [aposobs{j}(intv+1,1) aposobs{j}(intv+1,2) ...
                bposobs{j}(intv+1,1) bposobs{j}(intv+1,2) ...
                aposgen{j}(intv+1,1) aposgen{j}(intv+1,2) ...
                bposgen{j}(intv+1,1) bposgen{j}(intv+1,2)]*scaler;
        end;
    
        plot(dataall(:,1),dataall(:,2),'-or');hold on;  % obs a
        plot(dataall(:,3),dataall(:,4),'-ob'); hold on; % obs b
    
        plot(dataall(:,5),dataall(:,6),'-+g'); hold on; % pred a
        plot(dataall(:,7),dataall(:,8),'-+k'); hold on; % pred b
    
        plot(dataall(1,1),dataall(1,2),'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r');hold on;
        plot(dataall(1,3),dataall(1,4),'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b'); hold on;
    
        xlim([0 4000]);  ylim([0 3800]);
        trialnum = ['trial-' num2str(i)];
        title(trialnum);
        xlim([50 4000]);  ylim([50 3800]);
        saveas(hfig, [savedir '/V' num2str(vermodel) '_trial' trialnum '.png']);
    end;
end

if strcmp(input,'all')
    save([savedir '/estpart_forcemodel.v' num2str(vermodel) '_all.mat'],'estpara');
    save([savedir '/estpart_forcemodel.v' num2str(vermodel) '_all.prevobs.mat'],'estparaObsprev');
else
    save([savedir '/estpart_forcemodel.v' num2str(vermodel) '.mat'],'estpara');
    save([savedir '/estpart_forcemodel.v' num2str(vermodel) '.prevobs.mat'],'estparaObsprev');
end




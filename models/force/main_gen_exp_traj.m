% read in trajectory info for charade videos 
% use the pre-computed force parameters to generate trajectories by
% recombining B traj and force field

clear all; 
close all; 

exlab = 'exp3_gen';

% all input changed to read from excel file for reference across code
loadflag = 1; % 1: load the previous saved file; 0; compute the file
plotflagtraj = 1; % 1: plot traj and save in files; 0: no traj plot

scaler = 100; 
intv = 5;   
rng(1); 
[id,txt,~] = xlsread('./charades_traj_summary.xlsx','all');
[idForceAll,txtForceAll,~] = xlsread('./charades_traj_summary.xlsx','selected_exp2');
[idForce,~,~] = xlsread('./charades_traj_summary.xlsx','selected_exp3_forceA');
[idB,txtB,~] = xlsread('./charades_traj_summary.xlsx','selected_exp3_trajB');

% get the trajectory info of the selected animations for Traj B
for i = 1:length(idB)
    idxselTrajB(i,1) = find(idB(i)==id(:,2)); %indB(i); % find videos based on unique ID
    semlabelTrajB{i,1} = txtB{i}; % txt{idxselTrajB(i)+1,2};
    for j = 1:6  % x1, y1, ori1, x2, y2, ori2
        temp = txt{idxselTrajB(i)+1,3+j}; % x1
        temp2 = str2double((strsplit((temp(3:end-2)),''', ''')));
        cordcellTrajB{i,j} = temp2;        
    end
end

% get the trajectory info of the selected animations for 27 force videos
for i = 1:length(idForceAll)
    idxsel(i,1) = find(idForceAll(i)==id(:,2));
    semlabel{i,1} = txtForceAll{i};
    for j = 1:6  % x1, y1, ori1, x2, y2, ori2
        temp = txt{idxsel(i)+1,3+j}; % x1
        temp2 = str2double((strsplit((temp(3:end-2)),''', ''')));
        cordcell{i,j} = temp2;        
    end;
    % frameN(i,1) = length(temp2);
end

for i = 1:length(idForce)
    idxEstpara(i, 1) = find(idForce(i)==idForceAll);
end


%% rst 2: simlarity matrix from force parameter
tic;
if loadflag == 1
    load(['./rst/exp2/estpart_forcemodel.v3.mat'],'estpara');
end
toc; 
%%

hfig = figure;

% indBInAll = idB;
% indForceInAll = indForce;

for fj = 1:length(idForce) 
    for i = 1:length(idB) 
        disp([i fj]);
        x=[]; y=[]; vx=[]; vy=[]; sp1=[]; sp2=[]; acx=[]; acy=[];

        bindex = i;
        x(1,:) = cordcellTrajB{bindex,1};
        y(1,:) = cordcellTrajB{bindex,2}; 
        x(2,:) = cordcellTrajB{bindex,4};
        y(2,:) = cordcellTrajB{bindex,5};     
    
        dataraw_traj = []; aposgen = []; bposgen = []; aposobs = []; bposobs = []; 
        dataraw_traj = [x(1,:)' y(1,:)' x(2,:)' y(2,:)'];
        % % pad the first clip
        datapad = repmat(dataraw_traj(1,:),2*intv+1,1);
        dataraw_traj = [datapad; dataraw_traj];

        findex = idxEstpara(fj);
        dataraw_force = [cordcell{findex,1}' cordcell{findex,2}' cordcell{findex,4}' cordcell{findex,5}' ];
        % % pad the first clip
        datapad = repmat(dataraw_force(1,:),2*intv+1,1);
        dataraw_force = [datapad; dataraw_force];
 
        num_traj = size(dataraw_traj,1); 
        num_force = size(dataraw_force,1); 
        if num_force >= num_traj 
            error('force frame should be shorter than trajB frame');
            framenum = num_traj; 
        else            
            framenum = num_force;
        end
        
        % mirror along x axis
        [distXmirror, dataXmirror] = mirrorx(dataraw_force(1:framenum,:),dataraw_traj(1:framenum,:));
        [distYmirror, dataYmirror] = mirrory(dataraw_force(1:framenum,:),dataraw_traj(1:framenum,:));
        
        if distXmirror <= distYmirror 
            dataraw_force = dataXmirror;
        else
            dataraw_force = dataYmirror;
        end

        estparavid = []; framesall = []; 
        aposobs =[]; bposobs=[];  aposobsf=[]; bposobsf=[];aposgen=[]; bposgen=[];
        moveflag = []; countnomovefr = 0; 
        framesall = 1+intv:intv:framenum-intv;
        estparavid = estpara{findex}(1:end,:);  % remove the first padded clip     
        for fi = 1:size(framesall,2)
            turnframe = framesall(fi);
            framerange = [max(1,turnframe-intv):min(turnframe+intv, framenum)];
    
            aposobs{fi} = [dataraw_traj(framerange,1) dataraw_traj(framerange,2)]/scaler;
            bposobs{fi} = [dataraw_traj(framerange,3) dataraw_traj(framerange,4)]/scaler;

            aposobsf{fi} = [dataraw_force(framerange,1) dataraw_force(framerange,2)]/scaler;
            bposobsf{fi} = [dataraw_force(framerange,3) dataraw_force(framerange,4)]/scaler;

            if fi==1
                aposprev = aposobs{fi}*0;
                aposprev(intv+1,:) = aposobs{fi}(1,:);  % important for self-prop force estimate
                aposprev(intv+2,:) = aposobs{fi}(2,:);

                aposprevf = aposobsf{fi}*0;
                aposprevf(intv+1,:) = aposobsf{fi}(1,:);  % important for self-prop force estimate
                aposprevf(intv+2,:) = aposobsf{fi}(2,:);  

            end
          
            moveflag(fi) = sum(abs(diff(aposobsf{fi})),"all");

            if sum(moveflag(1:fi)) == 0  % no movements for agentA in the force video
                aposgen{fi} = aposobsf{1}-(bposobsf{1}-bposobs{1});  
                aposprevf = aposgen{fi}  ;
                countnomovefr = countnomovefr+1;
            else
                aposgentransf{fi} = aposobsf{fi}+(bposobs{fi-countnomovefr}-bposobsf{fi-countnomovefr});  
                [aposgen{fi},bposgen{fi},force2, forcesel] = LJfuncCompforceGen1(estparavid(fi,:),aposgentransf{fi}, aposprevf,aposobsf{fi},bposobsf{fi},aposprev,aposobs{fi},bposobs{fi},1);
                aposprev = aposgen{fi}; %  %use pred pos for prev   
            end
        end

        aposgenall{i,fj} = aposgen;
        bposgenall{i,fj} = bposobs;
    
        if plotflagtraj ==1
            clf(hfig);
            intv = 5;
            data = [];
            for j = 1:size(aposgen,2)
                data(j,:) = [aposobs{j}(intv+1,1) aposobs{j}(intv+1,2) ...
                    bposobs{j}(intv+1,1) bposobs{j}(intv+1,2) ...
                    aposgen{j}(intv+1,1) aposgen{j}(intv+1,2) ...
                    bposobs{j}(intv+1,1) bposobs{j}(intv+1,2) ...
                    aposobsf{j}(intv+1,1) aposobsf{j}(intv+1,2) ...
                    bposobsf{j}(intv+1,1) bposobsf{j}(intv+1,2)]*scaler; 
            end
        
            plot(data(:,3),data(:,4),'-ob'); hold on; % obs b, traj video

            plot(data(:,5),data(:,6),'-+g'); hold on; % pred a
            plot(data(:,7),data(:,8),'-+k'); hold on; % pred b

            plot(data(1,1),data(1,2),'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r'); hold on;
            plot(data(1,3),data(1,4),'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b'); hold on;
            plot(data(1,5),data(1,6),'o', 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g'); hold on;
            trialnum = ['trial:' ' force:' num2str(fj) semlabel{findex}  '  agentB: ' num2str(i) semlabelTrajB{bindex}];
            title(trialnum);


            xlim([50 4000]);  ylim([50 3800]); 
            pause(0.5);

            trialNumFile = ['trial' '_fj' num2str(fj) semlabel{findex} '_agentB' num2str(i) semlabelTrajB{bindex}];
            
            dir = ['./rst/' exlab '/gen/'];
            if ~exist(dir, 'dir')
                mkdir(dir);
            end
            saveas(hfig, [dir 'gentraj_' trialNumFile '.png']);

        end
    end
end
%% generate videos
for fj = 1:length(idForce) 
    for i = 1:length(idB) 
        disp([i fj]);
        findex = idxEstpara(fj);
        bindex = i;

        aposgen = aposgenall{i,fj} ;
        bposgen = bposgenall{i,fj} ;

        intv = 5;
        data = [];
        count = 1;
        for j = 1:size(aposgen,2)
            for k = 1:floor(size(aposgen{j},1)/2) % YY: get the first half of the window
                data(count,:) = [aposgen{j}(k,1) aposgen{j}(k,2) ...
                    bposgen{j}(k,1) bposgen{j}(k,2)]*scaler;   
                count = count+1;
            end
        end

        genposall{i,fj} = data; 
        trialNumFile = ['trial' '_fj' num2str(fj) semlabel{findex} '_agentB' num2str(i) semlabelTrajB{bindex} ];
        filename = strrep(['.\rst\' exlab '\gen\genvid_' trialNumFile '.gif'], '\', '/');
        % generate videos
        for j = 1:size(data,1)    
            plot(data(j,1),data(j,2),'o','MarkerSize',20,  'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'b');hold on;  % obs a
            plot(data(j,3),data(j,4),'o', 'MarkerSize',20,'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'g'); hold off; % obs b
        
            xlim([50 4000]);  ylim([50 3800]); 
            axis square; axis off; 
            drawnow;
            % Capture frame
            frame = getframe(gcf);
            img = frame2im(frame);
            [imind, cm] = rgb2ind(img, 256);        

            % Write to GIF file
            if j == 1
                imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
            else
                imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
            end
        end
    end
end

save(['./rst/' exlab '/rst_ExpTrajGenPos.mat'],'genposall');



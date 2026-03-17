% read in trajectory info for charade videos and compute similarity matrix

clear all; 
close all; 

exlab = 'exp2'; 

saverst = 1; % save distance matrix

% get labels of estpara
rng(1); 
[numS,txtS,~] = xlsread('./charades_traj_summary.xlsx','selected_exp2');
semlabel = txtS(1:end,1);
vidnum = length(numS);

labeldistmatrix = ones(vidnum,vidnum);

% read in visual features
[num,txt,~] = xlsread('./charades_traj_summary.xlsx','all');

% split_data = split(semlabel, "_");  % Returns an N×2 string array
% numS = double(split_data(:,1));  
% txtS = split_data(:,2); 

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

%% get features of the selected videos
reverseflag = 0;
for i = 1:vidnum 
    x=[]; y=[]; vx=[]; vy=[]; sp1=[]; sp2=[]; acx=[]; acy=[];
    if reverseflag == 0 % 1: reverse the frame order for Exp 3; 0: Exp 1
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
    dataraw = [];  aposobs = []; bposobs = []; 
    dataraw0 = [x(1,:)' y(1,:)' x(2,:)' y(2,:)'];  % a: 1-2; b: 3-4000

    %% take the stride as force model
    % step 1: pad the first clip
    datapad = repmat([x(1,1) y(1,1) x(2,1) y(2,1)],11,1);
    dataraw0 = [datapad; dataraw0];

    % step 2: frame stride
    framenum = size(dataraw0,1); 
    intv = 5;%  % temporal window +/- intv.
    framesel = 1+intv:intv:framenum-intv;

    dataraw = dataraw0(framesel,:);

    datadist = sqrt((dataraw(:,1)-dataraw(:,3)).^2 + (dataraw(:,2)-dataraw(:,4)).^2);
    
    datapadv = repmat(dataraw(1,:),1,1);
    datarawv = [datapadv; dataraw];
    datav = diff(datarawv,1);

    datapada = repmat(datav(1,:),1,1);
    datarawa = [datapada; datav];
    dataa = diff(datarawa,1);

    estpara{i} = [datav dataa datadist];
end

feavecindx = 1:size(estpara{1},2); 

rstparamat = [];
for i = 1:vidnum 
    rstparamat = [rstparamat; [estpara{i} ones(size(estpara{i},1),1)*i]];
end
meanpara = mean(rstparamat);
varpara = var(rstparamat);
stdpara = std(rstparamat);

%% Method: use hist of force features
feavecindxsel = feavecindx; 
stdscale = ones(1,length(feavecindx))*20; 

% Define the number of bins
numBins = ones(1,length(feavecindx))*200; 
for i = 1:length(feavecindx)
    binedges{i} = linspace(meanpara(i)-stdscale(i)*stdpara(i),meanpara(i)+stdscale(i)*stdpara(i),numBins(i));
end

% compute historame of parameter over frames
for i = 1:vidnum
    datai = []; counts1fi=[];
    datai = estpara{i}(:,feavecindx);

    % remove no movement frames
    indx_d = [];
    datairaw = estpara{i}(:,feavecindx);
    indx_d = find(sum(abs(datairaw(:,1:4)),2)==0);
    datai(indx_d,:)=[];

    for fi = 1:length(feavecindxsel)
        fival = feavecindxsel(fi);
        [counts1fi(fi,:),~] = histcounts(datai(:,fival), binedges{fival}, 'Normalization', 'probability');
    end
    countsall{i} = counts1fi;
end

% Compute histograms
for i = 1:vidnum
    for j = 1:vidnum 
        if i == j
            distval(i,j) = 0;
        else
            for fi = 1:length(feavecindxsel)
                counts1 = countsall{i}(fi,:);
                counts2 = countsall{j}(fi,:);
                % Euclidean distance
                distvalfi(fi) = sqrt(sum((counts1 - counts2).^2));
                           
            end
            distval(i,j) = sum(distvalfi);
            distvalfiall{i,j} = distvalfi;
        end       
    end
end

kinefeatdistval = distval;%log(distval+1);%

% Reorder distance matrix to match desired order
desiredOrder = readlines('./../../utils/video_order_from_hc_human_dmat.txt');
desiredOrder = desiredOrder(desiredOrder ~= "");
desiredIDs = cellfun(@(x) str2double(extractBefore(x,'_')), cellstr(desiredOrder));
videoIDs = numS(:, 1); % whichever column has the ID
[~, reorderIdx] = ismember(desiredIDs, videoIDs);

% Check for mismatches
if any(reorderIdx == 0)
    disp('Unmatched IDs:');
    disp(desiredIDs(reorderIdx == 0));
end

% Reorder
kinefeatdistval = kinefeatdistval(reorderIdx, reorderIdx);
semlabel = semlabel(reorderIdx);

% % save dissimilarity matrices
if saverst ==1
    dir = './distMat';
    if ~exist(dir, 'dir')
        mkdir(dir);
    end
    writematrix(kinefeatdistval, [dir '/kinematic_feat_hist_dist.csv']);
end

figure; imagesc(kinefeatdistval);colorbar; title(['kinematic feature model']);
xticks(1:vidnum);xticklabels(semlabel); yticks([1:vidnum]);yticklabels(semlabel);axis square;
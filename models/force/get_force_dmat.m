% compute similarity matrix from learned force parameters

clear all; 
close all; 

exlab = 'exp2';
vermodel = 2; 

input = 'stim'; % 'stim' has 54 selected animations.


saverst = 1; % save distance matrix

load(['./rst/exp2/estpart_forcemodel.v' num2str(vermodel) '_all.mat'],'estpara');

[numS,txtS,~] = xlsread('charades_traj_summary.xlsx','selected_exp2');
semlabel = txtS(1:end,1);
vidnum = length(numS);

% add force vec for estpara
for i = 1:vidnum 
    estpara{i}=[estpara{i}(:,1:6) estpara{i}(:,8:10)];    % selfA, intA, selfB
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
    indx_d=[];
    datairaw = estpara{i}(:,feavecindx);
    indx_d = find(sum(abs(datairaw(:,1:6)),2)+sum(abs(datairaw(:,7:9)),2)==0);
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

forcemoddistval = distval;%log(distval+1);%

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
forcemoddistval = forcemoddistval(reorderIdx, reorderIdx);
semlabel = semlabel(reorderIdx);

% % save dissimilarity matrices
if saverst ==1
    dir = './distMat';
    if ~exist(dir, 'dir')
        mkdir(dir);
    end
    writematrix(forcemoddistval, [dir '/force_hist_dist.csv']);
end

figure; 
imagesc(forcemoddistval);colorbar; title(['force model']);
xticks(1:vidnum);xticklabels(semlabel); yticks([1:vidnum]);yticklabels(semlabel);axis square;

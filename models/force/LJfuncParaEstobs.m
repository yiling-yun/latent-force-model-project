function [sse] = LJfuncParaEstobs(parameters,x)

aposobs = x(:,1:2);  % A agent traj
bposobs = x(:,3:4);  % B agent traj (ref)
aposprev = x(:,5:6); % A agent previous time point
% List of forces 
[apos,bpos,~] = LJfuncCompforceobs(parameters,aposprev,aposobs,bposobs,1);

xpred(:,1:2) = apos;
xpred(:,3:4) = bpos;

% objective function: minimize dist between obs and pred; parameters should
% be greater than 0
xobs = x(:,1:4);
sse = mean(sqrt(sum((xpred-xobs).^2,2))) + ...
    10^6*((parameters(1)<0)+(parameters(1)>40)+(parameters(2)<0)+(parameters(2)>40)+(parameters(3)<0)+...
    (parameters(4)<0)+(parameters(4)>40)+(parameters(5)<0)+(parameters(5)>40)+(parameters(6)<0));
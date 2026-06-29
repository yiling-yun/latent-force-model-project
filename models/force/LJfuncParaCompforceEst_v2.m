function [sse] = LJfuncParaCompforceEst_v2(parameters, x)
% V2 objective for Pass 2 Agent A.

aposobs  = x(:,1:2);
bposobs  = x(:,3:4);
aposprev = x(:,5:6);

[apos, bpos, ~] = LJfuncCompforce(parameters, aposprev, aposobs, bposobs, 1);

xpred(:,1:2) = apos; xpred(:,3:4) = bpos;
xobs = x(:,1:4);
sse = mean(sqrt(sum((xpred-xobs).^2,2))) + ...
    10^6*((parameters(1)<0)+(parameters(1)>40)+(parameters(2)<0)+(parameters(2)>40)+(parameters(3)<0)+(parameters(3)>2)+...
    (parameters(4)<0)+(parameters(4)>40)+(parameters(5)<0)+(parameters(5)>40)+(parameters(6)<0)+(parameters(6)>2));

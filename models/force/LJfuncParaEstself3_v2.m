function [sse] = LJfuncParaEstself3_v2(parameters, x)
% V2 objective for Pass 1 Agent B self-force.

aposobs  = x(:,1:2);
aposprev = x(:,3:4);

[apos, ~] = LJfuncself(parameters, aposprev, aposobs, 1);

sse = mean(sqrt(sum((apos-aposobs).^2,2))) + ...
    10^6*((parameters(1)<0)+(parameters(1)>40)+(parameters(2)<0)+(parameters(2)>40)+(parameters(3)<0)+(parameters(3)>2));

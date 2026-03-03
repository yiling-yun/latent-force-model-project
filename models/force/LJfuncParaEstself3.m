function [sse] = LJfuncParaEstself3(parameters,x)

aposobs = x(:,1:2);  % B reference agent
aposprev = x(:,3:4);
% List of forces 
[apos,~] = LJfuncself(parameters,aposprev,aposobs,1);

sse = mean(sqrt(sum((apos-aposobs).^2,2))) + ...
    10^6*((parameters(1)<0)+(parameters(1)>40)+(parameters(2)<0)+(parameters(2)>40)+(parameters(3)<0));
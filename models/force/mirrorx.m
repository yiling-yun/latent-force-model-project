function [distmirror, datamirror] = mirrorx(data,datatarget)

P = data(:,1:2)-data(1,1:2);
datamirror(:,1:2) = [P(:,1) -P(:,2)]+data(1,1:2);
P = data(:,3:4)-data(1,3:4);
datamirror(:,3:4) = [P(:,1) -P(:,2)]+data(1,3:4);

temp1 = datamirror(:,3)-datatarget(:,3);
temp2 = datamirror(:,4)-datatarget(:,4);
distmirror = sum(sqrt(temp1.^2 + temp2.^2 ));

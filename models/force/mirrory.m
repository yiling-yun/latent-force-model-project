function [distmirror, datamirror] = mirrorx(data,datatarget)

orig = (data(1,1:2)+data(1,3:4))/2; 
P = data(:,1:2)-orig;
datamirror(:,1:2) = [-P(:,1) P(:,2)]+orig;
P = data(:,3:4)-orig;
datamirror(:,3:4) = [-P(:,1) P(:,2)]+orig;

temp1 = datamirror(:,3)-datatarget(:,3);
temp2 = datamirror(:,4)-datatarget(:,4);
distmirror = sum(sqrt(temp1.^2 + temp2.^2 ));

        % mirror along y axis
        % orig = (dataraw_force(1,1:2)+dataraw_force(1,3:4))/2; 
        % P = dataraw_force(1:framenum,1:2)-orig;
        % dataraw_force(1:framenum,1:2) = [-P(:,1) P(:,2)]+orig;
        % P = dataraw_force(1:framenum,3:4)-orig;
        % dataraw_force(1:framenum,3:4) = [-P(:,1) P(:,2)]+orig;
clear;clc;close all;
thicken = 2;

figure
colormap gray
set(gcf,'position',[15 134 379 328])
load(['~/Dropbox/Two_Photon/M' num2str(1) '.mat'])
map = padarray(map, [thicken+2 thicken+2], 0, 'both');
map = bwmorph(map,'remove');
map = imdilate(map, strel('square', thicken));
map = double(map);
map(map == 0) = nan;
h = imagesc(map);
set(h, 'AlphaData', ~isnan(map));
axis off
title('Mouse M1')
drawnow

figure
colormap gray
set(gcf,'position',[395 134 375 328])
cc = 0;
for mm = 2:5
    cc = cc + 1;
    subplot(2,2,cc)
    load(['~/Dropbox/Two_Photon/M' num2str(mm) '.mat'])
    map = padarray(map, [thicken+2 thicken+2], 0, 'both');
    map = bwmorph(map,'remove');
    map = imdilate(map, strel('square', thicken));
    map = double(map);
    map(map == 0) = nan;
    h = imagesc(map);
    set(h, 'AlphaData', ~isnan(map));
    axis off
    title(['Mouse M' num2str(mm)])
    drawnow
end




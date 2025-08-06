clear;clc;close all;
thicken = 2;
Mframe = [110 120 50 10 52];

figure
colormap gray
set(gcf,'position',[15 134 379 328])
load(['~/Dropbox/Two_Photon/M' num2str(1) '.mat'])
movie = movie(:,:,Mframe(1));
movie(movie==0) = nan;
colormap gray
h = imagesc(movie);
set(h, 'AlphaData', ~isnan(movie))
axis off
drawnow

figure
colormap gray
set(gcf,'position',[395 134 375 328])
cc = 0;
for mm = 2:5
    cc = cc + 1;
    subplot(2,2,cc)
    load(['~/Dropbox/Two_Photon/M' num2str(mm) '.mat'])
    movie = movie(:,:,Mframe(mm));
    movie(movie==0) = nan;
    colormap gray
    h = imagesc(movie);
    set(h, 'AlphaData', ~isnan(movie))
    axis off
    drawnow
end

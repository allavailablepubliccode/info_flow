%==========================================================================
% Generate Figure 1: Visualize processed 2-photon imaging maps for 5 mice
%
% This script loads binarized 2-photon maps (from M1 to M5), enhances the
% outlines using morphological operations, and displays them in grayscale.
% The first figure shows Mouse M1 alone. The second figure shows M2–M5
% arranged in a 2x2 grid.
%
%==========================================================================

clear; clc; close all;

% Thickness of outline to render
outline_thickness = 2;

%% Plot Mouse M1 individually
figure;
colormap gray;
set(gcf, 'Position', [15, 134, 379, 328]);

% Load and process map for Mouse M1
load(['~/Dropbox/Two_Photon/M' num2str(1) '.mat']);  % Loads variable `map`
map = padarray(map, [outline_thickness + 2, outline_thickness + 2], 0, 'both');  % Add padding
map = bwmorph(map, 'remove');                % Outline: keep boundaries only
map = imdilate(map, strel('square', outline_thickness));  % Thicken the outline
map = double(map);                           % Convert to double for plotting
map(map == 0) = NaN;                          % Make background transparent

% Plot
h = imagesc(map);
set(h, 'AlphaData', ~isnan(map));            % Use alpha to hide NaNs
axis off;
title('Mouse M1');
drawnow;

%% Plot Mice M2–M5 in a 2x2 subplot
figure;
colormap gray;
set(gcf, 'Position', [395, 134, 375, 328]);

for mouse_idx = 2:5
    subplot_idx = mouse_idx - 1;
    subplot(2, 2, subplot_idx);

    % Load and process map
    load(['~/Dropbox/Two_Photon/M' num2str(mouse_idx) '.mat']);  % Loads variable `map`
    map = padarray(map, [outline_thickness + 2, outline_thickness + 2], 0, 'both');
    map = bwmorph(map, 'remove');
    map = imdilate(map, strel('square', outline_thickness));
    map = double(map);
    map(map == 0) = NaN;

    % Plot
    h = imagesc(map);
    set(h, 'AlphaData', ~isnan(map));
    axis off;
    title(['Mouse M' num2str(mouse_idx)]);
    drawnow;
end

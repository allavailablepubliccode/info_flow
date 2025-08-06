%==========================================================================
% Generate Figure 2: Single-frame visualization of 2-photon movies (M1–M5)
%
% This script loads selected frames from 2-photon movie stacks (M1 to M5),
% masks out background (zero-valued pixels), and displays the activity maps
% for each mouse. M1 is shown alone; M2–M5 are shown in a 2x2 grid.
%
% Author: Erik D. Fagerholm
% Date: 6 August 2025
%==========================================================================

clear; clc; close all;

% Optional thickness parameter (not used here but kept for consistency)
outline_thickness = 2;

% Frame indices to display for each mouse (M1 to M5)
Mframe = [110, 120, 50, 10, 52];

%% Plot Movie Frame for Mouse M1
figure;
colormap gray;
set(gcf, 'Position', [15, 134, 379, 328]);

% Load and display selected frame from Mouse M1
load(['~/Dropbox/Two_Photon/M' num2str(1) '.mat']);  % Loads variable `movie`
frame = movie(:, :, Mframe(1));
frame(frame == 0) = NaN;  % Mask background

% Plot
h = imagesc(frame);
set(h, 'AlphaData', ~isnan(frame));  % Transparent background
axis off;
drawnow;

%% Plot Movie Frames for Mice M2–M5
figure;
colormap gray;
set(gcf, 'Position', [395, 134, 375, 328]);

for mouse_idx = 2:5
    subplot_idx = mouse_idx - 1;
    subplot(2, 2, subplot_idx);

    % Load and extract selected frame
    load(['~/Dropbox/Two_Photon/M' num2str(mouse_idx) '.mat']);  % Loads `movie`
    frame = movie(:, :, Mframe(mouse_idx));
    frame(frame == 0) = NaN;

    % Plot
    h = imagesc(frame);
    set(h, 'AlphaData', ~isnan(frame));
    axis off;
    title(['Mouse M' num2str(mouse_idx)]);
    drawnow;
end

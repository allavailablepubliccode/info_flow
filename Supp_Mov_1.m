%==========================================================================
% Supplementary Movie 1: Visualization of cortical activity across 5 mice
%
% This script:
%   - Loads the first 10 seconds of calcium imaging movies from 5 mice
%   - Masks zero-valued pixels (optional, for clarity)
%   - Displays all movies in parallel subplots
%   - Exports an MPEG-4 video to the desktop
%
% Author: Erik D. Fagerholm
% Date: 6 August 2025
%==========================================================================

clear; clc; close all;

%% === Settings ===
n_mice     = 5;
frame_rate = 13;        % Frames per second (from acquisition)
n_seconds  = 10;        % Duration of movie in seconds
n_frames   = frame_rate * n_seconds;

%% === Load and Prepare Movies ===
movies = cell(1, n_mice);

for mm = 1:n_mice
    load(['~/Dropbox/Two_Photon/M' num2str(mm) '.mat']);  % Loads `movie`
    movie_trimmed = movie(:, :, 1:n_frames);
    movie_trimmed(movie_trimmed == 0) = NaN;              % Optional: hide zeros
    movies{mm} = movie_trimmed;
end

%% === Set Up Video Writer ===
output_path = fullfile(getenv('HOME'), 'Desktop', 'mouse_visual_cortex_movie.mp4');
v = VideoWriter(output_path, 'MPEG-4');
v.FrameRate = frame_rate;
open(v);

%% === Set Up Figure ===
fig = figure('Color', 'w', 'Position', [1, 356, 1024, 181]);

[h, w, ~] = size(movies{1});
axes_handles = gobjects(1, n_mice);
image_handles = gobjects(1, n_mice);

% Create subplots and first frame
for mm = 1:n_mice
    axes_handles(mm) = subplot(1, n_mice, mm);
    frame_data = movies{mm}(:, :, 1);
    image_handles(mm) = imagesc(frame_data, 'Parent', axes_handles(mm));
    colormap(axes_handles(mm), gray);
    axis(axes_handles(mm), 'off');
    set(image_handles(mm), 'AlphaData', ~isnan(frame_data));
end

%% === Animate and Write Video ===
for frame = 1:n_frames
    for mm = 1:n_mice
        frame_data = movies{mm}(:, :, frame);
        set(image_handles(mm), 'CData', frame_data, 'AlphaData', ~isnan(frame_data));
    end
    drawnow;
    frame_captured = getframe(fig);
    writeVideo(v, frame_captured);
end

close(v);
disp(['Movie saved to: ', output_path]);

% generate_fig1.m
% Generate Figure 1: Venus fluorescence image with overlaid manual region masks

clear; clc; close all;

%% === Load data ===
info = load('~/Dropbox/Work/AT_Calcium/20230222_LE_13002_data.mat');
Brain = load('~/Dropbox/Work/AT_Calcium/lobule_position.mat');
[cerebrum_mask, cerebellum_mask] = get_all_region_masks(info);

%% === Create combined brain mask for visualization ===
cerebrum_mask   = double(cerebrum_mask) * 2;   % Encode cerebrum as '2'
cerebellum_mask = double(cerebellum_mask);     % Cerebellum as '1'
shared_mask     = cerebrum_mask + cerebellum_mask;
shared_mask(shared_mask == 0) = NaN;           % Set background to NaN for transparency

%% === Draw three manual regions over the base mask ===
f = figure('Name','Draw 3 regions over the mask', 'Units','normalized', 'OuterPosition',[0 0 1 1]);
imagesc2(shared_mask);
colormap([0 0 0; 0.6 0.6 0.6]);
title('Draw 3 regions over the mask');
hold on;

manual_masks = false([size(shared_mask), 3]);   % 3D binary mask: (rows, cols, region)
boundaries = cell(1, 3);                         % To store polygon boundaries

for i = 1:3
    h = drawfreehand();                          % Manual drawing tool
    manual_masks(:,:,i) = createMask(h);         % Convert to binary mask
    B = bwboundaries(manual_masks(:,:,i));       % Extract region boundary
    boundaries{i} = B{1};
    plot(boundaries{i}(:,2), boundaries{i}(:,1), 'r-', 'LineWidth', 1.5);
end

pause(1);
close(f);

%% === Load Venus image ===
Venus = info.data.Venus_fluorescence_image;

%% === Create final figure with overlays ===
figure;
set(gcf, 'Position', [15 1 348 536]);

% --- Subplot 1: Venus fluorescence with region overlays ---
ax1 = subplot(2,1,1);
imagesc2(Venus);
colormap(ax1, 'gray');
hold on;
for k = 1:3
    plot(boundaries{k}(:,2), boundaries{k}(:,1), 'r-', 'LineWidth', 1.5);
end
title('Venus Fluorescence with Region Overlays');

% --- Subplot 2: Mask overlay showing cerebrum and cerebellum ---
ax2 = subplot(2,1,2);
imagesc2(shared_mask);
colormap(ax2, [0 0 0; 0.6 0.6 0.6]);
hold on;
for k = 1:3
    plot(boundaries{k}(:,2), boundaries{k}(:,1), 'r-', 'LineWidth', 1.5);
end
title('Anatomical Masks with Region Overlays');

%% === Save final figure ===
saveas(gcf, '~/Desktop/Fig1.pdf', 'pdf');

%% === Helper Functions ===

function [cerebrum_mask, cerebellum_mask] = get_all_region_masks(info)
    % Generate binary masks for cerebrum and cerebellum from segmented areas

    sz = size(info.data.Venus_fluorescence_image);
    cerebrum_mask = false(sz);
    cerebellum_mask = false(sz);

    % Aggregate cerebrum regions
    for i = 1:numel(info.data.segment_area.cerebrum)
        for j = 1:numel(info.data.segment_area.cerebrum{i})
            mask = info.data.segment_area.cerebrum{i}{j};
            if ~isempty(mask)
                cerebrum_mask = cerebrum_mask | logical(mask);
            end
        end
    end

    % Aggregate cerebellum regions
    for i = 1:numel(info.data.segment_area.cerebellum)
        for j = 1:numel(info.data.segment_area.cerebellum{i})
            mask = info.data.segment_area.cerebellum{i}{j};
            if ~isempty(mask)
                cerebellum_mask = cerebellum_mask | logical(mask);
            end
        end
    end
end

function h = imagesc2(img_data)
    % Wrapper for imagesc with automatic alpha handling for NaNs
    h = imagesc(img_data);
    axis image off;
    if ndims(img_data) == 2
        set(h, 'AlphaData', ~isnan(img_data));
    elseif ndims(img_data) == 3
        set(h, 'AlphaData', ~isnan(img_data(:,:,1)));
    end
    if nargout < 1
        clear h
    end
end

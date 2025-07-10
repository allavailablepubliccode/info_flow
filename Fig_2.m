% generate_fig2.m
% Generate Figure 2: Smoothed calcium imaging over time for cerebrum and cerebellum

clear; clc; close all;

%% === Load segmentation and anatomical masks ===
info = load('~/Dropbox/Work/AT_Calcium/20230222_LE_13002_data.mat');
[cerebrum_mask, cerebellum_mask] = get_all_region_masks(info);

%% === Load calcium imaging time series ===
frames = 512;
fn = ['~/Dropbox/Work/AT_Calcium/tif/data-001_ratio_' num2str(1) '.tif'];
[rows, cols] = size(imread(fn, 1));

data_cerebellum = zeros(rows, cols, frames);
data_cerebrum   = zeros(rows, cols, frames);

% === Loop over timepoints and extract masked data ===
for tt = 1:frames
    disp(['Processing frame ' num2str(tt)]);

    img = double(imread(fn, tt));

    % --- Background subtraction ---
    flat = img(:);
    bg_thresh = prctile(flat, 1);              % bottom 1% intensity
    bg_mask = img <= bg_thresh;
    background = mean(img(bg_mask));           % estimate background
    img = img - background;

    % --- Apply cerebellum and cerebrum masks separately ---
    cere_img = img;  cere_img(cerebrum_mask == 0) = NaN;
    bell_img = img;  bell_img(cerebellum_mask == 0) = NaN;

    data_cerebrum(:,:,tt)   = cere_img;
    data_cerebellum(:,:,tt) = bell_img;
end

%% === Apply Gaussian smoothing in space and time ===
data_cerebrum   = gaussian_smooth(data_cerebrum, 3, 2);
data_cerebellum = gaussian_smooth(data_cerebellum, 3, 2);

%% === Plot 9 timepoints as image panels ===
figure;
c = 0;
for tt = 20:60:500
    c = c + 1;
    subplot(3, 3, c);

    A = data_cerebellum(:,:,tt);
    B = data_cerebrum(:,:,tt);
    A(isnan(A)) = 0;
    B(isnan(B)) = 0;

    C = A + B;
    C(C == 0) = NaN;

    imagesc2(C);
    title(['Frame ' num2str(tt)]);
end

%% === Helper Functions ===

function [cerebrum_mask, cerebellum_mask] = get_all_region_masks(info)
    % Generate binary masks for cerebrum and cerebellum regions

    sz = size(info.data.Venus_fluorescence_image);
    cerebrum_mask = false(sz);
    cerebellum_mask = false(sz);

    % --- CEREBRUM ---
    for i = 1:numel(info.data.segment_area.cerebrum)
        for j = 1:numel(info.data.segment_area.cerebrum{i})
            mask = info.data.segment_area.cerebrum{i}{j};
            if ~isempty(mask)
                cerebrum_mask = cerebrum_mask | logical(mask);
            end
        end
    end

    % --- CEREBELLUM ---
    for i = 1:numel(info.data.segment_area.cerebellum)
        for j = 1:numel(info.data.segment_area.cerebellum{i})
            mask = info.data.segment_area.cerebellum{i}{j};
            if ~isempty(mask)
                cerebellum_mask = cerebellum_mask | logical(mask);
            end
        end
    end
end

function smoothed = gaussian_smooth(data, sigmaSpace, sigmaTime)
    % Apply spatial and temporal Gaussian smoothing to a 3D [H x W x T] array

    [H, W, T] = size(data);
    smoothed = nan(H, W, T);

    % === Spatial Smoothing ===
    kSizeSpace = 2 * ceil(3 * sigmaSpace) + 1;
    Gspace = fspecial('gaussian', [kSizeSpace, kSizeSpace], sigmaSpace);

    spatialSmoothed = nan(H, W, T);

    for t = 1:T
        frame = data(:,:,t);
        mask = ~isnan(frame);
        frame(~mask) = 0;

        convFrame = conv2(frame, Gspace, 'same');
        convMask  = conv2(double(mask), Gspace, 'same');

        normFrame = convFrame ./ convMask;
        normFrame(convMask == 0) = NaN;

        spatialSmoothed(:,:,t) = normFrame;
    end

    % === Temporal Smoothing ===
    kSizeTime = 2 * ceil(3 * sigmaTime) + 1;
    halfWin = floor(kSizeTime / 2);
    Gtime = fspecial('gaussian', [1, kSizeTime], sigmaTime);
    Gtime = Gtime / sum(Gtime);

    padded = padarray(spatialSmoothed, [0, 0, halfWin], NaN, 'both');

    for t = 1:T
        window = padded(:,:,t:t+kSizeTime-1);
        weighted = bsxfun(@times, window, reshape(Gtime, 1, 1, []));
        valid = ~isnan(window);

        weightSum = sum(bsxfun(@times, valid, reshape(Gtime, 1, 1, [])), 3);
        frameSum = nansum(weighted, 3);

        normFrame = frameSum ./ weightSum;
        normFrame(weightSum == 0) = NaN;

        smoothed(:,:,t) = normFrame;
    end
end

function h = imagesc2(img_data)
    % Wrapper for imagesc that supports NaN transparency
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

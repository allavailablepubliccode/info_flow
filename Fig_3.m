% generate_fig3.m
% Generate Figure 3: Time series of cerebellum and cerebrum activity

clear; clc; close all;

%% === Load segmentation mask info ===
info = load('~/Dropbox/Work/AT_Calcium/20230222_LE_13002_data.mat');
[cerebrum_mask, cerebellum_mask] = get_all_region_masks(info);
Fs = info.data.sampling_frequency;

numfiles = 28;

%% === Process and save extracted signals from each recording ===
for ii = 1:numfiles
    disp(['Processing file ' num2str(ii) ' of ' num2str(numfiles)]);

    % Load .tif image sequence
    fn = ['~/Dropbox/Work/AT_Calcium/tif/data-001_ratio_' num2str(ii) '.tif'];
    frames = count_framenum(fn);
    [rows, cols] = size(imread(fn, 1));

    data_cerebellum = zeros(rows, cols, frames);
    data_cerebrum   = zeros(rows, cols, frames);

    % === Loop through timepoints ===
    for tt = 1:frames
        img = double(imread(fn, tt));

        % Background subtraction using lowest 1% intensity
        flat = img(:);
        bg_thresh = prctile(flat, 1);
        bg_mask = img <= bg_thresh;
        background = mean(img(bg_mask));
        img = img - background;

        % Apply cerebrum and cerebellum masks
        bell_img = img;  bell_img(cerebellum_mask == 0) = NaN;
        cere_img = img;  cere_img(cerebrum_mask == 0)   = NaN;

        data_cerebellum(:,:,tt) = bell_img;
        data_cerebrum(:,:,tt)   = cere_img;
    end

    % Crop data to mask region
    data_cerebellum = MaskData(data_cerebellum, cerebellum_mask);
    data_cerebrum   = MaskData(data_cerebrum, cerebrum_mask);

    % Smooth in space and time
    data_cerebellum = gaussian_smooth(data_cerebellum, 3, 2);
    data_cerebrum   = gaussian_smooth(data_cerebrum, 3, 2);

    % Average over space to get 1D time series
    data_cerebellum = squeeze(nanmean(nanmean(data_cerebellum, 1), 2));
    data_cerebrum   = squeeze(nanmean(nanmean(data_cerebrum, 1), 2));

    % Save extracted time series
    save(['~/Dropbox/Work/AT_Calcium/mat/data-001_ratio_' num2str(ii) '.mat'], ...
        'data_cerebellum', 'data_cerebrum');
end

%% === Concatenate and plot combined time series across all sessions ===
cat_cerebellum = [];
cat_cerebrum   = [];

for ii = 1:numfiles
    load(['~/Dropbox/Work/AT_Calcium/mat/data-001_ratio_' num2str(ii) '.mat'], ...
        'data_cerebellum', 'data_cerebrum');
    cat_cerebellum = [cat_cerebellum; data_cerebellum];
    cat_cerebrum   = [cat_cerebrum; data_cerebrum];
end

tt = (1:numel(cat_cerebellum)) / Fs;

% === Plot ===
figure;
plot(tt, cat_cerebellum, 'k'); hold on;
plot(tt, cat_cerebrum,   'r');
xlabel('Time (s)');
ylabel('Signal');
legend({'Cerebellum', 'Cerebrum'});
title('Concatenated Neural Activity Across Sessions');
hold off;

%% === Helper Functions ===

function N = count_framenum(tifpath)
    % Count number of frames in a .tif stack
    N = 0;
    try
        while true
            N = N + 1;
            imread(tifpath, N);
        end
    catch
        N = N - 1;
    end
end

function [cerebrum_mask, cerebellum_mask] = get_all_region_masks(info)
    % Extract binary masks for cerebrum and cerebellum

    sz = size(info.data.Venus_fluorescence_image);
    cerebrum_mask = false(sz);
    cerebellum_mask = false(sz);

    for i = 1:numel(info.data.segment_area.cerebrum)
        for j = 1:numel(info.data.segment_area.cerebrum{i})
            mask = info.data.segment_area.cerebrum{i}{j};
            if ~isempty(mask)
                cerebrum_mask = cerebrum_mask | logical(mask);
            end
        end
    end

    for i = 1:numel(info.data.segment_area.cerebellum)
        for j = 1:numel(info.data.segment_area.cerebellum{i})
            mask = info.data.segment_area.cerebellum{i}{j};
            if ~isempty(mask)
                cerebellum_mask = cerebellum_mask | logical(mask);
            end
        end
    end
end

function masked = MaskData(data, mask)
    % Apply binary mask to 2D or 3D data and crop to bounding box

    mask = double(mask);
    [rowIdx, colIdx] = find(mask);
    if isempty(rowIdx)
        error('Mask is empty.');
    end

    rowMin = min(rowIdx); rowMax = max(rowIdx);
    colMin = min(colIdx); colMax = max(colIdx);

    if ndims(data) == 3
        cropped = data(rowMin:rowMax, colMin:colMax, :);
        mask_crop = mask(rowMin:rowMax, colMin:colMax);
        mask3D = repmat(mask_crop, 1, 1, size(cropped, 3));
        masked = cropped;
        masked(~mask3D) = NaN;
    else
        cropped = data(rowMin:rowMax, colMin:colMax);
        mask_crop = mask(rowMin:rowMax, colMin:colMax);
        masked = cropped;
        masked(~mask_crop) = NaN;
    end
end

function smoothed = gaussian_smooth(data, sigmaSpace, sigmaTime)
    % Apply separable Gaussian smoothing in space and time

    [H, W, T] = size(data);
    smoothed = nan(H, W, T);

    % === Spatial smoothing ===
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

    % === Temporal smoothing ===
    kSizeTime = 2 * ceil(3 * sigmaTime) + 1;
    halfWin = floor(kSizeTime / 2);
    Gtime = fspecial('gaussian', [1, kSizeTime], sigmaTime);
    Gtime = Gtime / sum(Gtime);

    padded = padarray(spatialSmoothed, [0 0 halfWin], NaN, 'both');

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

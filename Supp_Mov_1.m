% Supplementary Movie 1: Overlay of Cerebellum and Cerebrum Calcium Signals
% Outputs PNG frames with alpha, assembles them into white-background AVI

clear; clc; close all;

% === Settings ===
info = load('~/Dropbox/Work/AT_Calcium/20230222_LE_13002_data.mat');
[cerebrum_mask, cerebellum_mask] = get_all_region_masks(info);
shared_mask = cerebrum_mask | cerebellum_mask;

frames = 512;
numfiles = 2;
output_dir = '~/Desktop/frames_transparent/';
mkdir(output_dir);

fps = info.data.sampling_frequency;
sigmaSpace = 3;      % Spatial Gaussian kernel
sigmaTime  = 2;      % Temporal Gaussian smoothing
frame_counter = 0;
target_size = [];    % To enforce uniform frame dimensions

for ii = 1:numfiles
    disp(['Processing file ' num2str(ii) '/' num2str(numfiles)]);
    fn = ['~/Dropbox/Work/AT_Calcium/tif/data-001_ratio_' num2str(ii) '.tif'];
    [rows, cols] = size(imread(fn, 1));

    % Preallocate
    data_cerebellum = zeros(rows, cols, frames);
    data_cerebrum   = zeros(rows, cols, frames);

    % === Load and background-subtract all frames ===
    for tt = 1:frames
        tmp = double(imread(fn, tt));
        bg_thresh = prctile(tmp(:), 1);
        background = mean(tmp(tmp <= bg_thresh));
        tmp = tmp - background;

        im_cerebellum = tmp; im_cerebellum(cerebellum_mask == 0) = NaN;
        im_cerebrum   = tmp; im_cerebrum(cerebrum_mask == 0) = NaN;

        data_cerebellum(:,:,tt) = im_cerebellum;
        data_cerebrum(:,:,tt)   = im_cerebrum;
    end

    % Mask and smooth
    data_cerebellum(~repmat(cerebellum_mask,1,1,frames)) = NaN;
    data_cerebrum(~repmat(cerebrum_mask,1,1,frames)) = NaN;

    data_cerebellum = gaussian_smooth(data_cerebellum, sigmaSpace, sigmaTime);
    data_cerebrum   = gaussian_smooth(data_cerebrum, sigmaSpace, sigmaTime);

    [H, W, T] = size(data_cerebrum);
    if isempty(target_size), target_size = [H, W]; end

    cmap = parula(256);  % Colormap

    % === Save per-frame PNGs ===
    for t = 1:T
        norm_cerebellum = normalize_intensity(data_cerebellum(:,:,t));
        norm_cerebrum   = normalize_intensity(data_cerebrum(:,:,t));

        norm_cerebellum(isnan(norm_cerebellum)) = 0;
        norm_cerebrum(isnan(norm_cerebrum)) = 0;

        composite = NaN(H, W);
        composite(cerebellum_mask) = norm_cerebellum(cerebellum_mask);
        composite(cerebrum_mask) = norm_cerebrum(cerebrum_mask);

        idx = round(composite * 255) + 1;
        idx(isnan(idx)) = 1;
        RGB = reshape(cmap(idx, :), [H, W, 3]);

        alpha = double(~isnan(composite));

        % Pad to match target size
        padH = max(0, target_size(1) - size(RGB,1));
        padW = max(0, target_size(2) - size(RGB,2));
        RGB = padarray(RGB, [padH, padW], 1, 'post');
        alpha = padarray(alpha, [padH, padW], 0, 'post');

        frame_counter = frame_counter + 1;
        imwrite(RGB, fullfile(output_dir, sprintf('frame_%04d.png', frame_counter)), 'Alpha', alpha);
    end
end

disp('✅ PNGs written to:');
disp(output_dir);

% === Assemble PNG frames into AVI ===
output_video = '~/Desktop/CalciumOverlay_whiteBG.avi';
png_files = dir(fullfile(output_dir, 'frame_*.png'));
[~, sort_idx] = sort({png_files.name});
png_files = png_files(sort_idx);

v = VideoWriter(output_video);
v.FrameRate = fps;
open(v);

% Use first frame to fix size
first_img = imread(fullfile(output_dir, png_files(1).name));
target_size = size(first_img, 1:2);

for i = 1:length(png_files)
    [img, ~, alpha] = imread(fullfile(output_dir, png_files(i).name));
    img = im2double(img);
    alpha = im2double(alpha);

    padH = max(0, target_size(1) - size(img,1));
    padW = max(0, target_size(2) - size(img,2));

    img = padarray(img, [padH, padW], 1, 'post');
    alpha = padarray(alpha, [padH, padW], 0, 'post');

    img = img(1:target_size(1), 1:target_size(2), :);
    alpha = alpha(1:target_size(1), 1:target_size(2));

    % Composite over white
    white_bg = ones(size(img));
    alpha3 = repmat(alpha, 1, 1, 3);
    blended = alpha3 .* img + (1 - alpha3) .* white_bg;

    writeVideo(v, blended);
end

close(v);
disp(['✅ Video written to: ' output_video]);


% === Helper: Normalize to [0,1], ignore NaNs ===
function out = normalize_intensity(im)
    im = double(im);
    im(isnan(im)) = NaN;
    minv = nanmin(im(:));
    maxv = nanmax(im(:));
    if maxv > minv
        out = (im - minv) / (maxv - minv);
    else
        out = zeros(size(im));
    end
end

% === Helper: Get cerebrum/cerebellum masks ===
function [cerebrum_mask, cerebellum_mask] = get_all_region_masks(info)
    sz = size(info.data.Venus_fluorescence_image);
    cerebrum_mask = false(sz); cerebellum_mask = false(sz);
    for i = 1:numel(info.data.segment_area.cerebrum)
        for j = 1:numel(info.data.segment_area.cerebrum{i})
            mask = info.data.segment_area.cerebrum{i}{j};
            if ~isempty(mask), cerebrum_mask = cerebrum_mask | logical(mask); end
        end
    end
    for i = 1:numel(info.data.segment_area.cerebellum)
        for j = 1:numel(info.data.segment_area.cerebellum{i})
            mask = info.data.segment_area.cerebellum{i}{j};
            if ~isempty(mask), cerebellum_mask = cerebellum_mask | logical(mask); end
        end
    end
end

% === Helper: Apply 3D Gaussian smoothing with NaN support ===
function smoothed = gaussian_smooth(data, sigmaSpace, sigmaTime)
    [H, W, T] = size(data);
    smoothed = nan(H, W, T);

    % Spatial filter
    kSizeSpace = 2 * ceil(3*sigmaSpace) + 1;
    Gspace = fspecial('gaussian', [kSizeSpace, kSizeSpace], sigmaSpace);
    spatial = nan(H, W, T);
    for t = 1:T
        frame = data(:,:,t);
        if all(isnan(frame), 'all'), continue; end
        mask = ~isnan(frame);
        frame(~mask) = 0;
        convFrame = conv2(frame, Gspace, 'same');
        convMask  = conv2(double(mask), Gspace, 'same');
        normFrame = convFrame ./ convMask;
        normFrame(convMask == 0) = NaN;
        spatial(:,:,t) = normFrame;
    end

    % Temporal filter
    kSizeTime = 2 * ceil(3*sigmaTime) + 1;
    halfWin = floor(kSizeTime / 2);
    Gtime = fspecial('gaussian', [1, kSizeTime], sigmaTime);
    Gtime = Gtime / sum(Gtime);

    padded = padarray(spatial, [0, 0, halfWin], NaN, 'both');
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

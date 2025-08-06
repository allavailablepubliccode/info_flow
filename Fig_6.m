%==========================================================================
% Generate Figure 6: Significant predictive connectivity across regions
%
% This script:
%   - Loads data for M1–M5 and extracts PC1 (or mean) for each of 6 regions
%   - Computes predictive strength (R²) using a power-law KDE model
%   - Performs circular permutation test to get null distribution of R²
%   - Applies FDR correction to obtain significant links
%   - Plots R² matrices for each mouse (only significant and high values)
%
% Author: [Your Name]
% Date: [Insert Date]
%==========================================================================

clear; clc; close all;

%% === Parameters ===
N = 1;                        % Lag (in timepoints)
nSurrogates = 1000;           % Permutation samples per pair
alpha_FDR = 0.05;             % FDR threshold
use_pc1 = true;               % Use PC1 instead of region average
nMice = 5; nRegions = 6;

% Initialize storage
R2full     = nan(nMice, nRegions, nRegions);
pvals      = nan(nMice, nRegions, nRegions);
alpha_mat  = nan(nMice, nRegions, nRegions);
beta_mat   = nan(nMice, nRegions, nRegions);

%% === Main Loop: across mice and region pairs ===
for ii = 1:nMice
    fprintf('Processing Mouse %d...\n', ii);
    
    % Load and reshape
    clear movie map
    load(['~/Dropbox/Two_Photon/M' num2str(ii) '.mat']);
    movie = reshape(movie, size(movie,1)*size(movie,2), size(movie,3));
    map   = reshape(map,   size(map,1)*size(map,2), 1);

    for jj = 1:nRegions  % Source region
        First_idx = find(map == jj);
        if isempty(First_idx), continue; end

        First_mat = movie(First_idx, :);
        First = extract_region_signal(First_mat, use_pc1);
        First = First(N:end);  % Apply lag

        for kk = 1:nRegions  % Target region
            if jj == kk, continue; end

            Second_idx = find(map == kk);
            if isempty(Second_idx), continue; end

            Second_mat = movie(Second_idx, :);
            Second = extract_region_signal(Second_mat, use_pc1);
            Second = Second(1:end-N+1);

            if numel(First) < 10 || numel(Second) < 10, continue; end

            [R2, ~, alpha, beta] = plaw_kde(First(:), Second(:));
            if R2 < 0
                R2full(ii,jj,kk) = NaN;
                continue;
            end

            R2full(ii,jj,kk)     = R2;
            alpha_mat(ii,jj,kk)  = alpha;
            beta_mat(ii,jj,kk)   = beta;

            % Permutation test (circular shift)
            surrogate_R2s = nan(nSurrogates, 1);
            for s = 1:nSurrogates
                shift = randi(numel(First));
                First_shifted = circshift(First, shift);
                surrogate_R2s(s) = plaw_kde(First_shifted(:), Second(:));
            end

            pvals(ii,jj,kk) = mean(surrogate_R2s >= R2);
        end
    end
end

%% === FDR Correction ===
p_flat = pvals(:);
valid = ~isnan(p_flat);

[p_fdr, passed] = fdr_bh(p_flat(valid), alpha_FDR, 'pdep', 'yes');
sig_mask = false(size(p_flat));
sig_mask(valid) = passed;
sig_mask = reshape(sig_mask, size(R2full));

% Mask R² values: only significant and > 0.7 shown
R2masked = R2full;
R2masked(~sig_mask) = NaN;
R2masked(R2masked < 0.7) = NaN;

%% === Plot Significant R² Matrices for Each Mouse ===
figure('Color', 'w'); set(gcf, 'Position', [1, 378, 1024, 159]);
for ii = 1:nMice
    subplot(1, nMice, ii);
    tmp = squeeze(R2masked(ii,:,:));
    h = imagesc(tmp);
    set(h, 'AlphaData', ~isnan(tmp));
    title(['Mouse M' num2str(ii)]);
    colormap gray;
    clim([min(R2masked(:)), max(R2masked(:))]);
end

% Separate colorbar
figure('Color','w');
colormap gray;
clim([min(R2masked(:)), max(R2masked(:))]);
colorbar;
sgtitle('Significant R² values (FDR-corrected)');

%% === Summary Metrics: AM Input Consistency ===
fprintf('\nNumber of significant connections after FDR:\n');
disp(sum(sig_mask(:)));

AM_idx = 2;  % Anteromedial target
labels = {'A','M','L','P','R','V'};
group_R2_to_AM = nan(nMice, nRegions);

for i = 1:nMice
    for j = 1:nRegions
        if j == AM_idx, continue; end
        group_R2_to_AM(i,j) = R2masked(i,j,AM_idx);
    end
end

% Average R² and presence of significant input across mice
mean_R2_to_AM = nanmean(group_R2_to_AM, 1);
presence_mask = ~isnan(group_R2_to_AM);
consistency_to_AM = sum(presence_mask, 1) / nMice;

%% === Subfunctions ===

function sig = extract_region_signal(region_mat, use_pc1)
    % Returns either first PC or average time series for a region
    if use_pc1
        sig = pca_first_component(region_mat);
    else
        sig = mean(region_mat, 1)';
    end
end

function pc1 = pca_first_component(data_mat)
    % First principal component of a region (time series)
    data_mat = detrend(data_mat')';  % Optional detrending
    [~, score, ~] = pca(data_mat');
    pc1 = score(:,1);
end

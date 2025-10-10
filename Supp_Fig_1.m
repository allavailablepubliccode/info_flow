%==========================================================================
% Generate Supplementary Figure 1: Significant predictive connectivity across regions
%
% This script:
%   - Loads data for M1–M5 and extracts PC1 (or mean) for each of 6 regions
%   - Computes predictive strength (R²) using a power-law KDE model
%   - Performs circular permutation test to get null distribution of R²
%   - Applies Bonferroni correction to obtain significant links
%   - Plots R² matrices for each mouse (only significant values)
%
%==========================================================================

clear; clc; close all; rng(0);

%% === Parameters ===
N = 1;                        % Lag (in timepoints)
nSurrogates = 1000;           % Permutation samples per pair
alpha_FDR = 0.01;             % FDR threshold
use_pc1 = true;               % Use PC1 instead of region average
nMice = 5; nRegions = 6;

% Uncomment below to recompute (already saved in Fig6results.mat)
% % Initialize storage
% R2full     = nan(nMice, nRegions, nRegions);
% pvals      = nan(nMice, nRegions, nRegions);
% alpha_mat  = nan(nMice, nRegions, nRegions);
% beta_mat   = nan(nMice, nRegions, nRegions);
%
% for ii = 1:nMice
%     fprintf('Processing Mouse %d...\n', ii);
%     clear movie map
%     load(['~/Dropbox/Two_Photon/M' num2str(ii) '.mat']);
%     movie = reshape(movie, size(movie,1)*size(movie,2), size(movie,3));
%     map   = reshape(map,   size(map,1)*size(map,2), 1);
%
%     for jj = 1:nRegions
%         First_idx = find(map == jj);
%         if isempty(First_idx), continue; end
%         First_mat = movie(First_idx, :);
%         First = extract_region_signal(First_mat, use_pc1);
%         First = First(N:end);
%
%         for kk = 1:nRegions
%             if jj == kk, continue; end
%             Second_idx = find(map == kk);
%             if isempty(Second_idx), continue; end
%             Second_mat = movie(Second_idx, :);
%             Second = extract_region_signal(Second_mat, use_pc1);
%             Second = Second(1:end-N+1);
%             if numel(First) < 10 || numel(Second) < 10, continue; end
%
%             [R2, ~, alpha, beta] = plaw_kde(First(:), Second(:));
%             if R2 < 0
%                 R2full(ii,jj,kk) = NaN;
%                 continue;
%             end
%             R2full(ii,jj,kk)     = R2;
%             alpha_mat(ii,jj,kk)  = alpha;
%             beta_mat(ii,jj,kk)   = beta;
%
%             surrogate_R2s = nan(nSurrogates, 1);
%             for s = 1:nSurrogates
%                 shift = randi(numel(First));
%                 First_shifted = circshift(First, shift);
%                 surrogate_R2s(s) = plaw_kde(First_shifted(:), Second(:));
%             end
%             pvals(ii,jj,kk) = mean(surrogate_R2s >= R2);
%         end
%     end
% end
% save('Fig6results.mat','R2full','pvals','alpha_mat','beta_mat');

load('Fig6results.mat','R2full','pvals','alpha_mat','beta_mat');
%% === Bonferroni Correction (off-diagonal only) ===

alpha_bonf = 0.01;  % overall alpha level

% Create mask for diagonal entries (self-connections)
diag_mask = false(size(pvals));
for m = 1:nMice
    diag_mask(m,:,:) = eye(nRegions);
end

% Flatten p-values (exclude diagonal and NaNs)
p_flat = pvals(:);
valid_idx = find(~diag_mask(:) & ~isnan(p_flat));
n_tests = numel(valid_idx);

% Apply Bonferroni correction: threshold = alpha / number of tests
alpha_corrected = alpha_bonf / n_tests;
sig_mask_bonf = false(size(pvals));
sig_mask_bonf(valid_idx) = p_flat(valid_idx) < alpha_corrected;

% Mask R² values
R2masked_bonf = R2full;
R2masked_bonf(~sig_mask_bonf) = NaN;

%% === Plot Bonferroni-Significant R² Matrices for Each Mouse ===
figure('Color','w');
set(gcf,'Position',[1, 378, 1024, 159]);
for ii = 1:nMice
    subplot(1, nMice, ii);
    tmp  = squeeze(R2masked_bonf(ii,:,:));
    mask = squeeze(sig_mask_bonf(ii,:,:));

    tmp(eye(nRegions)==1) = NaN;
    mask(eye(nRegions)==1) = false;

    h = imagesc(tmp);
    set(h,'AlphaData',~isnan(tmp));
    axis square;
    colorbar
    colormap(flipud(gray));
    clim([0.1350, 0.9024]);
    title(['Mouse M' num2str(ii)]);

    % Overlay 'X' on non-significant off-diagonals
    for r = 1:nRegions
        for c = 1:nRegions
            if r == c, continue; end
            if ~mask(r,c)
                text(c,r,'X','Color',[0.6 0 0],'FontSize',10, ...
                    'FontWeight','bold','HorizontalAlignment','center');
            end
        end
    end
end
sgtitle('Bonferroni-Corrected Significant Connections (q = 0.01)');

%% === Summary ===
n_sig_bonf = nnz(sig_mask_bonf(~diag_mask & ~isnan(pvals)));
fprintf('Bonferroni correction (α = %.3f, %d tests)\n', alpha_bonf, n_tests);
fprintf('Bonferroni-corrected threshold = %.3e\n', alpha_corrected);
fprintf('Number of surviving pairs: %d\n', n_sig_bonf);


%% === Subfunctions ===
function sig = extract_region_signal(region_mat, use_pc1)
if use_pc1
    sig = pca_first_component(region_mat);
else
    sig = mean(region_mat, 1)';
end
end

function pc1 = pca_first_component(data_mat)
data_mat = detrend(data_mat')';
[~, score, ~] = pca(data_mat');
pc1 = score(:,1);
end

%==========================================================================
% Generate Figure 5: Significant predictive connectivity across regions
%
% This script:
%   - Loads data for M1–M5 and extracts PC1 (or mean) for each of 6 regions
%   - Computes predictive strength (R²) using a power-law KDE model
%   - Performs circular permutation test to get null distribution of R²
%   - Applies FDR correction to obtain significant links
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

%% === FDR Correction (off-diagonal only) ===

% Create mask for diagonal entries (self-connections)
diag_mask = false(size(pvals));
for m = 1:nMice
    diag_mask(m,:,:) = eye(nRegions);
end

% Flatten p-values and masks
p_flat = pvals(:);
diag_mask_flat = diag_mask(:);
valid_idx = find(~diag_mask_flat & ~isnan(p_flat));  % Off-diagonal, non-NaN

% Apply FDR to only off-diagonal valid p-values
[p_fdr, passed] = fdr_bh(p_flat(valid_idx), alpha_FDR, 'pdep', 'yes');

% Build full-size significance mask
sig_mask = false(size(pvals));
sig_mask(valid_idx) = passed;

% Apply significance mask to R²
R2masked = R2full;
R2masked(~sig_mask) = NaN;
R2masked(R2masked<0.55) = NaN;

%% === Plot FDR-Significant R² Matrices for Each Mouse ===
figure('Color', 'w'); set(gcf, 'Position', [1, 378, 1024, 159]);
for ii = 1:nMice
    subplot(1, nMice, ii);
    tmp  = squeeze(R2masked(ii,:,:));
    mask = squeeze(sig_mask(ii,:,:));

    % Remove diagonal from plot and mask
    tmp(eye(nRegions)==1) = NaN;
    mask(eye(nRegions)==1) = false;

    h = imagesc(tmp);
    set(h, 'AlphaData', ~isnan(tmp));  % Transparent where NaN
    axis square;
    colormap(flipud(gray));  % makes white = low, black = high
    clim([0.1350, 0.9024]);
    title(['Mouse M' num2str(ii)]);
    colorbar
    
    % Overlay 'X' on non-significant off-diagonals
    for r = 1:nRegions
        for c = 1:nRegions
            if r == c, continue; end
            if ~mask(r,c)
                text(c, r, 'X', 'Color', [0.6 0 0], 'FontSize', 10, ...
                    'FontWeight', 'bold', 'HorizontalAlignment', 'center');
            end
        end
    end
end
sgtitle('Only FDR-Significant Connections Shown');

%% === Summary: FDR-Passing Connections (Excluding Diagonal) ===
n_total = nRegions * (nRegions - 1) * nMice;
n_sig = nnz(sig_mask(~diag_mask & ~isnan(pvals)));

fprintf('Out of %d total region pairs × %d mice = %d off-diagonal tests:\n', ...
    nRegions*(nRegions-1), nMice, n_total);
fprintf('FDR (q = %.3f) survivors (excluding diagonal): %d\n\n', ...
    alpha_FDR, n_sig);

%% === Hold-Out Validation to Assess Generalization ===
fprintf('\n=== Hold-Out Validation ===\n');

% R2_holdout = nan(nMice, nRegions, nRegions);
% for ii = 1:nMice
%     fprintf('Validating Mouse %d...\n', ii);
%     clear movie map
%     load(['~/Dropbox/Two_Photon/M' num2str(ii) '.mat']);
%     movie = reshape(movie, size(movie,1)*size(movie,2), size(movie,3));
%     map   = reshape(map,   size(map,1)*size(map,2), 1);
%     nT = size(movie,2);
%     splitIdx = round(nT * 0.8);
% 
%     for jj = 1:nRegions
%         idxA = find(map == jj);
%         if isempty(idxA), continue; end
%         A = extract_region_signal(movie(idxA,:), use_pc1);
% 
%         for kk = 1:nRegions
%             if jj == kk, continue; end
%             idxB = find(map == kk);
%             if isempty(idxB), continue; end
%             B = extract_region_signal(movie(idxB,:), use_pc1);
% 
%             % Skip if alpha/beta missing
%             if isnan(alpha_mat(ii,jj,kk)) || isnan(beta_mat(ii,jj,kk))
%                 continue;
%             end
% 
%             % Training = first 80%, Testing = last 20%
%             A_train = A(1:splitIdx);
%             B_train = B(1:splitIdx);
%             A_test  = A(splitIdx+1:end);
%             B_test  = B(splitIdx+1:end);
% 
%             % Compute KDE for test segment using training pdf
%             [pdfA, xi] = ksdensity(A_train, 'Function', 'pdf');
%             qA = interp1(xi, pdfA, A_test, 'linear', 'extrap');
%             qA(qA <= 0) = eps; % avoid log(0)
% 
%             % Apply transformation using fitted alpha/beta
%             alpha = alpha_mat(ii,jj,kk);
%             beta  = beta_mat(ii,jj,kk);
%             B_est = A_test + alpha * (log(qA) - mean(log(qA))) + ...
%                     beta  * (A_test - mean(A_test));
% 
%             % Compute hold-out R²
%             R2_holdout(ii,jj,kk) = 1 - sum((B_test - B_est).^2) / sum((B_test - mean(B_test)).^2);
%         end
%     end
% end
% 
% % Compare hold-out vs. original fit
% valid_idx = ~isnan(R2full(:)) & ~isnan(R2_holdout(:));
% deltaR2 = R2full(valid_idx) - R2_holdout(valid_idx);
% fprintf('Mean ΔR² (train - holdout) = %.4f ± %.4f (SD)\n', mean(deltaR2), std(deltaR2));
% 
% % Optional: save
% save('HoldoutValidation.mat','R2_holdout','deltaR2');
load('HoldoutValidation.mat','R2_holdout','deltaR2');

% %% === Randomized Cross-Validation (20 random splits per pair) ===
% fprintf('\n=== Randomized Cross-Validation (20× 80/20 splits) ===\n');
% 
% nSplits = 20;
% R2_train_cv = [];
% R2_test_cv  = [];
% 
% for ii = 1:nMice
%     fprintf('Mouse %d...\n', ii);
%     clear movie map
%     load(['~/Dropbox/Two_Photon/M' num2str(ii) '.mat']);
%     movie = reshape(movie, size(movie,1)*size(movie,2), size(movie,3));
%     map   = reshape(map,   size(map,1)*size(map,2), 1);
% 
%     for jj = 1:nRegions
%         idxA = find(map == jj);
%         if isempty(idxA), continue; end
%         A = extract_region_signal(movie(idxA,:), use_pc1);
% 
%         for kk = 1:nRegions
%             if jj == kk, continue; end
%             idxB = find(map == kk);
%             if isempty(idxB), continue; end
%             B = extract_region_signal(movie(idxB,:), use_pc1);
% 
%             % Skip invalid entries
%             if isnan(alpha_mat(ii,jj,kk)) || isnan(beta_mat(ii,jj,kk))
%                 continue;
%             end
% 
%             alpha = alpha_mat(ii,jj,kk);
%             beta  = beta_mat(ii,jj,kk);
%             nT = numel(A);
% 
%             for s = 1:nSplits
%                 % Random 80/20 split
%                 idx_train = randperm(nT, round(0.8*nT));
%                 idx_test  = setdiff(1:nT, idx_train);
% 
%                 A_train = A(idx_train);
%                 B_train = B(idx_train);
%                 A_test  = A(idx_test);
%                 B_test  = B(idx_test);
% 
%                 % KDE on training data
%                 [pdfA, xi] = ksdensity(A_train, 'Function', 'pdf');
%                 qA_test = interp1(xi, pdfA, A_test, 'linear', 'extrap');
%                 qA_test(qA_test <= 0) = eps;
% 
%                 % Predict test data
%                 B_est_test = A_test + alpha*(log(qA_test)-mean(log(qA_test))) + ...
%                              beta*(A_test-mean(A_test));
% 
%                 % Compute R² for test and train
%                 R2_test = 1 - sum((B_test - B_est_test).^2) / sum((B_test - mean(B_test)).^2);
% 
%                 qA_train = interp1(xi, pdfA, A_train, 'linear', 'extrap');
%                 qA_train(qA_train <= 0) = eps;
%                 B_est_train = A_train + alpha*(log(qA_train)-mean(log(qA_train))) + ...
%                               beta*(A_train-mean(A_train));
%                 R2_train = 1 - sum((B_train - B_est_train).^2) / sum((B_train - mean(B_train)).^2);
% 
%                 R2_train_cv(end+1) = R2_train;
%                 R2_test_cv(end+1)  = R2_test;
%             end
%         end
%     end
% end
% 
% % Optional: save
% save('CrossVal_Randomized.mat','R2_train_cv','R2_test_cv');
load('CrossVal_Randomized.mat','R2_train_cv','R2_test_cv');

% Compute statistics
deltaR2_cv = R2_train_cv - R2_test_cv;
nSamples = numel(deltaR2_cv);
meanDelta = mean(deltaR2_cv);
sdDelta = std(deltaR2_cv);
semDelta = sdDelta / sqrt(nSamples);

fprintf('Mean ΔR² (train - test) = %.4f ± %.4f (SEM) over %d random splits\n', ...
    meanDelta, semDelta, nSamples);

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

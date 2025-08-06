%==========================================================================
% Generate Figure 5: Predicting AM from V1 via power-law transformation
%
% This script:
%   - Loads calcium imaging data for one mouse
%   - Extracts V1 and AM region signals (average or PC1)
%   - Fits a nonlinear transformation model (plaw_kde) from V1 to AM
%   - Normalizes and plots the true vs. predicted AM signals
%   - Performs a circular permutation test to assess significance
%
% Author: Erik D. Fagerholm
% Date: 6 August 2025
%==========================================================================

clear; clc; close all;

%% === Parameters ===
mouse_id = 3;
use_pc1 = true;         % Whether to use first PC instead of average signal
n_iter = 1000;          % Permutation iterations for significance test

%% === Load Data ===
load(['~/Dropbox/Two_Photon/M' num2str(mouse_id) '.mat']);  % Loads `movie`, `map`

% Reshape to 2D: [pixels x time]
movie = reshape(movie, size(movie,1)*size(movie,2), size(movie,3));
map   = reshape(map,   size(map,1)*size(map,2), 1);

% Extract indices for brain regions
V1_idx = find(map == 6);
AM_idx = find(map == 2);

% Extract time series data
V1_mat = movie(V1_idx, :);
AM_mat = movie(AM_idx, :);

% === Extract region activity signals ===
if use_pc1
    V1 = pca_first_component(V1_mat);  % More robust to noise
    AM  = pca_first_component(AM_mat);
else
    V1 = mean(V1_mat)';  % Mean across pixels
    AM  = mean(AM_mat)';
end

%% === Fit nonlinear model: AM = plaw_kde(V1) ===
[R2_true, AM_est, alpha, beta] = plaw_kde(V1, AM);

% Normalize both signals to [0,1] for plotting
AM_norm     = normalize_to_unit_range(AM, AM_est);
AM_est_norm = normalize_to_unit_range(AM_est, AM);

% Time axis
Fs = numel(V1) / 5 / 60;  % Estimate sampling rate (Hz) from 5 min recording
t = (1:numel(V1)) / Fs;

%% === Plot predicted vs true AM signals ===
figure('Color','w'); set(gcf, 'Position', [0, 284, 1025, 253]);
plot(t(1:109), AM_norm(1:109), 'k', 'LineWidth', 1.2); hold on;
plot(t(1:109), AM_est_norm(1:109), 'r', 'LineWidth', 1.2);
legend('AM (true)', 'AM (estimated)', 'Location', 'southeast');
xlabel('Time (s)'); ylabel('Normalized signal');
title(sprintf('Mouse M%d | R^2 = %.3f | α = %.3f | β = %.3f', ...
    mouse_id, R2_true, alpha, beta));
set(gca, 'FontSize', 14, 'LineWidth', 1.2);

%% === Permutation test (circular shift) ===
R2_null = zeros(n_iter, 1);
N = numel(AM);

for i = 1:n_iter
    shift = randi(N - 1);
    AM_perm = circshift(AM, shift);
    R2_null(i) = plaw_kde(V1, AM_perm);  % Only R² needed
end

% Compute empirical p-value
p_val = mean(R2_null >= R2_true);

%% === Report Results ===
fprintf('Mouse %d results:\n', mouse_id);
fprintf('Alpha: %.4f\n', alpha);
fprintf('Beta : %.4f\n', beta);
fprintf('True R²: %.4f\n', R2_true);
fprintf('Permutation-based p-value: %.4f\n', p_val);

%% === Subfunctions ===

function pc1 = pca_first_component(data_mat)
    % Returns the first principal component (temporal)
    data_mat = detrend(data_mat')';         % Optional: remove linear trend
    [~, score, ~] = pca(data_mat');         % PCA along time dimension
    pc1 = score(:,1);                       % First PC time course
end

function normed = normalize_to_unit_range(sig, ref)
    % Normalizes `sig` using min/max range of both sig and ref
    lo = min([sig(:); ref(:)]);
    hi = max([sig(:); ref(:)]);
    normed = (sig - lo) / (hi - lo);
end

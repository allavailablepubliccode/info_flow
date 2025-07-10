% Figure 10: Heatmaps of Expectation–Entropy Occupancy and SEM

clear; clc; close all; warning('off','all');

% === Parameters ===
numfiles = 28;               % Number of trials
Fs = 99.8722;                % Sampling frequency (Hz)
window = 30;                 % Sliding window length (short)
step = 30;                   % Step size (non-overlapping)
nbins = 70;                  % Histogram bin resolution

% === Preallocate storage ===
all_mu = cell(1, numfiles);       % Per-trial expectation
all_logvar = cell(1, numfiles);   % Per-trial log-variance

% === Loop over trials ===
for ii = 1:numfiles
    % Load data
    load(['~/Dropbox/Work/AT_Calcium/mat/data-001_ratio_' num2str(ii) '.mat'], ...
        'data_cerebellum', 'data_cerebrum');

    % Predict cerebellum signal from cerebrum
    [~, ~, ~, ~, macro_hat, ~] = ...
        test_transform_mv(data_cerebrum, data_cerebellum);

    % Sliding window mean and log-variance
    x = macro_hat(:)';
    T = length(x);
    n_windows = floor((T - window)/step);
    mu = zeros(1, n_windows);
    logvar = zeros(1, n_windows);

    for i = 1:n_windows
        idx = (i-1)*step + (1:window);
        xi = x(idx);
        mu(i) = mean(xi);
        logvar(i) = log(var(xi) + 1e-8);
    end

    all_mu{ii} = mu;
    all_logvar{ii} = logvar;
end

% === Build common bin edges from all pooled data ===
all_mu_cat = cat(2, all_mu{:});
all_logvar_cat = cat(2, all_logvar{:});
[~, xedges] = histcounts(all_mu_cat, nbins);
[~, yedges] = histcounts(all_logvar_cat, nbins);
x_centers = 0.5 * (xedges(1:end-1) + xedges(2:end));
y_centers = 0.5 * (yedges(1:end-1) + yedges(2:end));

% === Per-trial 2D histograms with Gaussian smoothing ===
count_all = nan(numfiles, nbins, nbins);
g = fspecial('gaussian', [10 10], 1);  % Gaussian kernel

for ii = 1:numfiles
    mu = all_mu{ii};
    logvar = all_logvar{ii};
    [counts2d, ~, ~] = histcounts2(mu, logvar, xedges, yedges);
    counts2d(counts2d == 0) = NaN;

    % Apply Gaussian smoothing while preserving NaNs
    mask = ~isnan(counts2d);
    counts2d_filled = counts2d;
    counts2d_filled(~mask) = 0;

    smoothed = imfilter(counts2d_filled, g, 'replicate');
    normalizer = imfilter(double(mask), g, 'replicate');
    smoothed(mask) = smoothed(mask) ./ max(normalizer(mask), 1e-8);

    count_all(ii,:,:) = permute(smoothed, [3 1 2]);
end

% === Mean and SEM across trials (ignoring NaNs) ===
count_mean = squeeze(nanmean(count_all, 1));
count_sem  = squeeze(nanstd(count_all, 1)) / sqrt(numfiles);

% Mask out bins visited by fewer than N subjects
min_subjects = 2;
subject_presence = ~isnan(count_all);
bin_support = squeeze(sum(subject_presence, 1));
count_mean(bin_support < min_subjects) = NaN;
count_sem(bin_support < min_subjects)  = NaN;

% === Optional smoothing of mean and SEM maps ===
count_mean_smooth = smooth_ignore_nan(count_mean, g);
count_sem_smooth  = smooth_ignore_nan(count_sem,  g);

% === Interpolation for visualization ===
[xq, yq] = meshgrid(linspace(min(x_centers), max(x_centers), 200), ...
                    linspace(min(y_centers), max(y_centers), 200));

mean_interp = interp2(x_centers, y_centers, count_mean_smooth', xq, yq, 'cubic');
sem_interp  = interp2(x_centers, y_centers, count_sem_smooth',  xq, yq, 'cubic');

% Optional alpha mask to suppress extrapolation artifacts
alpha_mask = ~isnan(interp2(x_centers, y_centers, ...
    double(~isnan(count_mean_smooth')), xq, yq, 'nearest'));

% === Plotting ===
figure('Position', [100, 100, 1000, 400], 'Color','w');

clamp = 0.8;  % Clip maximum values to enhance contrast

subplot(1,2,1);
contourf(xq, yq, mean_interp, 20, 'LineColor', 'none'); axis xy;
colormap('viridis'); colorbar;
xlabel('Mean (Expectation)', 'FontSize', 14);
ylabel('Log Variance (Entropy Proxy)', 'FontSize', 14);
title('Mean dwell time across subjects (smoothed)', 'FontSize', 16);
caxis([min(mean_interp(:)), clamp*max(mean_interp(:))]);

subplot(1,2,2);
contourf(xq, yq, sem_interp, 20, 'LineColor', 'none'); axis xy;
colormap('viridis'); colorbar;
xlabel('Mean (Expectation)', 'FontSize', 14);
ylabel('Log Variance (Entropy Proxy)', 'FontSize', 14);
title('Standard error across subjects (smoothed)', 'FontSize', 16);
caxis([min(sem_interp(:)), clamp*max(sem_interp(:))]);

% Export
exportgraphics(gcf, '~/Desktop/Fig_10.pdf', 'ContentType', 'vector');


% === Helper: Smoothing that respects NaNs ===
function A_smooth = smooth_ignore_nan(A, kernel)
    mask = ~isnan(A);
    A_filled = A;
    A_filled(~mask) = 0;

    A_blur = imfilter(A_filled, kernel, 'replicate');
    norm_factor = imfilter(double(mask), kernel, 'replicate');

    A_smooth = A_blur ./ max(norm_factor, 1e-8);
    A_smooth(~mask & norm_factor == 0) = NaN;
end


% === Helper: Transform macro via micro, same as Figure 9 ===
function [alpha, beta, R2, R2rev, macro_hat, micro_hat, entropy_term, linear_term] = test_transform_mv(micro, macro)
    micro = micro(:);
    macro = macro(:);

    % Estimate log-density of micro
    [counts, edges] = histcounts(micro, 100, 'Normalization', 'pdf');
    bin_centers = 0.5*(edges(1:end-1) + edges(2:end));
    logq_vals = log(counts + 1e-8);
    logq_x = interp1(bin_centers, logq_vals, micro, 'linear', 'extrap');

    logq_mean = mean(logq_x);
    mu_x = mean(micro);
    opts = optimset('MaxFunEvals', 10000, 'MaxIter', 10000, 'Display', 'off');

    errfun = @(p) sum((macro - (micro + ...
        p(1)*(logq_x - logq_mean) + p(2)*(micro - mu_x))).^2);
    params = fminsearch(errfun, [0.1, 0.1], opts);

    alpha = params(1);
    beta  = params(2);

    entropy_term = alpha * (logq_x - logq_mean);
    linear_term  = beta  * (micro - mu_x);
    macro_hat = micro + entropy_term + linear_term;

    % Optional reverse transformation
    [counts2, edges2] = histcounts(macro, 100, 'Normalization', 'pdf');
    bin_centers2 = 0.5*(edges2(1:end-1) + edges2(2:end));
    logq_y = interp1(bin_centers2, log(counts2 + 1e-8), macro, 'linear', 'extrap');
    logq_mean2 = mean(logq_y);
    mu_y = mean(macro);

    errfun_rev = @(p) sum((micro - (macro + ...
        p(1)*(logq_y - logq_mean2) + p(2)*(macro - mu_y))).^2);
    params_rev = fminsearch(errfun_rev, [0.1, 0.1], opts);

    micro_hat = macro + ...
        params_rev(1)*(logq_y - logq_mean2) + ...
        params_rev(2)*(macro - mu_y);

    resid     = macro - macro_hat;
    resid_rev = micro - micro_hat;
    R2        = 1 - sum(resid.^2)     / sum((macro - mean(macro)).^2);
    R2rev     = 1 - sum(resid_rev.^2) / sum((micro - mean(micro)).^2);
end

%==========================================================================
% Generate Figure 3: Wasserstein-based parameter recovery from transformed densities
%
% This script:
%   - Simulates samples evolving over time from a diffusing Gaussian
%   - Applies a nonlinear transformation involving entropy and drift terms
%   - Adds noise to simulate observations
%   - Fits transformation parameters (alpha, beta) by minimizing Wasserstein-2 loss
%   - Compares true vs. fit distributions at selected time points
%
% Author: Erik D. Fagerholm
% Date: 6 August 2025
%==========================================================================

clear; clc; close all; rng(0);  % For reproducibility

%% === Model Parameters ===
mu_0      = 0;       % Initial mean
sigma2_0  = 1;       % Initial variance
D         = 1;       % Diffusion coefficient
alpha_true = 0.2;    % True alpha (logq term coefficient)
beta_true  = 0.1;    % True beta (drift coefficient)
T         = 100;     % Number of time points
t         = linspace(0, 10, T);  % Time grid
Nu        = 0.1;     % Observation noise level
xvals     = linspace(-5, 5, 1000);  % Evaluation grid
time_indices = [1, 10];  % Time points to evaluate fit

%% === Simulate Transformed Samples and Noisy Observations ===
samples_per_t = 2000;
q_true_all = cell(1, T);     % Transformed true samples
samples_obs = cell(1, T);    % Noisy observed samples

for ti = 1:T
    sigma2_t = sigma2_0 + 2 * D * t(ti);
    mu_t = mu_0;

    x = mu_t + sqrt(sigma2_t) * randn(samples_per_t, 1);      % Base Gaussian
    logq = -0.5 * ((x - mu_t).^2 / sigma2_t);                 % Log-density
    logq = logq - mean(logq);                                % Centered logq

    x_trans = x + alpha_true * logq + beta_true * (x - mu_t);  % Apply transformation
    x_obs = x_trans + Nu * randn(size(x_trans));               % Add observation noise

    q_true_all{ti} = x_trans;
    samples_obs{ti} = x_obs;
end

%% === Fit alpha and beta using Wasserstein loss ===
lossfun = @(params) compute_total_wasserstein(params, samples_obs, mu_0, sigma2_0, D, t, samples_per_t, time_indices);

params0 = [0, 0];                  % Initial guess [alpha, beta]
lb = [-20, -20];                   % Lower bounds
ub = [20, 20];                     % Upper bounds
opts = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'MaxFunEvals', 1e5, ...
    'MaxIter', 1e4);

[params_fit, fval] = fmincon(lossfun, params0, [],[],[],[], lb, ub, [], opts);
alpha_fit = params_fit(1);
beta_fit  = params_fit(2);

%% === Plot true vs. fit densities at selected time points ===
figure('Color', 'w'); hold on;
set(gcf, 'Position', [600, 300, 500, 400]);

legend_entries = {};
f_true_all_eval = cell(1, 2);
f_fit_all_eval  = cell(1, 2);
ymax = 0;

for i = 1:2
    tid = time_indices(i);
    tval = t(tid);

    % True density
    [f_true, ~] = ksdensity(q_true_all{tid}, xvals);
    f_true_all_eval{i} = f_true;
    plot(xvals, f_true, '-', 'Color', [0.8 0 0], 'LineWidth', 2);
    legend_entries{end+1} = sprintf('t = %.1f (true)', tval);

    % Fit density
    sigma2_t = sigma2_0 + 2 * D * t(tid);
    mu_t = mu_0;
    x = mu_t + sqrt(sigma2_t) * randn(samples_per_t, 1);
    logq = -0.5 * ((x - mu_t).^2 / sigma2_t);
    logq = logq - mean(logq);
    x_mod = x + alpha_fit * logq + beta_fit * (x - mu_t);

    [f_fit, ~] = ksdensity(x_mod, xvals);
    f_fit_all_eval{i} = f_fit;
    plot(xvals, f_fit, '--k', 'LineWidth', 2);
    legend_entries{end+1} = sprintf('t = %.1f (fit)', tval);

    ymax = max([ymax, f_true, f_fit]);
end

xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', 16);
title('Wasserstein-based fit of transformed densities');
legend(legend_entries, 'Location', 'northeast');
set(gca, 'FontSize', 14, 'LineWidth', 1.2);
ylim([0, ymax * 1.05]);

%% === Report Fit Errors ===
alpha_err = abs(alpha_fit - alpha_true) / abs(alpha_true) * 100;
beta_err  = abs(beta_fit  - beta_true ) / abs(beta_true )  * 100;
fprintf('Alpha recovery error: %.2f%%\n', alpha_err);
fprintf('Beta  recovery error: %.2f%%\n', beta_err);

%% === Compute Additional Curve Metrics ===
[W2, TVD, L2] = compute_fit_metrics(f_true_all_eval, f_fit_all_eval, xvals);
fprintf('Avg Wasserstein-2 (CDF²): %.4f\n', W2);
fprintf('Avg Total Variation Dist: %.4f\n', TVD);
fprintf('Avg L2 error: %.4f\n', L2);

%% === Subfunctions ===

function loss = compute_total_wasserstein(params, samples_obs, mu_0, sigma2_0, D, t, n, time_indices)
    % Computes sum of Wasserstein-2 distances between model and observed data
    alpha = params(1);
    beta  = params(2);
    loss = 0;

    for idx = 1:numel(time_indices)
        ti = time_indices(idx);
        sigma2_t = sigma2_0 + 2 * D * t(ti);
        mu_t = mu_0;

        x = mu_t + sqrt(sigma2_t) * randn(n, 1);
        logq = -0.5 * ((x - mu_t).^2 / sigma2_t);
        logq = logq - mean(logq);
        x_mod = x + alpha * logq + beta * (x - mu_t);

        [f1, x1] = ksdensity(x_mod);
        [f2, x2] = ksdensity(samples_obs{ti});

        x_common = linspace(min([x1 x2]), max([x1 x2]), 300);
        f1i = interp1(x1, f1, x_common, 'linear', 0);
        f2i = interp1(x2, f2, x_common, 'linear', 0);

        % Normalize
        f1i = f1i / trapz(x_common, f1i);
        f2i = f2i / trapz(x_common, f2i);

        cdf1 = cumsum(f1i) / sum(f1i);
        cdf2 = cumsum(f2i) / sum(f2i);
        loss = loss + trapz(x_common, (cdf1 - cdf2).^2);
    end
end

function [W2, TVD, L2] = compute_fit_metrics(f_true_all, f_fit_all, xvals)
    % Computes Wasserstein-2 (CDF²), Total Variation Distance, and L2 error
    n = length(f_true_all);
    W2 = 0; TVD = 0; L2 = 0;

    for i = 1:n
        f1 = f_true_all{i};
        f2 = f_fit_all{i};
        dx = mean(diff(xvals));

        cdf1 = cumsum(f1) * dx;
        cdf2 = cumsum(f2) * dx;

        W2  = W2  + trapz(xvals, (cdf1 - cdf2).^2);
        TVD = TVD + 0.5 * trapz(xvals, abs(f1 - f2));
        L2  = L2  + trapz(xvals, (f1 - f2).^2);
    end

    W2  = W2  / n;
    TVD = TVD / n;
    L2  = L2  / n;
end

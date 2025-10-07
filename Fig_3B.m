%==========================================================================
% Generate Figure 3B: Recovery of transformation parameters from dynamic signal
%
% This script:
%   1. Simulates a 1D stochastic signal with time-varying drift
%   2. Applies a nonlinear transformation based on log-density and global mean
%   3. Estimates summary statistics in sliding windows
%   4. Recovers transformation parameters (alpha, beta) using the stats model
%   5. Reconstructs transformed signal and compares to ground truth
%
%==========================================================================

clear; clc; close all; rng(0);  % Reproducibility

%% === 1. Generate synthetic signal with drift ===
T = 1000;                 % Number of time points
dt = 0.05;                % Time step
t = (0:T-1) * dt;         % Time vector
D = 2;                    % Diffusion coefficient

a = sin(2 * pi * 0.5 * t);  % Time-varying drift
x = zeros(1, T);            % Initialize signal

for i = 2:T
    x(i) = x(i-1) + a(i-1)*dt + sqrt(2*D*dt)*randn;
end

%% === 2. Estimate log-density log q(x(t)) via KDE ===
[fhat, xgrid] = ksdensity(x);
logq_vals = log(fhat + 1e-8);  % Avoid log(0)
logq_x = interp1(xgrid, logq_vals, x, 'linear', 'extrap');
logq_mean = mean(logq_x);

%% === 3. Apply ground-truth transformation ===
alpha_true = 0.2;
beta_true  = 0.1;
mu_global = mean(x);

x_lambda = x + ...
    alpha_true * (logq_x - logq_mean) + ...
    beta_true  * (x - mu_global);

%% === 4. Compute summary stats in sliding windows ===
window = 100; step = 20;
n_windows = floor((T - window) / step);
mu_t     = zeros(1, n_windows);  % Mode
sigma2_t = zeros(1, n_windows);  % Variance
q_mu_t   = zeros(1, n_windows);  % PDF at mode

for i = 1:n_windows
    idx = (i-1)*step + (1:window);
    xi = x(idx);

    [pdf_vals, grid] = ksdensity(xi);
    [~, id] = max(pdf_vals);

    mu_t(i) = grid(id);       % Mode
    sigma2_t(i) = var(xi);    % Variance
    q_mu_t(i) = pdf_vals(id); % Estimated density at mode
end

%% === 5. Apply transformation to summary stats ===
mu_lambda = mu_t + (beta_true .* sigma2_t) ./ (1 + alpha_true);
q_lambda  = sqrt(1 + alpha_true) ./ sqrt(2 * pi .* sigma2_t);

% Add mild observational noise
mu_lambda_obs = mu_lambda + 0.03 * randn(size(mu_lambda));
q_lambda_obs  = q_lambda  .* (1 + 0.03 * randn(size(q_lambda)));

%% === 6. Fit alpha and beta using summary stats ===
errfun = @(params) sum( ...
    (mu_lambda_obs - (mu_t + (params(2) .* sigma2_t) ./ (1 + params(1)))).^2 + ...
    (q_lambda_obs  - sqrt(1 + params(1)) ./ sqrt(2 * pi .* sigma2_t)).^2 ...
);

params0 = [0, 0];  % Initial guess [alpha, beta]
estimated = fminsearch(errfun, params0);
alpha_fit = estimated(1);
beta_fit  = estimated(2);

%% === 7. Reconstruct signal using recovered parameters ===
x_lambda_recon = x + ...
    alpha_fit * (logq_x - logq_mean) + ...
    beta_fit  * (x - mu_global);

%% === 8. Plot original, transformed, and reconstructed signals ===
figure('Color','w');
set(gcf, 'Position', [100, 100, 700, 500]);
hold on;

plot(t, x_lambda, 'r', 'LineWidth', 1.5, 'DisplayName', 'Ground-truth');
plot(t, x_lambda_recon, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Recovered');

xlabel('$\mathit{Time}$',  'Interpreter','latex', 'FontSize', 22);
ylabel('$\mathit{Signal}$','Interpreter','latex', 'FontSize', 22);
legend('Location','north','Interpreter','latex','FontSize',12,'Box','off');
set(gca,'FontSize',16,'LineWidth',1.2);
xlim([1 5]);  % Adjust zoom for clarity
% ylim([-1.5 0.5]);  % Optional

%% === 9. Report recovery errors ===
mae_alpha   = abs(alpha_fit - alpha_true);
rmse_alpha  = sqrt((alpha_fit - alpha_true)^2);
relerr_alpha = 100 * mae_alpha / abs(alpha_true);

mae_beta   = abs(beta_fit - beta_true);
rmse_beta  = sqrt((beta_fit - beta_true)^2);
relerr_beta = 100 * mae_beta / abs(beta_true);

fprintf('\nAlpha error: MAE=%.4f, RMSE=%.4f, %%Error=%.2f%%\n', mae_alpha, rmse_alpha, relerr_alpha);
fprintf('Beta  error: MAE=%.4f, RMSE=%.4f, %%Error=%.2f%%\n', mae_beta, rmse_beta, relerr_beta);

%% === Additional fit quality metrics (optional) ===
tv_dist = sum(abs(x_lambda - x_lambda_recon)) / length(x_lambda);
l2_err  = sqrt(mean((x_lambda - x_lambda_recon).^2));
fprintf('Total Variation Distance: %.4f\n', tv_dist);
fprintf('L2 Error: %.4f\n', l2_err);

% generate_fig5.m
% Figure 5: Recovery of transformation parameters from summary statistics of a synthetic signal

clear; clc; close all; rng(0);  % for reproducibility

%% === 1. Generate synthetic signal with drift ===
T = 1000;         % number of time points
dt = 0.01;        % time step
t = (0:T-1) * dt; % time vector
D = 0.05;         % diffusion coefficient

a = sin(2 * pi * 0.5 * t);  % oscillatory drift
x = zeros(1, T);            % initialize signal

% Euler-Maruyama integration of stochastic drift
for i = 2:T
    x(i) = x(i-1) + a(i-1)*dt + sqrt(2*D*dt)*randn;
end

%% === 2. Estimate log-density log q(x(t)) via histogram ===
[counts, edges] = histcounts(x, 100, 'Normalization', 'pdf');
bin_centers = 0.5 * (edges(1:end-1) + edges(2:end));
logq_vals = log(counts + 1e-8);       % avoid log(0)
logq_mean = mean(logq_vals);          % mean of log-density

logq_x = interp1(bin_centers, logq_vals, x, 'linear', 'extrap');  % log-density at each x(t)

%% === 3. Ground truth transformation (entropy + expectation flow) ===
alpha_true = 0.3;
beta_true  = 0.9;
mu_global  = mean(x);

x_lambda = x + ...
    alpha_true * (logq_x - logq_mean) + ...
    beta_true * (x - mu_global);  % apply transformation

%% === 4. Compute summary stats in sliding windows ===
window = 100;
step = 10;
n_windows = floor((T - window)/step);

mu_t     = zeros(1, n_windows);
sigma2_t = zeros(1, n_windows);
q_mu_t   = zeros(1, n_windows);

for i = 1:n_windows
    idx = (i-1)*step + (1:window);
    xi = x(idx);
    
    % Kernel density estimate to get mode (peak) of PDF
    [pdf_vals, grid] = ksdensity(xi);
    [~, id] = max(pdf_vals);
    
    mu_t(i)     = grid(id);    % mode as proxy for location
    sigma2_t(i) = var(xi);     % variance
    q_mu_t(i)   = max(pdf_vals); % height at mode
end

%% === 5. Apply same transformation analytically to summary stats ===
mu_lambda = mu_t + (beta_true .* sigma2_t) ./ (1 + alpha_true);
q_lambda  = sqrt(1 + alpha_true) ./ sqrt(2 * pi .* sigma2_t);

% Add noise to simulate empirical observations
mu_lambda_obs = mu_lambda + 0.03 * randn(size(mu_lambda));
q_lambda_obs  = q_lambda .* (1 + 0.03 * randn(size(q_lambda)));

%% === 6. Estimate alpha and beta from observed summary stats ===
errfun = @(params) sum( ...
    (mu_lambda_obs - (mu_t + (params(2) .* sigma2_t) ./ (1 + params(1)))).^2 + ...
    (q_lambda_obs  - sqrt(1 + params(1)) ./ sqrt(2 * pi .* sigma2_t)).^2 ...
);

params0 = [0.1, 0.1];  % initial guess
estimated = fminsearch(errfun, params0);
alpha_fit = estimated(1);
beta_fit  = estimated(2);

%% === 7. Evaluate estimation accuracy ===
mae_alpha    = abs(alpha_fit - alpha_true);
rmse_alpha   = sqrt((alpha_fit - alpha_true)^2);
relerr_alpha = 100 * mae_alpha / abs(alpha_true);

mae_beta    = abs(beta_fit - beta_true);
rmse_beta   = sqrt((beta_fit - beta_true)^2);
relerr_beta = 100 * mae_beta / abs(beta_true);

%% === 8. Plot original, transformed, and recovered signals ===
figure('Color','w');
set(gcf, 'Position', [100, 100, 700, 500]);

plot(t, x,          'k-', 'LineWidth', 1.5, 'DisplayName', 'Original $x(t)$'); hold on;
plot(t, x_lambda,   'r-', 'LineWidth', 1.5, 'DisplayName', 'Transformed $x_\lambda(t)$');

% === 9. Reconstruct signal using recovered parameters ===
x_lambda_recon = x + ...
    alpha_fit * (logq_x - logq_mean) + ...
    beta_fit  * (x - mu_global);

plot(t, x_lambda_recon, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Recovered $x_\lambda(t)$');

xlabel('$\mathit{Time}$',  'Interpreter', 'latex', 'FontSize', 22);
ylabel('$\mathit{Signal}$','Interpreter', 'latex', 'FontSize', 22);
set(gca, 'FontSize', 16, 'LineWidth', 1.2);
legend('Location','northeast', 'Interpreter','latex', 'FontSize', 16, 'Box','off');

%% === 10. Print stats ===
fprintf('\nGround Truth alpha: %.4f\n', alpha_true);
fprintf('Estimated     alpha: %.4f\n', alpha_fit);
fprintf('Alpha MAE: %.4f, RMSE: %.4f, %%Error: %.2f%%\n', ...
    mae_alpha, rmse_alpha, relerr_alpha);

fprintf('\nGround Truth beta : %.4f\n', beta_true);
fprintf('Estimated     beta : %.4f\n', beta_fit);
fprintf('Beta  MAE: %.4f, RMSE: %.4f, %%Error: %.2f%%\n', ...
    mae_beta, rmse_beta, relerr_beta);

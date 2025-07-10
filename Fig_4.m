% generate_fig4.m
% Figure 4: Simulated forward and inverse transformation of a Gaussian

clear; clc; close all;

%% === Simulation Parameters ===
mu_0      = 0;            % initial mean
sigma2_0  = 1/4;          % initial variance
D         = 0.02;         % diffusion coefficient
alpha_true = 1.2;         % true entropy weight
beta_true  = 7.4;         % true expectation weight
T         = 50;           % number of time steps
t         = linspace(0, 10, T);
Nu        = 0.25;         % observation noise
xvals     = linspace(-14, 14, 1000);  % domain for plotting

%% === Simulate forward Gaussian evolution ===
mu_t      = mu_0 * ones(size(t));          % constant mean
sigma2_t  = sigma2_0 + 2 * D * t;          % diffusion over time

% Apply entropy+expectation flow (ground truth)
mu_lambda = mu_t + (beta_true .* sigma2_t) ./ (1 + alpha_true);
q_lambda_height = sqrt(1 + alpha_true) ./ sqrt(2 * pi .* sigma2_t);

% Add Gaussian noise to simulate observation error
rng(1);  % for reproducibility
mu_lambda_obs = mu_lambda + Nu * randn(size(mu_lambda));
q_lambda_obs  = q_lambda_height .* (1 + Nu * randn(size(q_lambda_height)));

%% === Parameter Inversion (recovery from noisy measurements) ===
errfun = @(params) sum( ...
    (mu_lambda_obs - (mu_t + (params(2) * sigma2_t) ./ (1 + params(1)))).^2 + ...
    (q_lambda_obs  - sqrt(1 + params(1)) ./ sqrt(2 * pi * sigma2_t)).^2 ...
);

params0 = [0.1, 0.1];  % initial guess
estimated = fminsearch(errfun, params0);
alpha_fit = estimated(1);
beta_fit  = estimated(2);

% Reconstruct peak height and location using estimated parameters
mu_lambda_fit      = mu_t + (beta_fit .* sigma2_t) ./ (1 + alpha_fit);
q_lambda_fit_height = sqrt(1 + alpha_fit) ./ sqrt(2 * pi .* sigma2_t);

%% === Plot example timepoints ===
t_idx = round(linspace(1, T, 2));  % select early and late timepoints

% Precompute max y-value across curves for consistent y-limits
ymax = 0;
for i = 1:length(t_idx)
    tid = t_idx(i);
    q_orig = normpdf(xvals, mu_t(tid), sqrt(sigma2_t(tid)));
    q_true = normpdf(xvals, mu_lambda(tid), sqrt(1 / (2 * pi * sigma2_t(tid)) * (1 + alpha_true))^(-1));
    q_fit  = normpdf(xvals, mu_lambda_fit(tid), sqrt(1 / (2 * pi * sigma2_t(tid)) * (1 + alpha_fit))^(-1));
    ymax = max([ymax, q_orig, q_true, q_fit], [], 'all');
end
ymax = ymax * 1.1;

% === Plot ===
figure('Color','w'); 
set(gcf,'Position',[676 1 349 536]);

for i = 1:length(t_idx)
    tid = t_idx(i);

    % Generate distributions
    q_orig = normpdf(xvals, mu_t(tid), sqrt(sigma2_t(tid)));
    q_true = normpdf(xvals, mu_lambda(tid), sqrt(1 / (2 * pi * sigma2_t(tid)) * (1 + alpha_true))^(-1));
    q_fit  = normpdf(xvals, mu_lambda_fit(tid), sqrt(1 / (2 * pi * sigma2_t(tid)) * (1 + alpha_fit))^(-1));

    % Plot
    subplot(2,1,i)
    plot(xvals, q_orig, 'b-', 'LineWidth', 1.5); hold on;
    plot(xvals, q_true, 'r-', 'LineWidth', 1.5);
    plot(xvals, q_fit,  'k--', 'LineWidth', 1.5);
    xlim([-2 5]); 
    ylim([0, ymax * 0.92]);
    title(sprintf('t = %.1f', t(tid)), 'FontSize', 12);
end

% Annotate figure
sgtitle('Evolution of original, transformed, and recovered Gaussians', 'FontSize', 16);
legend({'Original', 'Transformed (true)', 'Transformed (recovered)'}, ...
    'Position',[0.35, 0.03, 0.3, 0.03], 'Orientation','horizontal', 'Box','off');

%% === Report recovery accuracy ===
alpha_error = abs(alpha_true - alpha_fit) / abs(alpha_true) * 100;
beta_error  = abs(beta_true - beta_fit)   / abs(beta_true)  * 100;

fprintf('Alpha recovery error: %.2f%%\n', alpha_error);
fprintf('Beta recovery error:  %.2f%%\n', beta_error);

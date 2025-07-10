% generate_fig6.m
% Figure 6: Visualization of signal trajectory in expectation–entropy space

clear; clc; close all; rng(0);  % for reproducibility

%% === 1. Generate synthetic signal with drift ===
T = 3000;              % number of time points
dt = 0.01;             % time step
t = (0:T-1) * dt;      % time vector
D = 0.05;              % diffusion coefficient
a = sin(2 * pi * 0.1 * t);  % slow sinusoidal drift

x = zeros(1, T);       % initialize signal
for i = 2:T
    x(i) = x(i-1) + a(i-1)*dt + sqrt(2*D*dt)*randn;
end

%% === 2. Estimate log-density log q(x(t)) via histogram ===
[counts, edges] = histcounts(x, 100, 'Normalization', 'pdf');
bin_centers = 0.5 * (edges(1:end-1) + edges(2:end));
logq_vals = log(counts + 1e-8);       % avoid log(0)
logq_mean = mean(logq_vals);          % baseline log-density

logq_x = interp1(bin_centers, logq_vals, x, 'linear', 'extrap');  % log-density at x(t)

%% === 3. Apply entropy + expectation transformation ===
alpha_true = 0.3;
beta_true  = 0.9;
mu_global  = mean(x);

x_lambda = x + ...
    alpha_true * (logq_x - logq_mean) + ...
    beta_true  * (x - mu_global);  % apply transformation

%% === 4. Compute summary stats in sliding windows ===
window = 100;       % sliding window size (frames)
step = 10;          % step size (frames)
n_windows = floor((T - window) / step);

% Preallocate
mu_t     = zeros(1, n_windows);
logvar_t = zeros(1, n_windows);
mu_lambda_t     = zeros(1, n_windows);
logvar_lambda_t = zeros(1, n_windows);

for i = 1:n_windows
    idx = (i-1)*step + (1:window);
    xi  = x(idx);
    xli = x_lambda(idx);

    mu_t(i)          = mean(xi);
    logvar_t(i)      = log(var(xi) + 1e-8);  % log-variance as entropy proxy
    mu_lambda_t(i)   = mean(xli);
    logvar_lambda_t(i) = log(var(xli) + 1e-8);
end

time_axis = (0:n_windows-1) * step * dt;  % convert frame to time

%% === 5. Plot trajectory in 3D information space ===
figure('Position', [100, 100, 1200, 800], 'Color', 'w');

plot3(mu_t, time_axis, logvar_t, ...
    'k-', 'LineWidth', 3, 'DisplayName', 'Original');

hold on;

plot3(mu_lambda_t, time_axis, logvar_lambda_t, ...
    'r-', 'LineWidth', 3, 'DisplayName', 'Transformed');

xlabel('Expectation (mean)', 'FontSize', 14);
ylabel('Time', 'FontSize', 14);
zlabel('Log variance (entropy proxy)', 'FontSize', 14);
grid on;
legend('Location','best', 'FontSize', 14);
view(58, 47);  % camera angle

% Save as vector graphic
exportgraphics(gca, '~/Desktop/Fig6.eps', 'ContentType', 'vector');

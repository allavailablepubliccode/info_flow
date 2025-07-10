% generate_fig7.m
% Figure 7: 2D histograms in expectation–entropy space

clear; clc; close all; rng(0);  % Reproducibility

%% === 1. Generate synthetic signal ===
T = 3000;                   % number of timepoints
dt = 0.01;                  % timestep
t = (0:T-1) * dt;           % time axis
D = 0.05;                   % diffusion
a = sin(2 * pi * 0.1 * t);  % drift function

x = zeros(1,T);
for i = 2:T
    x(i) = x(i-1) + a(i-1)*dt + sqrt(2*D*dt)*randn;
end

%% === 2. Estimate log-density log q(x(t)) ===
[counts, edges] = histcounts(x, 100, 'Normalization', 'pdf');
bin_centers = 0.5 * (edges(1:end-1) + edges(2:end));
logq_vals = log(counts + 1e-8);
logq_mean = mean(logq_vals);
logq_x = interp1(bin_centers, logq_vals, x, 'linear', 'extrap');

%% === 3. Apply entropy-expectation transformation ===
alpha = 0.3;
beta = 0.9;
mu_x = mean(x);
x_lambda = x + alpha * (logq_x - logq_mean) + beta * (x - mu_x);

%% === 4. Sliding window summary stats ===
window = 100;
step = 10;
n_windows = floor((T - window)/step);

mu = zeros(1, n_windows);       % mean of x
lv  = zeros(1, n_windows);      % log-variance of x (entropy proxy)
mu_l = zeros(1, n_windows);     % mean of x_lambda
lv_l = zeros(1, n_windows);     % log-variance of x_lambda

for i = 1:n_windows
    idx = (i-1)*step + (1:window);
    xi  = x(idx);
    xli = x_lambda(idx);
    mu(i)  = mean(xi);
    lv(i)  = log(var(xi) + 1e-8);
    mu_l(i) = mean(xli);
    lv_l(i) = log(var(xli) + 1e-8);
end

%% === 5. Define histogram edges ===
nbins = 60;
xrange = [min([mu, mu_l]), max([mu, mu_l])];
yrange = [min([lv, lv_l]), max([lv, lv_l])];

xedges = linspace(xrange(1), xrange(2), nbins+1);
yedges = linspace(yrange(1), yrange(2), nbins+1);
x_centers = 0.5 * (xedges(1:end-1) + xedges(2:end));
y_centers = 0.5 * (yedges(1:end-1) + yedges(2:end));

%% === 6. Compute 2D histograms ===
H_orig  = histcounts2(mu,   lv,   xedges, yedges);
H_trans = histcounts2(mu_l, lv_l, xedges, yedges);

%% === 7. Smooth histograms with 2D Gaussian ===
g = fspecial('gaussian', [3 3], 1);
smooth = @(A) imfilter(double(A), g, 'replicate');

H_orig_smooth  = smooth(H_orig);
H_trans_smooth = smooth(H_trans);

%% === 8. Interpolate for smooth visualization ===
[xq, yq] = meshgrid(linspace(xrange(1), xrange(2), 200), ...
                    linspace(yrange(1), yrange(2), 200));

H1_interp = interp2(x_centers, y_centers, H_orig_smooth',  xq, yq, 'cubic');
H2_interp = interp2(x_centers, y_centers, H_trans_smooth', xq, yq, 'cubic');

%% === 9. Plot ===
figure('Position', [200, 200, 1000, 400], 'Color', 'w');

% --- Original ---
subplot(1,2,1);
contourf(xq, yq, H1_interp, 20, 'LineColor', 'none'); axis xy;
colormap('viridis'); colorbar;
xlabel('Expectation', 'FontSize', 14);
ylabel('Entropy (log-variance)', 'FontSize', 14);
title('Original Signal', 'FontSize', 16);

% --- Transformed ---
subplot(1,2,2);
contourf(xq, yq, H2_interp, 20, 'LineColor', 'none'); axis xy;
colormap('viridis'); colorbar;
xlabel('Expectation', 'FontSize', 14);
ylabel('Entropy (log-variance)', 'FontSize', 14);
title('Transformed Signal', 'FontSize', 16);

% --- Save ---
exportgraphics(gcf, '~/Desktop/Fig7.pdf', 'ContentType', 'vector');

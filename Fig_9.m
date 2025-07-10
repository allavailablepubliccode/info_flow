% Figure 9: 3D Visualization of Normalized Mean-Variance Dynamics
% Across Trials Using SEM Envelope

clear; clc; close all; warning('off','all');

% === Parameters ===
numfiles = 28;               % Number of data files (trials)
Fs = 99.8722;                % Sampling frequency (Hz)
window = 300;                % Sliding window length (~3 seconds)
step = 30;                   % Step size (~10 Hz sampling)

% Preallocate results
alpha_all = zeros(1, numfiles);
beta_all  = zeros(1, numfiles);
R2_all    = zeros(1, numfiles);
all_mu = [];                 % Expectation (mean) per window, pooled
all_logvar = [];             % Log-variance per window, pooled
time_vec = [];

% === Process all trials ===
for ii = 1:numfiles
    % Load cerebrum and cerebellum time series
    load(['~/Dropbox/Work/AT_Calcium/mat/data-001_ratio_' num2str(ii) '.mat'], ...
        'data_cerebellum', 'data_cerebrum');

    % Predict cerebellum signal from cerebrum
    [alpha, beta, R2, ~, macro_hat, ~, entropy_term, linear_term] = ...
        test_transform_mv(data_cerebrum, data_cerebellum);

    % Store coefficients
    alpha_all(ii) = alpha;
    beta_all(ii)  = beta;
    R2_all(ii)    = R2;

    % Print trial diagnostics
    fprintf('Trial %d: α = %.2f, β = %.2f, R² = %.2f\n', ii, alpha, beta, R2);
    fprintf('         |Entropy effect| = %.3f, |Linear effect| = %.3f\n\n', ...
        mean(abs(entropy_term)), mean(abs(linear_term)));

    % --- Sliding window analysis on predicted signal ---
    x = macro_hat(:)';
    T = length(x);
    n_windows = floor((T - window)/step);
    mu = zeros(1, n_windows);
    logvar = zeros(1, n_windows);

    for i = 1:n_windows
        idx = (i-1)*step + (1:window);
        xi = x(idx);
        mu(i) = mean(xi);
        logvar(i) = log(var(xi) + 1e-8);  % Avoid log(0)
    end

    % Allocate pooled matrix on first pass
    if isempty(time_vec)
        time_vec = (0:n_windows-1) * step / Fs;
        all_mu = zeros(numfiles, n_windows);
        all_logvar = zeros(numfiles, n_windows);
    end

    all_mu(ii,:) = mu;
    all_logvar(ii,:) = logvar;
end

% === Compute grand average and SEM ===
mu_mean = mean(all_mu, 1);
mu_sem  = std(all_mu, 0, 1) / sqrt(numfiles);
lv_mean = mean(all_logvar, 1);
lv_sem  = std(all_logvar, 0, 1) / sqrt(numfiles);

% Create upper and lower surfaces for SEM tube
X = [mu_mean + mu_sem; mu_mean - mu_sem];
Y = [time_vec; time_vec];
Z = [lv_mean + lv_sem; lv_mean - lv_sem];

% === Normalize each axis (mean 0, std 1) ===
x = mu_mean; y = time_vec; z = lv_mean;
x_n = (x - mean(x)) / std(x);
y_n = (y - mean(y)) / std(y);
z_n = (z - mean(z)) / std(z);

% Normalized curve (3 x N)
curve_n = [x_n; y_n; z_n];

% Estimate local arc length to set tube radius
dists = sqrt(sum(diff(curve_n, 1, 2).^2, 1));
avg_dist = mean(dists);
radius = 0.4 * avg_dist * ones(1, length(x_n));

% Smooth and interpolate curve and radius
curve_n_smooth = smoothdata(curve_n, 2, 'gaussian', 3);
curve_interp = interp1(1:size(curve_n_smooth,2), curve_n_smooth', ...
    linspace(1, size(curve_n_smooth,2), 200), 'pchip')';
radius_interp = interp1(1:length(radius), radius, linspace(1, length(radius), 200), 'linear');

% Generate tube surface in normalized space
[Xn, Yn, Zn] = tubeplot(curve_interp, radius_interp, 50);

% Denormalize for rendering
Xtube = Xn * std(x) + mean(x);
Ytube = Yn * std(y) + mean(y);
Ztube = Zn * std(z) + mean(z);

% Normalize to [0,1] for visualization
Xvis = (Xtube - min(Xtube(:))) / range(Xtube(:));
Yvis = (Ytube - min(Ytube(:))) / range(Ytube(:));
Zvis = (Ztube - min(Ztube(:))) / range(Ztube(:));

% === Plot ===
figure('Color','w');
surf(Xvis, Yvis, Zvis, ...
    'FaceColor', [1 0 0], ...
    'FaceAlpha', 0.8, ...
    'EdgeColor', 'none', ...
    'FaceLighting', 'gouraud', ...
    'AmbientStrength', 0.4, ...
    'SpecularStrength', 0.2, ...
    'DiffuseStrength', 0.8);
hold on;

camlight('headlight'); material('metal');

xlabel('Normalized Expectation');
ylabel('Normalized Time');
zlabel('Normalized Entropy Proxy');
title('Figure 9. Normalized 3D Tube (SEM Envelope)');
view(32, 20); grid on;
axis([0 1 0 1 0 1]);
set(gca, 'FontSize', 13);

% Export to PDF
exportgraphics(gcf, '~/Desktop/Fig_9.pdf', 'ContentType','vector');

% === Summary stats ===
fprintf('\n===== Summary Across Trials =====\n');
fprintf('alpha = %.2f ± %.2f\n', mean(alpha_all), std(alpha_all)/sqrt(numfiles));
fprintf('beta  = %.2f ± %.2f\n', mean(beta_all),  std(beta_all)/sqrt(numfiles));
fprintf('R²     = %.2f ± %.2f\n', mean(R2_all),    std(R2_all)/sqrt(numfiles));


% === Function: Transform macro via micro ===
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

    % Fit macro ~ micro + logq_x + (micro - mu_x)
    errfun = @(p) sum((macro - (micro + ...
        p(1)*(logq_x - logq_mean) + p(2)*(micro - mu_x))).^2);
    params = fminsearch(errfun, [0.1, 0.1]);

    alpha = params(1);
    beta  = params(2);

    entropy_term = alpha * (logq_x - logq_mean);
    linear_term  = beta  * (micro - mu_x);
    macro_hat = micro + entropy_term + linear_term;

    % Reverse direction fit (optional)
    [counts2, edges2] = histcounts(macro, 100, 'Normalization', 'pdf');
    bin_centers2 = 0.5*(edges2(1:end-1) + edges2(2:end));
    logq_y = interp1(bin_centers2, log(counts2 + 1e-8), macro, 'linear', 'extrap');

    logq_mean2 = mean(logq_y);
    mu_y = mean(macro);

    errfun_rev = @(p) sum((micro - (macro + ...
        p(1)*(logq_y - logq_mean2) + p(2)*(macro - mu_y))).^2);
    params_rev = fminsearch(errfun_rev, [0.1, 0.1]);

    micro_hat = macro + ...
        params_rev(1)*(logq_y - logq_mean2) + ...
        params_rev(2)*(macro - mu_y);

    % Compute R²
    R2    = 1 - sum((macro - macro_hat).^2) / sum((macro - mean(macro)).^2);
    R2rev = 1 - sum((micro - micro_hat).^2) / sum((micro - mean(micro)).^2);
end


% === Function: Generate tube around 3D curve ===
function [x, y, z] = tubeplot(curve, radius, n)
    % curve: 3 x N matrix
    % radius: scalar or 1 x N
    % n: number of angular segments (resolution of tube)
    if nargin < 3, n = 8; end

    npoints = size(curve, 2);
    theta = linspace(0, 2*pi, n+1);
    circle = [cos(theta); sin(theta)];  % Unit circle

    x = zeros(n+1, npoints);
    y = zeros(n+1, npoints);
    z = zeros(n+1, npoints);

    for i = 1:npoints
        % Compute tangent at each point
        if i == 1
            tangent = curve(:,2) - curve(:,1);
        elseif i == npoints
            tangent = curve(:,end) - curve(:,end-1);
        else
            tangent = curve(:,i+1) - curve(:,i-1);
        end
        tangent = tangent / norm(tangent);

        % Arbitrary vector not aligned with tangent
        not_tangent = [1; 0; 0];
        if abs(dot(not_tangent, tangent)) > 0.9
            not_tangent = [0; 1; 0];
        end

        % Orthonormal basis
        normal = cross(tangent, not_tangent);
        normal = normal / norm(normal);
        binormal = cross(tangent, normal);

        r = radius(i);

        for j = 1:n+1
            offset = r * (normal * circle(1,j) + binormal * circle(2,j));
            pt = curve(:,i) + offset;
            x(j,i) = pt(1);
            y(j,i) = pt(2);
            z(j,i) = pt(3);
        end
    end
end

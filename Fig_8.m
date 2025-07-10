% generate_fig8.m
% Figure 8: Cross-region prediction of neural signals

clear; clc; close all; warning('off','all');

%% === Parameters ===
numfiles = 28;
Fs = 99.8722;  % sampling rate (Hz)

% Initialize pooled signals across sessions
all_t = [];
all_cerebrum = [];
all_cerebellum = [];
all_cerebrum_hat = [];
all_cerebellum_hat = [];

%% === Main loop: load + predict ===
for ii = 1:numfiles
    % Load region-averaged traces
    load(['~/Dropbox/Work/AT_Calcium/mat/data-001_ratio_' num2str(ii) '.mat'], ...
        'data_cerebellum', 'data_cerebrum');

    T = length(data_cerebrum);
    t = (0:T-1) / Fs;

    % Forward prediction: cerebellum → cerebrum
    [~, ~, ~, ~, cerebrum_hat, ~] = test_transform_mv(data_cerebellum, data_cerebrum);

    % Reverse prediction: cerebrum → cerebellum
    [~, ~, ~, ~, ~, cerebellum_hat] = test_transform_mv(data_cerebrum, data_cerebellum);

    % Accumulate across sessions
    all_t = [all_t, t + (ii-1)*(T/Fs)];
    all_cerebrum = [all_cerebrum, data_cerebrum(:)'];
    all_cerebellum = [all_cerebellum, data_cerebellum(:)'];
    all_cerebrum_hat = [all_cerebrum_hat, cerebrum_hat(:)'];
    all_cerebellum_hat = [all_cerebellum_hat, cerebellum_hat(:)'];
end

%% === Plot ===
figure('Color','w', 'Position', [100, 100, 900, 600]);

% --- Forward: cerebellum → cerebrum ---
subplot(2,1,1)
plot(all_t, all_cerebrum, 'k-', 'DisplayName', 'True cerebrum'); hold on;
plot(all_t, all_cerebrum_hat, 'r--', 'DisplayName', 'Predicted from cerebellum');
xlabel('Time (s)');
ylabel('Fluorescence');
title('Fig. 8A. Cerebellum → Cerebrum');
legend('Location','northeast');
set(gca, 'FontSize', 14);

% --- Reverse: cerebrum → cerebellum ---
subplot(2,1,2)
plot(all_t, all_cerebellum, 'k-', 'DisplayName', 'True cerebellum'); hold on;
plot(all_t, all_cerebellum_hat, 'b--', 'DisplayName', 'Predicted from cerebrum');
xlabel('Time (s)');
ylabel('Fluorescence');
title('Fig. 8B. Cerebrum → Cerebellum');
legend('Location','northeast');
set(gca, 'FontSize', 14);

% Save as needed:
% exportgraphics(gcf, '~/Desktop/Fig8.pdf', 'ContentType', 'vector');

%% === Helper function ===
function [alpha, beta, R2, R2rev, macro_hat, micro_hat] = test_transform_mv(micro, macro)
    % Estimate transformation from micro → macro using entropy and expectation flows

    % --- Forward transformation (micro → macro) ---
    micro = micro(:);
    macro = macro(:);

    % Estimate log q(x) via histogram
    [counts, edges] = histcounts(micro, 100, 'Normalization', 'pdf');
    bin_centers = 0.5*(edges(1:end-1) + edges(2:end));
    logq_vals = log(counts + 1e-8);
    logq_x = interp1(bin_centers, logq_vals, micro, 'linear', 'extrap');

    logq_mean = mean(logq_x);
    mu_x = mean(micro);

    % Optimize alpha, beta to minimize prediction error
    opts = optimset('MaxFunEvals', 10000, 'MaxIter', 10000, 'Display', 'off');
    errfun = @(p) sum((macro - (micro + ...
        p(1)*(logq_x - logq_mean) + p(2)*(micro - mu_x))).^2);
    params = fminsearch(errfun, [0.1, 0.1], opts);
    alpha = params(1);
    beta  = params(2);

    % Predicted macro signal
    macro_hat = micro + alpha*(logq_x - logq_mean) + beta*(micro - mu_x);
    resid = macro - macro_hat;
    R2 = 1 - sum(resid.^2) / sum((macro - mean(macro)).^2);

    % --- Reverse transformation (macro → micro) ---
    [counts2, edges2] = histcounts(macro, 100, 'Normalization', 'pdf');
    bin_centers2 = 0.5*(edges2(1:end-1) + edges2(2:end));
    logq_y = interp1(bin_centers2, log(counts2 + 1e-8), macro, 'linear', 'extrap');
    logq_mean2 = mean(logq_y);
    mu_y = mean(macro);

    errfun_rev = @(p) sum((micro - (macro + ...
        p(1)*(logq_y - logq_mean2) + p(2)*(macro - mu_y))).^2);
    params_rev = fminsearch(errfun_rev, [0.1, 0.1], opts);

    % Predicted micro signal
    micro_hat = macro + ...
        params_rev(1)*(logq_y - logq_mean2) + ...
        params_rev(2)*(macro - mu_y);
    resid_rev = micro - micro_hat;
    R2rev = 1 - sum(resid_rev.^2) / sum((micro - mean(micro)).^2);
end

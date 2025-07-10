% Significance Testing of Directionality (R² vs R²rev)
% Compares predictive strength of cerebrum → cerebellum vs. cerebellum → cerebrum
% using pooled R² over all subjects and circular time shuffling as the null model

clear; clc; close all; warning('off','all');

% === Parameters ===
numfiles = 28;
Fs = 99.8722;                        % Sampling frequency
num_permutations = 1000;            % Surrogate iterations

% === Load all subject data ===
all_data_cerebrum = cell(numfiles, 1);
all_data_cerebellum = cell(numfiles, 1);

for ii = 1:numfiles
    load(['~/Dropbox/Work/AT_Calcium/mat/data-001_ratio_' num2str(ii) '.mat'], ...
         'data_cerebrum', 'data_cerebellum');
    all_data_cerebrum{ii}   = data_cerebrum(:);
    all_data_cerebellum{ii} = data_cerebellum(:);
end

% === Compute observed pooled R² and R²rev ===
all_macro = [];        % Cerebrum (predicted)
all_macro_hat = [];
all_micro = [];        % Cerebellum (predicted)
all_micro_hat = [];

for ii = 1:numfiles
    [~, ~, ~, ~, macro_hat, micro_hat] = ...
        test_transform_mv(all_data_cerebellum{ii}, all_data_cerebrum{ii});
    
    all_macro      = [all_macro; all_data_cerebrum{ii}];
    all_macro_hat  = [all_macro_hat; macro_hat];
    all_micro      = [all_micro; all_data_cerebellum{ii}];
    all_micro_hat  = [all_micro_hat; micro_hat];
end

% Observed R²: cerebellum → cerebrum
resid_macro = all_macro - all_macro_hat;
R2_observed = 1 - sum(resid_macro.^2) / sum((all_macro - mean(all_macro)).^2);

% Observed R²rev: cerebrum → cerebellum
resid_micro = all_micro - all_micro_hat;
R2rev_observed = 1 - sum(resid_micro.^2) / sum((all_micro - mean(all_micro)).^2);

% === Initialize null distributions ===
R2_null = zeros(num_permutations, 1);
R2rev_null = zeros(num_permutations, 1);
N = length(all_data_cerebrum{1});  % Assume same length across subjects

% === Generate null distributions ===
for p = 1:num_permutations
    disp(['Permutation ' num2str(p)]);

    all_macro_perm = [];
    all_macro_hat_perm = [];
    all_micro_perm = [];
    all_micro_hat_perm = [];

    for ii = 1:numfiles
        % Circularly shift one input per direction
        shifted_micro = circshift(all_data_cerebellum{ii}, randi(N-1));
        shifted_macro = circshift(all_data_cerebrum{ii}, randi(N-1));

        % Forward: cerebellum → cerebrum
        [~, ~, ~, ~, macro_hat_perm, ~] = ...
            test_transform_mv(shifted_micro, all_data_cerebrum{ii});
        all_macro_perm     = [all_macro_perm; all_data_cerebrum{ii}];
        all_macro_hat_perm = [all_macro_hat_perm; macro_hat_perm];

        % Reverse: cerebrum → cerebellum
        [~, ~, ~, ~, ~, micro_hat_rev] = ...
            test_transform_mv(all_data_cerebellum{ii}, shifted_macro);
        all_micro_perm     = [all_micro_perm; all_data_cerebellum{ii}];
        all_micro_hat_perm = [all_micro_hat_perm; micro_hat_rev];
    end

    % R² forward
    resid_perm = all_macro_perm - all_macro_hat_perm;
    R2_null(p) = 1 - sum(resid_perm.^2) / sum((all_macro_perm - mean(all_macro_perm)).^2);

    % R² reverse
    resid_rev_perm = all_micro_perm - all_micro_hat_perm;
    R2rev_null(p) = 1 - sum(resid_rev_perm.^2) / sum((all_micro_perm - mean(all_micro_perm)).^2);
end

% === Compute p-values ===
p_val_forward = mean(R2_null >= R2_observed);
p_val_reverse = mean(R2rev_null >= R2rev_observed);

fprintf('\n===== Significance Results =====\n');
fprintf('Observed R²:     %.4f, p = %.4f\n', R2_observed, p_val_forward);
fprintf('Observed R²rev:  %.4f, p = %.4f\n', R2rev_observed, p_val_reverse);


% === Helper function ===
function [alpha, beta, R2, R2rev, macro_hat, micro_hat] = test_transform_mv(micro, macro)
    % Apply transformation from micro → macro and reverse
    micro = micro(:);
    macro = macro(:);

    % Estimate logq(micro)
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

    resid = macro - macro_hat;
    R2 = 1 - sum(resid.^2) / sum((macro - mean(macro)).^2);

    % Reverse: macro → micro
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

    resid_rev = micro - micro_hat;
    R2rev = 1 - sum(resid_rev.^2) / sum((micro - mean(micro)).^2);
end

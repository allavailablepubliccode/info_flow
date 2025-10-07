%==========================================================================
% Table_S1.m
%
% Generate Supplementary Table S1:
% FDR-significant parameter estimates (R², α, β) across mice and regions.
%
%==========================================================================

clear; clc; close all;

%% === Load existing results ===
load('Fig6results.mat','R2full','alpha_mat','beta_mat','pvals');

region_names = {'AL','AM','LM','PM','RL','V1'};   % Region order used in analyses
alpha_FDR = 0.01;                            % FDR threshold

nMice = size(R2full,1);
nRegions = size(R2full,2);

%% === Apply FDR correction (off-diagonal only) ===
diag_mask = false(size(pvals));
for m = 1:nMice
    diag_mask(m,:,:) = eye(nRegions);
end

p_flat = pvals(:);
valid_idx = find(~diag_mask(:) & ~isnan(p_flat));

[p_fdr, passed] = fdr_bh(p_flat(valid_idx), alpha_FDR, 'pdep', 'yes');
sig_mask = false(size(pvals));
sig_mask(valid_idx) = passed;

%% === Collect results for significant pairs ===
results = [];
for ii = 1:nMice
    for jj = 1:nRegions
        for kk = 1:nRegions
            if sig_mask(ii,jj,kk)
                results = [results; ii, jj, kk, ...
                    R2full(ii,jj,kk), alpha_mat(ii,jj,kk), beta_mat(ii,jj,kk)];
            end
        end
    end
end

%% === Convert to table and add region names ===
if isempty(results)
    fprintf('No FDR-significant pairs found.\n');
else
    T = array2table(results, ...
        'VariableNames',{'Mouse','From','To','R2','Alpha','Beta'});

    % Convert numeric region indices to names
    FromNames = strings(height(T),1);
    ToNames   = strings(height(T),1);
    for r = 1:height(T)
        FromNames(r) = region_names{T.From(r)};
        ToNames(r)   = region_names{T.To(r)};
    end

    % Replace numeric indices with names for clarity
    T = removevars(T, {'From','To'});
    T.From = FromNames;
    T.To   = ToNames;

    % Reorder columns for readability
    T = movevars(T, {'From','To'}, 'Before', 'R2');

    %% === Display and save ===
    disp(T);
    writetable(T,'alpha_beta_table.csv');
    fprintf('\nSaved parameter table as alpha_beta_table.csv\n');
end

% Compute number of valid (non-NaN) entries
n_alpha = sum(~isnan(alpha_mat(:)));
n_beta  = sum(~isnan(beta_mat(:)));

% Compute means
mean_alpha = nanmean(alpha_mat(:));
mean_beta  = nanmean(beta_mat(:));

% Compute standard deviations
std_alpha = nanstd(alpha_mat(:));
std_beta  = nanstd(beta_mat(:));

% Compute standard errors
sem_alpha = std_alpha / sqrt(n_alpha);
sem_beta  = std_beta  / sqrt(n_beta);

fprintf('Alpha: mean = %.3e, SEM = %.3e (N = %d)\n', mean_alpha, sem_alpha, n_alpha);
fprintf('Beta : mean = %.3f,  SEM = %.3f  (N = %d)\n', mean_beta, sem_beta, n_beta);

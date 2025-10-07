function [R2, yhat, alpha, beta] = plaw_kde(data_in, data_out)
%==========================================================================
% plaw_kde: Fit nonlinear model with KDE-based entropy and linear terms
%
% Model:
%   y ≈ x + α * (logq(x) - mean(logq(x))) + β * (x - mean(x))
%
% Inputs:
%   data_in   - Input variable (e.g., V1 activity)
%   data_out  - Output variable to be predicted (e.g., AM activity)
%
% Outputs:
%   R2    - Coefficient of determination for model fit
%   yhat  - Model prediction of data_out
%   alpha - Coefficient for entropy-based term (log-density)
%   beta  - Coefficient for drift term (deviation from mean)
%
% Notes:
%   - Kernel density estimation is used to compute logq(x)
%   - Parameters (α, β) are optimized via bounded least-squares
%==========================================================================

% === 1. Estimate log-density logq(x) via KDE ===
[f, xi] = ksdensity(data_in);                       % Estimate PDF
logq = log(f + 1e-8);                               % Avoid log(0)
logq_x = interp1(xi, logq, data_in, 'linear', 'extrap');  % Interpolate to original x

% === 2. Precompute means for centerings ===
logq_mean = mean(logq_x);
mu_x = mean(data_in);

% === 3. Define loss function for optimization ===
errfun = @(p) sum((data_out - ...
    (data_in + p(1)*(logq_x - logq_mean) + p(2)*(data_in - mu_x))).^2);

% === 4. Optimize alpha and beta ===
params0 = [0, 0];                 % Initial guess
lb = [-5, -20];                   % Lower bounds
ub = [5,  20];                    % Upper bounds
opts = optimoptions('fmincon', ...
    'Display', 'off', ...
    'Algorithm', 'interior-point');

params = fmincon(errfun, params0, [],[],[],[], lb, ub, [], opts);
alpha = params(1);
beta  = params(2);

% === 5. Compute final prediction and R² ===
ent_term = alpha * (logq_x - logq_mean);
lin_term = beta  * (data_in - mu_x);
yhat = data_in + ent_term + lin_term;

resid = data_out - yhat;
R2 = 1 - sum(resid.^2) / sum((data_out - mean(data_out)).^2);
end

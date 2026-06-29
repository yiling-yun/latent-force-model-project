function [xmin, fmin] = purecmaes(fitfun, xstart, sigma, maxevals)
% PURECMAES  CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
% Derivative-free optimiser for non-smooth/noisy objectives.
% Based on Hansen's purecmaes tutorial implementation (2011/2016).
%
% Usage:
%   [xmin, fmin] = purecmaes(fitfun, xstart, sigma, maxevals)
%
% Inputs:
%   fitfun   - function handle, accepts row or column vector, returns scalar
%   xstart   - initial point (row or column vector, length N)
%   sigma    - initial step size (scalar); set to ~1/3 of expected search range
%   maxevals - maximum number of function evaluations
%
% Outputs:
%   xmin  - best solution found (row vector)
%   fmin  - objective value at xmin

xstart = xstart(:);   % ensure column vector
N = numel(xstart);

% --- Selection parameters ---
lambda  = 4 + floor(3 * log(N));          % offspring per generation
mu      = floor(lambda / 2);              % number of parents selected
w       = log(mu + 0.5) - log(1:mu)';    % recombination weights (unnormalised)
w       = w / sum(w);                     % normalise
mueff   = 1 / sum(w.^2);                 % effective selection mass

% --- Adaptation parameters ---
cc    = (4 + mueff/N) / (N + 4 + 2*mueff/N);          % cumulation for C
cs    = (mueff + 2) / (N + mueff + 5);                 % cumulation for sigma
c1    = 2 / ((N + 1.3)^2 + mueff);                    % rank-1 update for C
cmu   = min(1 - c1, ...
            2*(mueff - 2 + 1/mueff) / ((N+2)^2 + mueff));  % rank-mu update
damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1)) - 1) + cs; % step-size damping

% --- Initialise internal state ---
xmean     = xstart;
pc        = zeros(N, 1);   % evolution path for C
ps        = zeros(N, 1);   % evolution path for sigma
B         = eye(N);        % eigenvectors of C
D         = ones(N, 1);    % sqrt of eigenvalues of C
C         = eye(N);        % covariance matrix
invsqrtC  = eye(N);        % C^(-1/2)
eigeneval = 0;
chiN      = N^0.5 * (1 - 1/(4*N) + 1/(21*N^2));  % expectation of ||N(0,I)||

% --- Bookkeeping ---
counteval = 0;
fmin      = inf;
xmin      = xstart;

% --- Main loop ---
while counteval < maxevals

    % Generate and evaluate lambda offspring
    arx       = zeros(N, lambda);
    arfitness = zeros(1, lambda);
    for k = 1:lambda
        arx(:, k)    = xmean + sigma * B * (D .* randn(N, 1));
        arfitness(k) = feval(fitfun, arx(:, k)');
        counteval    = counteval + 1;
        if arfitness(k) < fmin
            fmin = arfitness(k);
            xmin = arx(:, k);
        end
        if counteval >= maxevals, break; end
    end
    if counteval >= maxevals, break; end

    % Sort offspring by fitness (ascending)
    [~, arindex] = sort(arfitness);

    % Update mean (weighted recombination of top-mu)
    xold  = xmean;
    xmean = arx(:, arindex(1:mu)) * w;

    % Update evolution path for sigma (ps)
    ps = (1 - cs) * ps + ...
         sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma;
    hsig = (norm(ps) / sqrt(1 - (1-cs)^(2*counteval/lambda)) / chiN) < (1.4 + 2/(N+1));

    % Update evolution path for C (pc)
    pc = (1 - cc) * pc + ...
         hsig * sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma;

    % Adapt covariance matrix C
    artmp = (1/sigma) * (arx(:, arindex(1:mu)) - repmat(xold, 1, mu));
    C = (1 - c1 - cmu) * C ...
        + c1 * (pc * pc' + (1 - hsig) * cc * (2 - cc) * C) ...
        + cmu * artmp * diag(w) * artmp';

    % Adapt step size sigma
    sigma = sigma * exp((cs / damps) * (norm(ps)/chiN - 1));

    % Update B and D from C every ~lambda/(c1+cmu)/N/10 evaluations
    if counteval - eigeneval > lambda / (c1 + cmu) / N / 10
        eigeneval = counteval;
        C = triu(C) + triu(C, 1)';       % enforce symmetry
        [B, Dmat] = eig(C);
        D = sqrt(max(diag(Dmat), 0));    % ensure non-negative
        invsqrtC = B * diag(D.^(-1)) * B';
    end

    % --- Stopping criteria ---
    if max(D) > 1e7 * min(D), break; end          % ill-conditioned covariance
    if arfitness(arindex(1)) == arfitness(arindex(min(mu+1,lambda))), break; end  % flat fitness
end

xmin = xmin';  % return as row vector

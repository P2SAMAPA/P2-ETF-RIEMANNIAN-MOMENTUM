"""
Riemannian geometry on the SPD manifold.
Computes Fréchet mean and manifold momentum using the AIRM metric.
"""

import numpy as np
import pandas as pd
from scipy.linalg import logm, expm, fractional_matrix_power

def spd_logm(X):
    """Matrix logarithm for SPD matrix X."""
    return logm(X)

def spd_expm(X):
    """Matrix exponential."""
    return expm(X)

def riemannian_distance(A, B):
    """Affine-invariant Riemannian metric distance."""
    A_inv = np.linalg.inv(A)
    prod = A_inv @ B
    log_prod = logm(prod)
    return np.sqrt(np.trace(log_prod @ log_prod))

def riemannian_mean(cov_mats, tol=1e-6, max_iter=50):
    """Compute Fréchet mean of a list of SPD matrices using iterative gradient descent."""
    n = len(cov_mats)
    if n == 1:
        return cov_mats[0].copy()
    
    mean = cov_mats[0].copy()
    for _ in range(max_iter):
        grad_sum = np.zeros_like(mean)
        for S in cov_mats:
            # Gradient of squared distance w.r.t mean: -2 * mean * logm(mean^{-1} S)
            mean_inv = np.linalg.inv(mean)
            grad_sum += logm(mean_inv @ S)
        grad = mean @ grad_sum / n
        mean_new = mean @ expm(-grad)
        if riemannian_distance(mean, mean_new) < tol:
            break
        mean = mean_new
    return mean

def log_map(base, X):
    """Logarithm map from base to X on SPD manifold: base^{1/2} * logm(base^{-1/2} X base^{-1/2}) * base^{1/2}."""
    base_sqrt = fractional_matrix_power(base, 0.5)
    base_inv_sqrt = fractional_matrix_power(base, -0.5)
    inner = base_inv_sqrt @ X @ base_inv_sqrt
    return base_sqrt @ logm(inner) @ base_sqrt

def exp_map(base, V):
    """Exponential map: base^{1/2} * expm(base^{-1/2} V base^{-1/2}) * base^{1/2}."""
    base_sqrt = fractional_matrix_power(base, 0.5)
    base_inv_sqrt = fractional_matrix_power(base, -0.5)
    inner = base_inv_sqrt @ V @ base_inv_sqrt
    return base_sqrt @ expm(inner) @ base_sqrt

class RiemannianMomentum:
    def __init__(self, cov_window=63, frechet_window=21, momentum_lookback=5):
        self.cov_window = cov_window
        self.frechet_window = frechet_window
        self.momentum_lookback = momentum_lookback

    def compute_covariance_sequence(self, returns: pd.DataFrame) -> list:
        """Compute rolling covariance matrices (SPD) over cov_window days."""
        covs = []
        for i in range(self.cov_window, len(returns) + 1):
            window = returns.iloc[i-self.cov_window:i]
            cov = window.cov().values
            # Regularize to ensure positive definiteness
            cov = cov + np.eye(cov.shape[0]) * 1e-6
            covs.append(cov)
        return covs

    def compute_manifold_momentum(self, returns: pd.DataFrame) -> dict:
        """
        Compute manifold momentum direction.
        Returns:
            - tangent_vector: vectorized tangent direction at current point
            - scores: projection of each ETF's recent return onto the manifold direction
        """
        if len(returns) < self.cov_window + self.frechet_window:
            return {'tangent_vector': None, 'scores': {}}

        covs = self.compute_covariance_sequence(returns)
        if len(covs) < self.frechet_window + self.momentum_lookback:
            return {'tangent_vector': None, 'scores': {}}

        # Current covariance (most recent)
        current_cov = covs[-1]
        # Fréchet mean of the past frechet_window covariances (excluding the latest few to avoid overlap)
        baseline_covs = covs[-self.frechet_window-self.momentum_lookback:-self.momentum_lookback]
        frechet_mean = riemannian_mean(baseline_covs)

        # Covariance from momentum_lookback days ago
        past_cov = covs[-self.momentum_lookback-1]

        # Tangent vector at past point pointing to current
        tangent = log_map(past_cov, current_cov)
        # Parallel transport to the Frechet mean (simplified: use tangent at mean pointing to current)
        tangent_at_mean = log_map(frechet_mean, current_cov)

        # Flatten tangent for scoring
        flat_tangent = tangent_at_mean.flatten()

        # Compute ETF scores: correlation between ETF returns and tangent direction
        tickers = returns.columns.tolist()
        scores = {}
        recent_returns = returns.iloc[-self.momentum_lookback:].values  # (lookback, n_assets)
        if recent_returns.shape[1] != len(tickers):
            return {'tangent_vector': flat_tangent.tolist(), 'scores': {}}

        # Use the first eigenvector of the tangent as the direction (since tangent is a matrix)
        # Simplified: project asset returns onto the dominant eigenvector of the tangent
        eigvals, eigvecs = np.linalg.eigh(tangent_at_mean)
        dominant_vec = eigvecs[:, np.argmax(np.abs(eigvals))]

        for i, ticker in enumerate(tickers):
            # Score = correlation between recent returns and dominant manifold direction
            ret_series = recent_returns[:, i]
            if np.std(ret_series) > 1e-6:
                corr = np.corrcoef(ret_series, dominant_vec[:len(tickers)])[0, 1]
            else:
                corr = 0.0
            scores[ticker] = float(corr)

        return {'tangent_vector': flat_tangent.tolist(), 'scores': scores}

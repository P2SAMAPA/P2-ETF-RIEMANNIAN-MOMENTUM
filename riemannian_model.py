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
            mean_inv = np.linalg.inv(mean)
            grad_sum += logm(mean_inv @ S)
        grad = mean @ grad_sum / n
        mean_new = mean @ expm(-grad)
        if riemannian_distance(mean, mean_new) < tol:
            break
        mean = mean_new
    return mean

def log_map(base, X):
    """Logarithm map from base to X on SPD manifold."""
    base_sqrt = fractional_matrix_power(base, 0.5)
    base_inv_sqrt = fractional_matrix_power(base, -0.5)
    inner = base_inv_sqrt @ X @ base_inv_sqrt
    return base_sqrt @ logm(inner) @ base_sqrt

def exp_map(base, V):
    """Exponential map."""
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
            cov = cov + np.eye(cov.shape[0]) * 1e-6
            covs.append(cov)
        return covs

    def compute_manifold_momentum(self, returns: pd.DataFrame) -> dict:
        """
        Compute manifold momentum direction.
        Returns:
            - tangent_vector: vectorized tangent direction at current point
            - scores: each ETF's loading on the dominant manifold direction
        """
        if len(returns) < self.cov_window + self.frechet_window:
            return {'tangent_vector': None, 'scores': {}}

        covs = self.compute_covariance_sequence(returns)
        if len(covs) < self.frechet_window + self.momentum_lookback:
            return {'tangent_vector': None, 'scores': {}}

        current_cov = covs[-1]
        baseline_covs = covs[-self.frechet_window-self.momentum_lookback:-self.momentum_lookback]
        frechet_mean = riemannian_mean(baseline_covs)
        past_cov = covs[-self.momentum_lookback-1]

        tangent = log_map(past_cov, current_cov)
        tangent_at_mean = log_map(frechet_mean, current_cov)

        flat_tangent = tangent_at_mean.flatten()

        tickers = returns.columns.tolist()
        scores = {}

        # Use the dominant eigenvector of the tangent matrix as the manifold direction
        eigvals, eigvecs = np.linalg.eigh(tangent_at_mean)
        dominant_vec = eigvecs[:, np.argmax(np.abs(eigvals))]

        # Each ETF's score is its loading on the dominant direction
        for i, ticker in enumerate(tickers):
            if i < len(dominant_vec):
                scores[ticker] = float(dominant_vec[i])
            else:
                scores[ticker] = 0.0

        return {'tangent_vector': flat_tangent.tolist(), 'scores': scores}

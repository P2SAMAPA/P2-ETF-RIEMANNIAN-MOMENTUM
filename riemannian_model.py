"""
Riemannian geometry on the SPD manifold.
Computes Fréchet mean and manifold momentum using the AIRM metric.
Includes bootstrapping and multi‑horizon averaging.
"""

import numpy as np
import pandas as pd
from scipy.linalg import logm, expm, fractional_matrix_power
from sklearn.utils import resample

def riemannian_distance(A, B):
    """Affine-invariant Riemannian metric distance."""
    A_inv = np.linalg.inv(A)
    prod = A_inv @ B
    log_prod = logm(prod)
    return np.sqrt(np.trace(log_prod @ log_prod))

def riemannian_mean(cov_mats, tol=1e-8, max_iter=100):
    """Compute Fréchet mean of a list of SPD matrices."""
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

class RiemannianMomentum:
    def __init__(self, cov_window=126, frechet_window=63, momentum_lookbacks=[5,10,21],
                 n_bootstrap=50, frechet_max_iter=100, frechet_tol=1e-8):
        self.cov_window = cov_window
        self.frechet_window = frechet_window
        self.momentum_lookbacks = momentum_lookbacks
        self.n_bootstrap = n_bootstrap
        self.frechet_max_iter = frechet_max_iter
        self.frechet_tol = frechet_tol

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
        Compute manifold momentum direction using multi‑horizon averaging and bootstrapping.
        """
        if len(returns) < self.cov_window + self.frechet_window + max(self.momentum_lookbacks):
            return {'tangent_vector': None, 'scores': {}, 'confidence_intervals': {}}

        covs = self.compute_covariance_sequence(returns)
        tickers = returns.columns.tolist()
        n_assets = len(tickers)

        # Average scores over multiple lookbacks
        all_scores = {t: [] for t in tickers}
        tangent_vectors = []

        for lookback in self.momentum_lookbacks:
            if len(covs) < self.frechet_window + lookback:
                continue

            current_cov = covs[-1]
            baseline_covs = covs[-self.frechet_window-lookback:-lookback]
            frechet_mean = riemannian_mean(baseline_covs, tol=self.frechet_tol, max_iter=self.frechet_max_iter)
            tangent_at_mean = log_map(frechet_mean, current_cov)

            # Bootstrap to get confidence intervals
            boot_scores = {t: [] for t in tickers}
            for _ in range(self.n_bootstrap):
                boot_idx = resample(range(len(baseline_covs)), n_samples=len(baseline_covs))
                boot_baseline = [baseline_covs[i] for i in boot_idx]
                boot_mean = riemannian_mean(boot_baseline, tol=self.frechet_tol, max_iter=self.frechet_max_iter//2)
                boot_tangent = log_map(boot_mean, current_cov)
                eigvals, eigvecs = np.linalg.eigh(boot_tangent)
                dominant = eigvecs[:, np.argmax(np.abs(eigvals))]
                for i, t in enumerate(tickers):
                    if i < len(dominant):
                        boot_scores[t].append(dominant[i])

            # Compute mean scores for this lookback
            eigvals, eigvecs = np.linalg.eigh(tangent_at_mean)
            dominant = eigvecs[:, np.argmax(np.abs(eigvals))]
            for i, t in enumerate(tickers):
                if i < len(dominant):
                    all_scores[t].append(dominant[i])

            tangent_vectors.append(tangent_at_mean.flatten())

        # Average scores across lookbacks
        final_scores = {}
        final_ci = {}
        for t in tickers:
            if all_scores[t]:
                final_scores[t] = float(np.mean(all_scores[t]))
                final_ci[t] = {
                    'lower': float(np.percentile(all_scores[t], 2.5)),
                    'upper': float(np.percentile(all_scores[t], 97.5))
                }
            else:
                final_scores[t] = 0.0
                final_ci[t] = {'lower': 0.0, 'upper': 0.0}

        # Average tangent vector across lookbacks
        avg_tangent = np.mean(tangent_vectors, axis=0).tolist() if tangent_vectors else None

        return {
            'tangent_vector': avg_tangent,
            'scores': final_scores,
            'confidence_intervals': final_ci
        }

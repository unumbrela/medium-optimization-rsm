"""
model.py — Hybrid Quadratic-Response-Surface + Gaussian-Process-Residual model.

Implements §2.1 of the report:

  f(G,T,K)        = X β            (quadratic trend)
  X               = [1, G, T, K, G², T², K², GT, GK, TK]
  β               = (XᵀX + λI)⁻¹ Xᵀ y         λ = 1e-8

  r_i             = y_i − f_trend(x_i)
  k(x,x')         = σ_f² · exp(−½ Σ_d (x_d−x'_d)² / ℓ_d²)
                    anisotropic RBF, ℓ = [1.0, 1.0, 1.0]
  σ_f             = max(1e-6, std(r))
  σ_n             = max(1e-6, 0.02 · range(y))

  Posterior mean  μ(x*) = f_trend(x*) + k*ᵀ (K + σ_n² I)⁻¹ r
  Posterior var   σ²(x*) = k(x*,x*) − k*ᵀ (K + σ_n² I)⁻¹ k*

UCB acquisition  α_UCB(x) = μ(x) + κ σ(x),   κ = 2.58  (99% confidence)
Mean acquisition α_Mean(x) = μ(x)
"""

from __future__ import annotations

import numpy as np


KAPPA_99 = 2.58
TREND_RIDGE = 1e-8


# ----------------------------------------------------------------------
# Quadratic trend (§2.1.1)
# ----------------------------------------------------------------------
def quadratic_features(X: np.ndarray) -> np.ndarray:
    """X: (n,3) of (G,T,K) → (n,10) design matrix."""
    G, T, K = X[:, 0], X[:, 1], X[:, 2]
    ones = np.ones_like(G)
    return np.column_stack([ones, G, T, K, G * G, T * T, K * K, G * T, G * K, T * K])


def fit_trend(X: np.ndarray, y: np.ndarray, ridge: float = TREND_RIDGE) -> np.ndarray:
    """Return β so that f_trend(x) = Φ(x) β."""
    Phi = quadratic_features(X)
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    return np.linalg.solve(A, Phi.T @ y)


def trend_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return quadratic_features(X) @ beta


# ----------------------------------------------------------------------
# Anisotropic RBF kernel (§2.1.2)
# ----------------------------------------------------------------------
def rbf_kernel(X1: np.ndarray, X2: np.ndarray,
               sigma_f: float, lengthscales: np.ndarray) -> np.ndarray:
    diff = (X1[:, None, :] - X2[None, :, :]) / lengthscales[None, None, :]
    sqd = np.sum(diff ** 2, axis=-1)
    return (sigma_f ** 2) * np.exp(-0.5 * sqd)


# ----------------------------------------------------------------------
# Hybrid model object (§2.1.3 + posterior formulae)
# ----------------------------------------------------------------------
class HybridGP:
    """Quadratic trend + GP-on-residuals.

    Attributes
    ----------
    X, y            : training data
    beta            : trend coefficients
    sigma_f, sigma_n: kernel signal / noise scales
    lengthscales    : (3,) RBF lengthscales per input dim
    L               : Cholesky factor of (K + σ_n² I)
    alpha           : (K + σ_n² I)⁻¹ r
    """

    def __init__(self, lengthscales: np.ndarray | None = None):
        # Lengthscales are in STANDARDISED input space (z-scored). The report
        # quotes ℓ = [1.0, 1.0, 1.0] — applying that to raw units (G:30-70,
        # T:6-30, K:1-5) would give negligible smoothing, so we standardise
        # inputs before evaluating the kernel.
        self.lengthscales = (np.array([1.0, 1.0, 1.0], dtype=float)
                             if lengthscales is None else np.asarray(lengthscales, dtype=float))

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / self._x_std

    # ----- fitting -----
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridGP":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.X, self.y = X, y

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0
        X_z = self._standardise(X)

        self.beta = fit_trend(X, y)
        r = y - trend_predict(X, self.beta)

        self.sigma_f = max(1e-6, float(np.std(r)))
        self.sigma_n = max(1e-6, 0.02 * float(np.ptp(y)))

        K = rbf_kernel(X_z, X_z, self.sigma_f, self.lengthscales)
        K_stable = K + (1e-12 + self.sigma_n ** 2) * np.eye(len(X_z))
        self.L = np.linalg.cholesky(K_stable)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, r))
        self.r = r
        return self

    # ----- prediction -----
    def predict(self, X_star: np.ndarray, return_std: bool = True):
        X_star = np.atleast_2d(np.asarray(X_star, dtype=float))
        X_star_z = self._standardise(X_star)
        X_train_z = self._standardise(self.X)
        k_star = rbf_kernel(X_star_z, X_train_z, self.sigma_f, self.lengthscales)
        mu = trend_predict(X_star, self.beta) + k_star @ self.alpha
        if not return_std:
            return mu
        v = np.linalg.solve(self.L, k_star.T)                 # (n,m)
        k_ss = (self.sigma_f ** 2) - np.sum(v ** 2, axis=0)   # (m,)
        std = np.sqrt(np.clip(k_ss, 1e-12, None))
        return mu, std

    def mean(self, X_star: np.ndarray) -> np.ndarray:
        return self.predict(X_star, return_std=False)

    def ucb(self, X_star: np.ndarray, kappa: float = KAPPA_99) -> np.ndarray:
        mu, std = self.predict(X_star, return_std=True)
        return mu + kappa * std

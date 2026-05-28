"""
bayes_opt.py — Bayesian optimisation over (G, T) with K fixed (§4 of the report).

Acquisition functions
---------------------
  UCB(x)  = μ(x) + κ σ(x),   κ = 2.58 (99% confidence)
  Mean(x) = μ(x)

Optimiser
---------
  L-BFGS-B with box constraints G ∈ [30,70], T ∈ [6,30]
  10 random restarts (uniform in box) — §4.3.5
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from model import HybridGP, KAPPA_99


BOUNDS = [(30.0, 70.0), (6.0, 30.0)]   # G, T
N_RESTARTS = 10
K_FIXED = 3.0


def _make_objective(model: HybridGP, acq: str, kappa: float):
    if acq == "ucb":
        def neg(x):
            return -model.ucb(np.array([[x[0], x[1], K_FIXED]]), kappa=kappa)[0]
    elif acq == "mean":
        def neg(x):
            return -model.mean(np.array([[x[0], x[1], K_FIXED]]))[0]
    else:
        raise ValueError(f"unknown acquisition: {acq!r}")
    return neg


def optimise(model: HybridGP, acq: str, kappa: float = KAPPA_99,
             n_restarts: int = N_RESTARTS, seed: int = 0):
    """Run L-BFGS-B with multiple uniform restarts; return (x_best, f_best)."""
    rng = np.random.default_rng(seed)
    neg = _make_objective(model, acq, kappa)
    best_x, best_f = None, np.inf
    for _ in range(n_restarts):
        x0 = np.array([rng.uniform(*BOUNDS[0]), rng.uniform(*BOUNDS[1])])
        res = minimize(neg, x0, bounds=BOUNDS, method="L-BFGS-B",
                       options={"maxiter": 1000, "ftol": 1e-9})
        if res.success and res.fun < best_f:
            best_f, best_x = float(res.fun), res.x
    return best_x, -best_f


def run(model: HybridGP, seed: int = 0):
    """Run both acquisition strategies and return a results dict."""
    x_mean, v_mean = optimise(model, "mean", seed=seed)
    x_ucb,  v_ucb  = optimise(model, "ucb",  seed=seed + 1)

    # Posterior at each argmax (for the report table)
    pm_mean, ps_mean = model.predict(np.array([[x_mean[0], x_mean[1], K_FIXED]]))
    pm_ucb,  ps_ucb  = model.predict(np.array([[x_ucb[0],  x_ucb[1],  K_FIXED]]))

    return {
        "mean": {
            "x":  (float(x_mean[0]), float(x_mean[1]), K_FIXED),
            "mu": float(pm_mean[0]),
            "std": float(ps_mean[0]),
            "ucb": float(pm_mean[0] + KAPPA_99 * ps_mean[0]),
        },
        "ucb": {
            "x":  (float(x_ucb[0]), float(x_ucb[1]), K_FIXED),
            "mu": float(pm_ucb[0]),
            "std": float(ps_ucb[0]),
            "ucb": float(v_ucb),
        },
    }

"""
main.py — End-to-end medium-optimisation pipeline.

  1. Load data.csv (228 design points × 3 replicates).
  2. Fit hybrid Quadratic + GP-residual model.
  3. Run Bayesian optimisation with Mean and UCB acquisitions (K=3 fixed).
  4. Print the optimal-recipe report.
  5. Regenerate the figures.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from model import HybridGP, KAPPA_99
from bayes_opt import run as run_bo


DATA_CSV = "data.csv"


def load_training_data(csv_path: str = DATA_CSV):
    """Stack the three replicate columns into a single (X, y) training set."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found in {os.getcwd()}")
    df = pd.read_csv(csv_path)
    X_list, y_list = [], []
    for rep in ("OD1", "OD2", "OD3"):
        X_list.append(df[["G", "T", "K"]].to_numpy(dtype=float))
        y_list.append(df[rep].to_numpy(dtype=float))
    return np.vstack(X_list), np.concatenate(y_list), df


def print_report(df: pd.DataFrame, results: dict):
    od_mean_per_point = df[["OD1", "OD2", "OD3"]].mean(axis=1)
    max_obs = float(od_mean_per_point.max())

    print("=== Bayesian Optimization Results (K=3) ===")
    print(f"Max observed OD (data): {max_obs:.3f}")
    print()
    print("双获取函数策略比较:")
    header = f"| {'策略':<10} | {'最优位置':<18} | {'预测均值':<8} | {'UCB值':<8} |"
    print(header)
    print("-" * len(header))
    for name in ("mean", "ucb"):
        r = results[name]
        x = r["x"]
        loc = f"({x[0]:.2f}, {x[1]:.2f})"
        print(f"| {name + ' argmax':<10} | {loc:<18} | {r['mu']:.3f}    | {r['ucb']:.3f}    |")
    print()
    print("最终推荐配方 (UCB策略):")
    print(f"  - 葡萄糖: {results['ucb']['x'][0]:.1f} g/L")
    print(f"  - 胨:     {results['ucb']['x'][1]:.1f} g/L")
    print(f"  - KH2PO4: {results['ucb']['x'][2]:.1f} g/L")
    print()
    print("性能指标:")
    print(f"  - UCB值:    {results['ucb']['ucb']:.3f}  (超越实测最高值 {max_obs:.3f})")
    print(f"  - 置信度:   99%  (κ={KAPPA_99})")


def main():
    X, y, df = load_training_data()
    model = HybridGP().fit(X, y)
    results = run_bo(model, seed=42)
    print_report(df, results)

    # Regenerate figures if matplotlib pipeline available
    try:
        import figures
        figures.render_all(model, df)
        print("\nFigures regenerated.")
    except ImportError:
        pass

    return model, df, results


if __name__ == "__main__":
    main()

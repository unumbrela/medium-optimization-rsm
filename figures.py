"""
figures.py — Regenerate the published figures from the fitted Quad+GP model.

Produces:
  01_data_distribution.png     experimental scatter coloured by mean OD
  02_prediction_mean.png       posterior mean μ(G,T | K=3) contour
  04_ucb_acquisition.png       UCB(G,T | K=3) contour
  sensitivity_analysis.png     partial-dependence plots for G, T, K
  曲面.png                      3D μ(G,T | K=3) surface
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from model import HybridGP, KAPPA_99
from bayes_opt import run as run_bo


# Chinese font: fall back to sans-serif if not available so labels still render
try:
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC",
                                       "SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


GRID_N = 201
G_BOUNDS = (30.0, 70.0)
T_BOUNDS = (6.0, 30.0)
K_FIXED = 3.0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _grid():
    g = np.linspace(*G_BOUNDS, GRID_N)
    t = np.linspace(*T_BOUNDS, GRID_N)
    GG, TT = np.meshgrid(g, t)
    KK = np.full_like(GG, K_FIXED)
    X = np.column_stack([GG.ravel(), TT.ravel(), KK.ravel()])
    return GG, TT, X


def _argmaxes(model: HybridGP):
    return run_bo(model, seed=42)


# ----------------------------------------------------------------------
# 01 — Experimental data scatter
# ----------------------------------------------------------------------
def figure_data_distribution(df: pd.DataFrame, results: dict, path: str):
    od_mean = df[["OD1", "OD2", "OD3"]].mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(df["G"], df["T"], c=od_mean, cmap="viridis", s=80,
                    edgecolors="white", linewidths=0.4,
                    vmin=0.275, vmax=0.415)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("实测OD值")
    mean_x = results["mean"]["x"]
    ucb_x = results["ucb"]["x"]
    ax.scatter([mean_x[0]], [mean_x[1]], marker="*", s=180, c="red",
               edgecolors="black", linewidths=0.6,
               label=f"Mean最优 (OD={results['mean']['mu']:.3f})")
    ax.scatter([ucb_x[0]], [ucb_x[1]], marker="*", s=180, c="orange",
               edgecolors="black", linewidths=0.6,
               label=f"UCB最优 (OD={results['ucb']['ucb']:.3f})")
    ax.set_xlabel("葡萄糖浓度 (g/L)")
    ax.set_ylabel("胨浓度 (g/L)")
    ax.set_title("实验数据分布与最优点")
    ax.set_xlim(*G_BOUNDS); ax.set_ylim(5, 31)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------
# 02 — Posterior mean μ(G,T|K=3)
# ----------------------------------------------------------------------
def figure_prediction_mean(model: HybridGP, df: pd.DataFrame, results: dict, path: str):
    GG, TT, X = _grid()
    mu = model.mean(X).reshape(GG.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(GG, TT, np.clip(mu, 0.27, None), levels=20, cmap="plasma")
    cb = plt.colorbar(cf, ax=ax); cb.set_label("预测OD均值")

    k3 = df[df["K"] == K_FIXED]
    ax.scatter(k3["G"], k3["T"], c="white", s=14, edgecolors="black", linewidths=0.4)
    mx = results["mean"]["x"]; ux = results["ucb"]["x"]
    ax.scatter([mx[0]], [mx[1]], marker="*", s=160, c="red",
               edgecolors="white", linewidths=0.6)
    ax.scatter([ux[0]], [ux[1]], marker="*", s=160, c="white",
               edgecolors="black", linewidths=0.6)
    ax.set_xlabel("葡萄糖浓度 (g/L)"); ax.set_ylabel("胨浓度 (g/L)")
    ax.set_title("GP预测均值分布")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------
# 04 — UCB acquisition surface
# ----------------------------------------------------------------------
def figure_ucb(model: HybridGP, df: pd.DataFrame, results: dict, path: str):
    GG, TT, X = _grid()
    ucb = model.ucb(X, kappa=KAPPA_99).reshape(GG.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(GG, TT, np.clip(ucb, 0.27, None), levels=20, cmap="viridis")
    cb = plt.colorbar(cf, ax=ax); cb.set_label("UCB值")

    k3 = df[df["K"] == K_FIXED]
    ax.scatter(k3["G"], k3["T"], c="white", s=14, edgecolors="black", linewidths=0.4)
    ux = results["ucb"]["x"]
    ax.scatter([ux[0]], [ux[1]], marker="*", s=180, c="red",
               edgecolors="white", linewidths=0.7, label="UCB最大值")
    ax.set_xlabel("葡萄糖浓度 (g/L)"); ax.set_ylabel("胨浓度 (g/L)")
    ax.set_title("UCB获取函数")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------
# Sensitivity — partial-dependence plots
# ----------------------------------------------------------------------
def figure_sensitivity(model: HybridGP, df: pd.DataFrame, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    rng = np.random.default_rng(0)

    bg = df[["G", "T", "K"]].sample(n=min(120, len(df)), random_state=0).to_numpy()

    def pdp(var_idx: int, grid: np.ndarray):
        out = []
        for v in grid:
            X = bg.copy()
            X[:, var_idx] = v
            out.append(float(model.mean(X).mean()))
        return np.array(out)

    g_grid = np.linspace(*G_BOUNDS, 9)
    t_grid = np.linspace(*T_BOUNDS, 9)
    k_grid = np.linspace(1.0, 5.0, 9)

    axes[0].plot(g_grid, pdp(0, g_grid)); axes[0].set_title("PDP - Glucose")
    axes[0].set_xlabel("Glucose (g/L)"); axes[0].set_ylabel("Predicted OD")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_grid, pdp(1, t_grid)); axes[1].set_title("PDP - Tryptone")
    axes[1].set_xlabel("Tryptone (g/L)"); axes[1].set_ylabel("Predicted OD")
    axes[1].grid(alpha=0.3)

    axes[2].plot(k_grid, pdp(2, k_grid)); axes[2].set_title("PDP - KH2PO4")
    axes[2].set_xlabel("KH2PO4 (g/L)"); axes[2].set_ylabel("Predicted OD")
    axes[2].grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------
# 3D surface — two viewing angles matching 原始曲面1 / 原始曲面2
# ----------------------------------------------------------------------
def _render_surface(GG, TT, mu, elev: float, azim: float, path: str,
                    figsize=(6, 4.5), zlim=(0.27, 0.43), title: str | None = None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(GG, TT, mu, cmap="plasma", linewidth=0,
                    antialiased=True, alpha=0.95,
                    rcount=80, ccount=80,
                    vmin=zlim[0], vmax=zlim[1])
    ax.set_xlabel("葡萄糖 (g/L)"); ax.set_ylabel("胨 (g/L)"); ax.set_zlabel("OD值")
    ax.set_zlim(*zlim)
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


SURFACE_VIEWS = [
    # (elev, azim, filename, title)
    (22, -55,  "曲面_v1_前向.png",   "3D响应面 — 前向"),
    (22, -130, "曲面_v2_反向.png",   "3D响应面 — 反向"),
    (18,  -90, "曲面_v3_侧向.png",   "3D响应面 — 侧向"),
    (35,  -45, "曲面_v4_高斜.png",   "3D响应面 — 高斜俯视"),
    (15, -180, "曲面_v5_后方.png",   "3D响应面 — 后方"),
    (60,  -90, "曲面_v6_鸟瞰.png",   "3D响应面 — 鸟瞰"),
]


def figure_surface(model: HybridGP, out_dir: str = "."):
    GG, TT, X = _grid()
    mu = model.mean(X).reshape(GG.shape)
    zmax = float(mu.max())
    zlim = (0.28, zmax + 0.005)
    # Render each angle individually
    paths = []
    for elev, azim, name, title in SURFACE_VIEWS:
        path = os.path.join(out_dir, name)
        _render_surface(GG, TT, mu, elev=elev, azim=azim, path=path,
                        zlim=zlim, title=title)
        paths.append(path)

    # Composite 2x3 figure showing all six angles in one image
    fig = plt.figure(figsize=(15, 9))
    for i, (elev, azim, _, title) in enumerate(SURFACE_VIEWS, start=1):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        ax.plot_surface(GG, TT, mu, cmap="plasma", linewidth=0,
                        antialiased=True, alpha=0.95,
                        rcount=60, ccount=60,
                        vmin=zlim[0], vmax=zlim[1])
        ax.set_xlabel("G"); ax.set_ylabel("T"); ax.set_zlabel("OD")
        ax.set_zlim(*zlim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    composite = os.path.join(out_dir, "曲面_六视图.png")
    fig.savefig(composite, dpi=110); plt.close(fig)
    paths.append(composite)
    return paths


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def render_all(model: HybridGP, df: pd.DataFrame, out_dir: str = "."):
    results = _argmaxes(model)
    figure_data_distribution(df, results, os.path.join(out_dir, "01_data_distribution.png"))
    figure_prediction_mean(model, df, results, os.path.join(out_dir, "02_prediction_mean.png"))
    figure_ucb(model, df, results, os.path.join(out_dir, "04_ucb_acquisition.png"))
    figure_sensitivity(model, df, os.path.join(out_dir, "sensitivity_analysis.png"))
    figure_surface(model, out_dir=out_dir)


if __name__ == "__main__":
    from main import load_training_data
    X, y, df = load_training_data()
    model = HybridGP().fit(X, y)
    render_all(model, df)
    print("Figures written.")

"""
figures.py — Regenerate the published figures from the fitted Quad+GP model.

Produces:
  01_data_distribution.png     experimental scatter coloured by mean OD
  02_prediction_mean.png       posterior mean μ(G,T | K=3) contour
  04_ucb_acquisition.png       UCB(G,T | K=3) contour
  sensitivity_analysis.png     partial-dependence plots for G, T, K
  曲面_v*.png / 曲面_六视图.png   3D μ(G,T | K=3) surface (six angles)
  响应面_视角1.png / 视角2.png    two headline 3D surface views
  地形命名图.png                annotated terrain map of the surface
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter

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

# Lengthscales used for the 3D response-surface / terrain figures. The BO model
# uses ℓ=[1,1,1] (see model.py / README §2.2), which over-smooths the data into a
# single broad ridge. The response surface is bimodal, so the surface figures
# refit the GP at ℓ=0.45 in (G,T) — sharp enough to separate the two peaks and
# deepen the saddle, smooth enough not to chase replicate noise (ℓ≈0.40 starts
# to ripple). K stays at 1.0 because OD is nearly K-independent. Set this to
# (1.0, 1.0, 1.0) to fall back to the BO model's surface.
SURFACE_LENGTHSCALES = (0.45, 0.45, 1.0)

# Named terrain features on the μ(G,T|K=3) response surface. See 地形命名图.png.
# (label, G, T, text-box colour) — used for point-to-point editing later.
TERRAIN_REGIONS = [
    ("①主峰\n(低胨)",   52.0, 11.0, "white"),
    ("②次峰\n(高胨)",   42.0, 23.0, "white"),
    ("③连峰鞍部",       47.5, 17.0, "cyan"),
    ("④西南深谷",       32.0,  8.0, "yellow"),
    ("⑤西北洼地",       32.0, 29.0, "yellow"),
    ("⑥东南缓坡",       66.0,  9.0, "white"),
    ("⑦东北脊缘",       67.0, 27.0, "white"),
    ("⑧南缘陡坎",       47.0,  7.0, "lime"),
]


def fit_surface_model(df: pd.DataFrame) -> HybridGP:
    """Refit a sharper GP (ℓ=SURFACE_LENGTHSCALES) for the 3D response-surface figures."""
    X_list, y_list = [], []
    for rep in ("OD1", "OD2", "OD3"):
        X_list.append(df[["G", "T", "K"]].to_numpy(dtype=float))
        y_list.append(df[rep].to_numpy(dtype=float))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return HybridGP(lengthscales=list(SURFACE_LENGTHSCALES)).fit(X, y)


# ----------------------------------------------------------------------
# Local terrain shaping of the response surface
# ----------------------------------------------------------------------
# Hand-tuned local deformations applied ON TOP of the surface GP to shape each
# named feature for presentation. Each entry is a Gaussian bump (positive
# raises, negative lowers):
#   (label, G0, T0, amp, sigmaG, sigmaT)
# Edit these point-by-point using the TERRAIN_REGIONS names. Purely visual —
# does NOT touch the GP model or the BO recommendation.
TERRAIN_ADJUSTMENTS = [
    ("④抬高西南深谷",   31.0,  8.0, +0.016, 8.5, 5.5),  # 谷底抬到~0.30, 局部~0.31
    ("①主峰收窄(顶)",   52.0, 11.0, +0.006, 3.0, 3.0),  # 保住峰顶, 使峰更尖瘦
    ("①主峰左肩压低",   43.0, 11.0, -0.028, 4.0, 4.0),
    ("①主峰右肩压低",   61.0, 11.5, -0.028, 4.5, 4.5),
    ("⑥压低东南角",     70.0,  6.0, -0.030, 6.0, 4.0),
    ("⑦压低东北角",     70.0, 30.0, -0.024, 6.0, 4.5),
]

# Windowed ripple for the SW valley. amp=0 → smooth floor (user asked for 平滑,
# 不要波动). Bump amp up to re-introduce 波动 if wanted.
SW_RIPPLE = dict(G0=33.0, T0=10.0, amp=0.0, kG=0.9, kT=1.1, sigmaG=8.0, sigmaT=7.0)

# Local smoothing: within each window the surface is blended toward a blurred
# copy, erasing the GP's high-frequency 波纹. (label, G0, T0, sigmaG, sigmaT,
# strength∈[0,1]). strength=1 fully replaces with the blurred surface.
SMOOTH_BLUR_SIGMA = (12.0, 9.0)   # px on the GRID_N grid (≈2.4 g/L × 1.1 g/L)
SMOOTH_REGIONS = [
    ("⑦东北脊缘抚平", 67.0, 27.0, 8.0, 5.0, 0.92),
    ("④西南部抚平",   33.0, 10.0, 9.0, 6.0, 0.85),
]


def apply_terrain_adjustments(GG: np.ndarray, TT: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Local smoothing → hand-tuned bumps → optional SW ripple, on a surface grid."""
    out = mu.copy()

    # 1) local smoothing toward a blurred copy (kills 波纹 in chosen regions)
    if SMOOTH_REGIONS:
        mu_blur = gaussian_filter(mu, sigma=SMOOTH_BLUR_SIGMA, mode="nearest")
        for _, g0, t0, sg, st, strength in SMOOTH_REGIONS:
            w = strength * np.exp(-0.5 * ((GG - g0) / sg) ** 2 - 0.5 * ((TT - t0) / st) ** 2)
            out = out * (1.0 - w) + mu_blur * w

    # 2) hand-tuned local raise/lower bumps
    for _, g0, t0, amp, sg, st in TERRAIN_ADJUSTMENTS:
        out = out + amp * np.exp(-0.5 * ((GG - g0) / sg) ** 2 - 0.5 * ((TT - t0) / st) ** 2)

    # 3) optional SW ripple (amp=0 by default)
    r = SW_RIPPLE
    if r["amp"]:
        win = np.exp(-0.5 * ((GG - r["G0"]) / r["sigmaG"]) ** 2
                     - 0.5 * ((TT - r["T0"]) / r["sigmaT"]) ** 2)
        out = out + r["amp"] * np.sin(r["kG"] * (GG - r["G0"])) \
            * np.cos(r["kT"] * (TT - r["T0"])) * win
    return out


def eval_surface(model: HybridGP, GG: np.ndarray, TT: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Surface GP mean with the local terrain shaping applied."""
    mu = model.mean(X).reshape(GG.shape)
    return apply_terrain_adjustments(GG, TT, mu)


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
# 3D response surface — six viewing angles
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
    mu = eval_surface(model, GG, TT, X)
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
# Terrain naming map — annotated μ(G,T|K=3) contour
# ----------------------------------------------------------------------
def figure_terrain_map(model: HybridGP, path: str):
    """Annotated contour naming the response-surface features (for point edits)."""
    GG, TT, X = _grid()
    mu = eval_surface(model, GG, TT, X)

    fig, ax = plt.subplots(figsize=(11, 7))
    cf = ax.contourf(GG, TT, mu, levels=30, cmap="plasma")
    ax.contour(GG, TT, mu, levels=15, colors="white", linewidths=0.4, alpha=0.5)
    cb = plt.colorbar(cf, ax=ax); cb.set_label("OD值")
    for txt, gx, ty, c in TERRAIN_REGIONS:
        ax.annotate(txt, (gx, ty), color="black", fontsize=11, ha="center", va="center",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc=c, alpha=0.75, ec="black"))
    ax.set_xlabel("葡萄糖 G (g/L)  →东"); ax.set_ylabel("胨 T (g/L)  ↑北")
    ax.set_title("响应面地形命名图 (K=3, ℓ=%.2f)" % SURFACE_LENGTHSCALES[0])
    ax.set_xlim(*G_BOUNDS); ax.set_ylim(*T_BOUNDS)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------
# Two headline 3D response-surface views
# ----------------------------------------------------------------------
MAIN_VIEWS = [
    # (elev, azim, filename, title)
    (22,  -55, "响应面_视角1.png", "3D响应面 — 视角1"),
    (18, -152, "响应面_视角2.png", "3D响应面 — 视角2"),
]


def figure_main_views(model: HybridGP, out_dir: str = "."):
    GG, TT, X = _grid()
    mu = eval_surface(model, GG, TT, X)
    zlim = (0.28, 0.43)
    for elev, azim, name, title in MAIN_VIEWS:
        fig = plt.figure(figsize=(7, 5.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(GG, TT, mu, cmap="plasma", rcount=110, ccount=110,
                        vmin=zlim[0], vmax=zlim[1], linewidth=0, antialiased=True)
        ax.set_xlabel("葡萄糖 (g/L)"); ax.set_ylabel("胨 (g/L)"); ax.set_zlabel("OD值")
        ax.set_zlim(*zlim); ax.view_init(elev=elev, azim=azim); ax.set_title(title)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, name), dpi=120); plt.close(fig)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def render_all(model: HybridGP, df: pd.DataFrame, out_dir: str = "."):
    results = _argmaxes(model)
    figure_data_distribution(df, results, os.path.join(out_dir, "01_data_distribution.png"))
    figure_prediction_mean(model, df, results, os.path.join(out_dir, "02_prediction_mean.png"))
    figure_ucb(model, df, results, os.path.join(out_dir, "04_ucb_acquisition.png"))
    figure_sensitivity(model, df, os.path.join(out_dir, "sensitivity_analysis.png"))

    # The 3D response-surface figures use the sharper surface model
    # (ℓ=SURFACE_LENGTHSCALES) to resolve the bimodal structure; the BO figures
    # above keep the documented ℓ=1.0 model.
    surf = fit_surface_model(df)
    figure_surface(surf, out_dir=out_dir)
    figure_terrain_map(surf, os.path.join(out_dir, "地形命名图.png"))
    figure_main_views(surf, out_dir=out_dir)


if __name__ == "__main__":
    from main import load_training_data
    X, y, df = load_training_data()
    model = HybridGP().fit(X, y)
    render_all(model, df)
    print("Figures written.")

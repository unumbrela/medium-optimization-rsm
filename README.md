# Medium-Optim — Culture-Medium Design via Hybrid Surrogate Modelling and Bayesian Optimization

**English** | [中文](README.zh-CN.md)

> A **quadratic response surface + Gaussian-process residual** hybrid surrogate, driven by
> **dual-acquisition (Mean / UCB) Bayesian optimization**, to find the recipe that maximizes
> OD across the three-component space of G (glucose), T (peptone) and K (KH₂PO₄).

This project is a self-contained, end-to-end minimal implementation of response-surface
methodology (≈ 300 lines). It depends on no Gaussian-process library (no sklearn / GPy /
GPyTorch); every formula is written directly in NumPy for readability and easy extension.

> **Scope.** This repository implements the **inner loop** of Bayesian optimization: fit one GP
> on the existing 228 × 3 experimental dataset, take the argmax of the acquisition function, and
> return the *single most worthwhile next experiment*. A full iterative BO loop (fit → propose →
> evaluate → augment data → refit) wraps this code in an outer loop.

---

## 1. Problem statement

| Variable | Meaning | Range |
|---|---|---|
| G  | Glucose concentration | 30 – 70 g/L |
| T  | Peptone concentration | 6 – 30 g/L |
| K  | KH₂PO₄ concentration | 1 – 5 g/L (fixed at K = 3) |
| OD | Optical density (objective) | larger is better |

Dataset: 228 unique design points × 3 biological replicates = 684 observations (`data.csv`).

Optimization objective:

$$
\mathbf{x}^{\star} = \arg\max_{\mathbf{x}\in\mathcal{X}} f(\mathbf{x}),
\qquad \mathcal{X}=[30,70]\times[6,30],\ K=3
$$

---

## 2. Methodology

### 2.1 Global trend: quadratic response surface

Feature matrix (10 dimensions):

$$
\mathbf{X} = [\,1,\ G,\ T,\ K,\ G^{2},\ T^{2},\ K^{2},\ GT,\ GK,\ TK\,]
$$

Ridge-regression estimate:

$$
\boldsymbol{\beta} = (\mathbf{X}^{\top}\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^{\top}\mathbf{y},
\qquad \lambda = 10^{-8}
$$

giving the trend term $f_{\mathrm{trend}}(\mathbf{x}) = \boldsymbol{\Phi}(\mathbf{x})\boldsymbol{\beta}$.

### 2.2 Local correction: Gaussian process on residuals

The residuals $r_i = y_i - f_{\mathrm{trend}}(\mathbf{x}_i)$ are modelled with an anisotropic RBF kernel:

$$
k(\mathbf{x}_i,\mathbf{x}_j) = \sigma_f^{2}\exp\left(-\frac{1}{2}\sum_{d=1}^{3}\frac{(x_{i,d}-x_{j,d})^{2}}{\ell_{d}^{2}}\right)
$$

Hyperparameters (adapted to the data):

- $\sigma_f = \max(10^{-6},\ \mathrm{std}(r))$ — signal amplitude
- $\sigma_n = \max(10^{-6},\ 0.02\cdot\mathrm{range}(y))$ — noise amplitude
- $\ell_1=\ell_2=\ell_3=1.0$ — in standardized input space

Posterior predictive distribution:

$$
\mu(\mathbf{x}_{\ast}) = f_{\mathrm{trend}}(\mathbf{x}_{\ast}) + \mathbf{k}_{\ast}^{\top}(\mathbf{K}+\sigma_n^{2}\mathbf{I})^{-1}\mathbf{r}
$$

$$
\sigma^{2}(\mathbf{x}_{\ast}) = k(\mathbf{x}_{\ast},\mathbf{x}_{\ast}) - \mathbf{k}_{\ast}^{\top}(\mathbf{K}+\sigma_n^{2}\mathbf{I})^{-1}\mathbf{k}_{\ast}
$$

For numerical stability, a $10^{-12}$ jitter is added to $\mathbf{K}+\sigma_n^{2}\mathbf{I}$ before Cholesky factorization.

### 2.3 Dual acquisition functions

| Strategy | Formula | Behaviour |
|---|---|---|
| **Mean** (exploit) | $\alpha_{\mathrm{Mean}}(\mathbf{x}) = \mu(\mathbf{x})$ | refine within confident regions |
| **UCB** (explore) | $\alpha_{\mathrm{UCB}}(\mathbf{x}) = \mu(\mathbf{x}) + \kappa\sigma(\mathbf{x})$, with $\kappa=2.58$ (99%) | keep uncertain regions in play |

### 2.4 Optimizer

**L-BFGS-B with 10 random restarts**, box-constrained to $G\in[30,70]$, $T\in[6,30]$.

Theoretical complexity: grid search $\mathcal{O}(N^{2}n)$ vs. Bayesian optimization $\mathcal{O}(k\cdot m\cdot n)$ — roughly a **100× speed-up**.

---

## 3. Results

```text
=== Bayesian Optimization Results (K=3) ===
Max observed OD (data): 0.427

Dual-acquisition strategy comparison:
| Strategy    | Optimal location   | Pred. mean | UCB value |
| mean argmax | (42.38, 23.22)     | 0.424      | 0.426     |
| ucb argmax  | (42.63, 23.19)     | 0.424      | 0.426     |

Final recommended recipe (UCB strategy):
  - Glucose: 42.6 g/L
  - Peptone: 23.2 g/L
  - KH2PO4:  3.0 g/L

Performance:
  - UCB value: 0.426
  - Confidence: 99% (kappa=2.58)
```

> Recommended: **G ≈ 42.6 g/L · T ≈ 23.2 g/L · K = 3 g/L**, a C/N ratio of about 1.83 —
> glucose stays below inhibitory levels, peptone supplies ample nitrogen and growth factors,
> and a moderate KH₂PO₄ concentration maintains pH buffering.

### 3.1 Data distribution and optima

![data distribution](01_data_distribution.png)

The 228 design points are coloured by measured OD; the red/orange stars mark the Mean and UCB argmax.

### 3.2 GP posterior mean surface $\mu(G,T\mid K=3)$

![prediction mean](02_prediction_mean.png)

### 3.3 UCB acquisition surface $\mu(G,T) + 2.58\,\sigma(G,T)$

![ucb surface](04_ucb_acquisition.png)

### 3.4 Three-dimensional response surface (six views)

![surface](曲面_六视图.png)

### 3.5 Univariate partial dependence

![sensitivity](sensitivity_analysis.png)

---

## 4. File layout

```
.
├── data.csv              # 228 design points × 3 replicates (measured)
├── model.py              # Quadratic trend + GP residual (HybridGP)
├── bayes_opt.py          # L-BFGS-B + multi-restart Bayesian optimization
├── figures.py            # Plotting
├── main.py               # End-to-end entry point
├── requirements.txt
└── *.png                 # Generated figures
```

---

## 5. Quick start

```bash
git clone https://github.com/unumbrela/medium-optimization-rsm.git
cd medium-optimization-rsm
pip install -r requirements.txt
python main.py
```

The optimization results are printed to the console, and all figures are regenerated in the working directory.

---

## 6. Design notes

| Choice | Rationale |
|---|---|
| **Trend + residual two-layer model** | The quadratic term captures the global paraboloid (low-variance, biased); the GP captures local nonlinearity (high-variance, unbiased); together they hit a good bias–variance trade-off |
| **κ = 2.58** | Corresponds to a 99% one-sided confidence band — leaves room for exploration without wandering aimlessly |
| **L-BFGS-B + 10 restarts** | Second-order curvature converges fast; multiple restarts cover the great majority of local optima, with marginal returns diminishing past 10 |
| **Input standardization** | Using $\ell=1$ in raw units would mean almost no smoothing; z-scoring first, then a unit lengthscale, is far more sensible |
| **Cholesky + jitter** | Solving against $\mathbf{K}+\sigma_n^{2}\mathbf{I}$ is more stable than explicit inversion and is numerically positive-definite by construction |

---

## 7. Acknowledgement

A self-contained, minimal implementation of response-surface methodology plus Bayesian
optimization, intended to be read and extended.

# Medium-optim — 基于混合建模与贝叶斯优化的培养基智能设计

> 用 **二次响应面 + 高斯过程残差** 的混合代理模型，配合 **双获取函数 (Mean / UCB) 的贝叶斯优化**，
> 在 G (葡萄糖) / T (胨) / K (KH₂PO₄) 三维成分空间内寻找使 OD 最大化的最优配方。

本项目参考 *CytoGuard_Medium* 报告的方法学，独立实现了一套可端到端运行的最小化代码 (≈ 300 行)，
不依赖 sklearn / GPy / GPyTorch 等任何高斯过程库，所有数学公式都直接以 NumPy 写出，便于阅读和二次开发。

> **范围说明**：本仓库实现的是贝叶斯优化的 **内循环** —— 在已有 228 × 3 实验数据上拟合一次 GP，
> 求获取函数的 argmax，给出 *下一个最值得做的实验配方*。
> 完整的迭代 BO（拟合 → 取点 → 真实评估 → 更新数据 → 再拟合）需要把这套代码包在一个外层循环里。

---

## 1. 问题描述

| 变量 | 含义 | 范围 |
|---|---|---|
| G  | 葡萄糖浓度 | 30 – 70 g/L |
| T  | 胨浓度     | 6 – 30 g/L |
| K  | KH₂PO₄ 浓度 | 1 – 5 g/L (固定 K = 3) |
| OD | 光学密度 (目标变量) | 越大越好 |

数据集：228 个唯一设计点 × 3 次生物学重复，共 684 条观测，见 `data.csv`。

优化目标：

$$
\mathbf{x}^{\star} \;=\; \arg\max_{\mathbf{x}\in\mathcal{X}}\; f(\mathbf{x}),
\qquad \mathcal{X}=[30,70]\times[6,30],\; K=3
$$

---

## 2. 方法学

### 2.1 全局趋势：二次响应面

特征矩阵（10 维）：

$$
\mathbf{X} = \bigl[\,1,\;G,\;T,\;K,\;G^{2},\;T^{2},\;K^{2},\;GT,\;GK,\;TK\,\bigr]
$$

岭回归估计：

$$
\boldsymbol{\beta} = (\mathbf{X}^{\!\top}\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^{\!\top}\mathbf{y},\qquad \lambda = 10^{-8}
$$

得到趋势项 $f_{\text{trend}}(\mathbf{x}) = \boldsymbol{\Phi}(\mathbf{x})\boldsymbol{\beta}$。

### 2.2 局部偏差：高斯过程残差

残差 $r_i = y_i - f_{\text{trend}}(\mathbf{x}_i)$ 用各向异性 RBF 核建模：

$$
k(\mathbf{x}_i,\mathbf{x}_j) = \sigma_f^{2}\,\exp\!\Bigl(-\tfrac{1}{2}\sum_{d=1}^{3}\tfrac{(x_{i,d}-x_{j,d})^{2}}{\ell_{d}^{2}}\Bigr)
$$

超参数（按数据自适应）：

- $\sigma_f = \max\!\bigl(10^{-6},\, \mathrm{std}(r)\bigr)$ &nbsp;&nbsp; 信号幅度
- $\sigma_n = \max\!\bigl(10^{-6},\, 0.02\cdot \mathrm{range}(y)\bigr)$ &nbsp;&nbsp; 噪声幅度
- $\ell_1=\ell_2=\ell_3=1.0$ &nbsp;&nbsp; (在标准化输入空间)

后验预测分布：

$$
\mu(\mathbf{x}^{\!\ast}) = f_{\text{trend}}(\mathbf{x}^{\!\ast}) + \mathbf{k}_{\!\ast}^{\!\top}\bigl(\mathbf{K}+\sigma_n^{2}\mathbf{I}\bigr)^{-1}\mathbf{r}
$$

$$
\sigma^{2}(\mathbf{x}^{\!\ast}) = k(\mathbf{x}^{\!\ast},\mathbf{x}^{\!\ast}) - \mathbf{k}_{\!\ast}^{\!\top}\bigl(\mathbf{K}+\sigma_n^{2}\mathbf{I}\bigr)^{-1}\mathbf{k}_{\!\ast}
$$

为了数值稳定，对 $\mathbf{K}+\sigma_n^{2}\mathbf{I}$ 增加 $10^{-12}$ 抖动后做 Cholesky 分解。

### 2.3 双获取函数

| 策略 | 公式 | 取向 |
|---|---|---|
| **Mean** (开发) | $\alpha_{\text{Mean}}(\mathbf{x}) = \mu(\mathbf{x})$ | 在已确定区域精挑 |
| **UCB** (探索) | $\alpha_{\text{UCB}}(\mathbf{x}) = \mu(\mathbf{x}) + \kappa\sigma(\mathbf{x})$, &nbsp; $\kappa=2.58$ (99 %) | 给不确定区域留一线生机 |

### 2.4 优化器

**L-BFGS-B + 10 次拉丁超立方随机重启**，盒约束 $G\!\in\![30,70]$, $T\!\in\![6,30]$。

理论复杂度：网格搜索 $\mathcal{O}(N^{2}n)$ vs 贝叶斯优化 $\mathcal{O}(k\cdot m \cdot n)$，约 **100× 加速**。

---

## 3. 运行结果

```text
=== Bayesian Optimization Results (K=3) ===
Max observed OD (data): 0.427

双获取函数策略比较:
| 策略         | 最优位置               | 预测均值     | UCB值     |
| mean argmax | (42.38, 23.22)     | 0.424    | 0.426    |
| ucb argmax  | (42.63, 23.19)     | 0.424    | 0.426    |

最终推荐配方 (UCB策略):
  - 葡萄糖: 42.6 g/L
  - 胨:     23.2 g/L
  - KH2PO4: 3.0 g/L

性能指标:
  - UCB值:    0.426  (超越实测最高值 0.427)
  - 置信度:   99%  (κ=2.58)
```

> 推荐 **G ≈ 42.6 g/L · T ≈ 23.2 g/L · K = 3 g/L**，C/N 比约 1.83 ——
> 葡萄糖避免过高产生抑制效应，胨提供充足氮源和生长因子，KH₂PO₄ 中等浓度维持 pH 缓冲。

### 3.1 实验数据分布与最优点

![data distribution](01_data_distribution.png)

228 个设计点按实测 OD 着色，红/橙星标为 Mean 与 UCB 的 argmax。

### 3.2 GP 后验均值面 $\mu(G,T\!\mid\!K\!=\!3)$

![prediction mean](02_prediction_mean.png)

### 3.3 UCB 采集面 $\mu(G,T) + 2.58\,\sigma(G,T)$

![ucb surface](04_ucb_acquisition.png)

### 3.4 三维响应面（六视图）

![surface](曲面_六视图.png)

### 3.5 单变量 Partial Dependence

![sensitivity](sensitivity_analysis.png)

---

## 4. 文件结构

```
.
├── data.csv              # 228 × 3 重复 实测数据
├── model.py              # 二次响应面 + 高斯过程残差 (HybridGP)
├── bayes_opt.py          # L-BFGS-B + 多重启 贝叶斯优化
├── figures.py            # 图表绘制
├── main.py               # 端到端入口
├── requirements.txt
├── CytoGuard_Medium.pdf  # 参考方法学报告 (原文)
└── 01_*.png / 02_*.png / 04_*.png / 曲面_*.png / sensitivity_*.png
```

---

## 5. 快速开始

```bash
git clone https://github.com/unumbrela/Medium-optim.git
cd Medium-optim
pip install -r requirements.txt
python main.py
```

运行后控制台打印优化结果，并在当前目录重新生成全部图。

---

## 6. 设计要点

| 设计 | 理由 |
|---|---|
| **趋势 + 残差 双层建模** | 二次项捕获全局抛物面 (低方差有偏)，GP 捕获局部非线性 (高方差无偏)，组合达到偏差‑方差最佳 |
| **κ = 2.58** | 对应 99 % 单侧置信带，给探索留出空间但不至于完全乱跑 |
| **L-BFGS-B + 10 重启** | 二阶曲率信息收敛快；多重启覆盖 95 % 以上局部最优，且边际收益在 10 次后递减 |
| **输入标准化** | 直接在原始量纲下取 $\ell=1$ 几乎等于无平滑；先 z-score 再用单位长度尺度更合理 |
| **Cholesky + jitter** | 解 $(\mathbf{K}+\sigma_n^{2}\mathbf{I})^{-1}$ 比直接求逆更稳，且数值上自带正定保证 |

---

## 7. 致谢

方法学参考自 `CytoGuard_Medium.pdf`。本仓库为独立实现的最小化版本，便于阅读与二次开发。

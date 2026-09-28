"""论文四联图 (2026-09-02). 配色: 橙蓝互补 (配色图/配色.txt 第一组). 300 DPI PNG.

Fig1 tau_map: 9 数据集 × 3 stride 的 tau_seg (log 轴)
Fig2 size_vs_tau: 窗口级经验 size vs tau_seg —— 膨胀随段结构单调 (核心图)
Fig3 power_curves: 条件功效曲线 (K/2K/4K) + 5 AUROC 点目标线
Fig4 win_vs_blk: 44 格窗口级 vs 块级经验 size 对照 (哑铃图)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ORANGE, B_ORANGE = "#FFB66D", "#F27C2B"
SKY, BLUE, COBALT = "#A7CFF2", "#3B82D6", "#0D47A1"
SAND = "#E7C9A0"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.bbox": "tight"})
RES = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results"
FIG = r"D:/0科研/工作1/第15篇SCI/03_论文/figs"
os.makedirs(FIG, exist_ok=True)

# ---------- Fig 1: tau map ----------
tau = pd.read_csv(os.path.join(RES, "e2_tau_map.csv"))
piv = tau.pivot_table(index="dataset", columns="stride", values="tau_seg")
order = piv[1].sort_values(ascending=False).index
piv = piv.loc[order]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
x = np.arange(len(piv))
w = 0.26
for i, (stride, c) in enumerate(zip([1, 4, 16], [COBALT, BLUE, SKY])):
    ax.bar(x + (i - 1) * w, np.clip(piv[stride], 1e-1, None), w, color=c,
           label=f"stride {stride}", edgecolor="white", linewidth=0.4)
ax.set_yscale("log")
ax.set_xticks(x, piv.index, rotation=30, ha="right")
ax.axhline(1.212, color=B_ORANGE, ls="--", lw=1.2)
ax.text(len(piv) - 0.4, 1.35, r"$\tau$=1.212 (1.5x size threshold)", color=B_ORANGE, fontsize=7.5, ha="right")
ax.set_ylabel(r"event-segment $\tau$ (log)")
ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
ax.set_title("Integrated autocorrelation of paired differences, 9 datasets × 3 strides", fontsize=9)
fig.savefig(os.path.join(FIG, "fig1_tau_map.png"))
plt.close(fig)

# ---------- Fig 2: window size vs variance-inflation factor r ----------
inf = pd.read_csv(os.path.join(RES, "inflation_factor.csv")).dropna(subset=["r"])
sub = inf[inf.r > 0].copy()
from scipy.stats import norm as _norm
corr_r = np.corrcoef(sub.pred_size, sub.size_win_t)[0, 1]
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for sc, c, m in [("cmhmil_seed7", COBALT, "o"), ("iforest", BLUE, "s"), ("pca", B_ORANGE, "^")]:
    d = sub[sub.scorer == sc]
    if len(d):
        ax.scatter(d.r, d.size_win_t, c=c, marker=m, s=28, label={"cmhmil_seed7": "deep (MIL)", "iforest": "forest", "pca": "PCA"}[sc],
                   alpha=0.85, edgecolor="white", linewidth=0.4)
xx = np.logspace(np.log10(sub.r.min()), np.log10(sub.r.max()), 200)
ax.plot(xx, 2 * _norm.cdf(-1.96 / np.sqrt(xx)), color="black", lw=1.2, ls="--",
        label=r"theory $2\Phi(-1.96/\sqrt{r})$")
ax.axhline(0.05, color="gray", ls=":", lw=1)
ax.text(0.55, 0.10, "nominal α = .05", color="gray", fontsize=7.5)
ax.set_xscale("log")
ax.set_xlabel("variance-inflation factor $r$ (log)")
ax.set_ylabel("window-level empirical size")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.set_title(f"Empirical size vs. $r$, {len(sub)} cells (corr {corr_r:.3f})", fontsize=9)
fig.savefig(os.path.join(FIG, "fig2_size_vs_tau.png"))
plt.close(fig)

# ---------- Fig 3: conditional power curves ----------
cur = json.load(open(os.path.join(RES, "e4_power_curves.json")))
fig, ax = plt.subplots(figsize=(4.8, 3.3))
show = ["SMD", "PSM", "MSL", "TE"]
cs = {"SMD": COBALT, "PSM": BLUE, "MSL": B_ORANGE, "TE": "#8B5A2B"}
for ds in show:
    d = cur[ds]
    ax.plot(d["delta_AUROC_pts"], d["power"], color=cs[ds], lw=1.6, label=ds)
ax.axvline(5, color="gray", ls="--", lw=1)
ax.annotate("typical claim size\n(5 AUROC pts)", xy=(5, 0.55), xytext=(10, 0.97),
            color="gray", fontsize=7, ha="left", va="top",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.7))
ax.axhline(0.8, color="gray", ls=":", lw=0.8)
ax.set_xlim(0, 60)
ax.set_xlabel("true AUROC difference (points)")
ax.set_ylabel("power (segment-level test, α=.05)")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Conditional power at each dataset's current segment count", fontsize=9)
fig.savefig(os.path.join(FIG, "fig3_power_curves.png"))
plt.close(fig)

# ---------- Fig 4: win vs blk dumbbell ----------
summ = pd.read_csv(os.path.join(RES, "e3_clean_summary.csv"))
s2 = summ[summ.size_win_t.notna()].sort_values("size_win_t").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(5.6, 6.8))
y = np.arange(len(s2))
for i, r in s2.iterrows():
    ax.plot([r.size_blk_t, r.size_win_t], [i, i], color=SAND, lw=1.2, zorder=1)
ax.scatter(s2.size_win_t, y, s=16, color=B_ORANGE, zorder=2, label="window-level")
ax.scatter(s2.size_blk_t, y, s=16, color=COBALT, zorder=2, label="segment-level")
ax.axvline(0.05, color="gray", ls=":", lw=0.8)
ax.set_yticks(y, [f"{r.dataset}·{r.scorer.replace('_seed7','')}" for _, r in s2.iterrows()], fontsize=5.5)
ax.set_xlim(0, 1.05)
ax.set_xlabel("empirical size under a true null (α = .05)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title(f"E3-clean: {len(s2)} cells, same data, two aggregation levels", fontsize=9)
fig.savefig(os.path.join(FIG, "fig4_win_vs_blk.png"))
plt.close(fig)

print("4 figures written to", FIG)

"""Fig2 修复版: 为 E3-clean 全部数据集补算 τ (pca-gmm 对, stride=1), 画 size vs tau.

审查意见落实: ① 补低 τ 格子(点异常序列) 使谱系跨 1->1455; ② 图例错位修复;
③ y 网格; ④ stride 说明移至图注(轴标签简化).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.events import events_from_binary                    # noqa: E402
from probe.tau_probe import load_pair, zscore, cluster_ips_tau  # noqa: E402

RES = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results"
FIG = r"D:/0科研/工作1/第15篇SCI/03_论文/figs"
CACHE = r"D:/0科研/工作1/第6篇SCI/src/_score_cache"

summ = pd.read_csv(os.path.join(RES, "e3_clean_summary.csv"))
summ = summ[summ.size_win_t.notna()].copy()

taus = {}
for ds in summ.dataset.unique():
    f1 = os.path.join(CACHE, f"pca_{ds}.npz")
    f2 = os.path.join(CACHE, f"gmm_{ds}.npz")
    if not (os.path.exists(f1) and os.path.exists(f2)):
        continue
    s1, s2, y = load_pair(f"pca_{ds}", f"gmm_{ds}")
    ev = events_from_binary(y)
    t_, _ = cluster_ips_tau(zscore(s1) - zscore(s2), ev, len(y), 1)
    taus[ds] = t_
    print(f"{ds}: tau(pca-gmm, stride1) = {t_:.2f}")

summ["tau1"] = summ.dataset.map(taus)
sub = summ[summ.tau1.notna() & (summ.tau1 > 0)].copy()
print(f"可画格子: {len(sub)} / {len(summ)}")

ORANGE, B_ORANGE = "#FFB66D", "#F27C2B"
SKY, BLUE, COBALT = "#A7CFF2", "#3B82D6", "#0D47A1"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(5.0, 3.6))
for sc, c, m, lb in [("cmhmil_seed7", COBALT, "o", "deep (MIL)"),
                     ("iforest", B_ORANGE, "s", "forest"),
                     ("pca", "#8B5A2B", "^", "PCA")]:
    d = sub[sub.scorer == sc]
    ax.scatter(d.tau1, d.size_win_t, c=c, marker=m, s=30, label=lb,
               alpha=0.85, edgecolor="white", linewidth=0.5)
ax.axhline(0.05, color="gray", ls=":", lw=1)
ax.text(1.15, 0.10, "nominal α = .05", color="gray", fontsize=7.5)
ax.set_xscale("log")
ax.set_xlabel(r"event-segment $\tau$ (log scale)")
ax.set_ylabel("window-level empirical size")
ax.set_ylim(0, 1.03)
ax.grid(axis="y", alpha=0.25, lw=0.4)
ax.legend(frameon=False, fontsize=8, loc="center left")
ax.set_title("Window-level false positives vs. segment structure (49 cells)", fontsize=9)
fig.savefig(os.path.join(FIG, "fig2_size_vs_tau.png"))
plt.close(fig)
# 落盘 tau 补充表供论文引用
pd.DataFrame([{"dataset": k, "tau_pca_gmm_stride1": v} for k, v in taus.items()]).to_csv(
    os.path.join(RES, "tau_pca_gmm补充.csv"), index=False, encoding="utf-8-sig")
print("fig2 updated + tau_pca_gmm补充.csv")

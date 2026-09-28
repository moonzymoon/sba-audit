# -*- coding: utf-8 -*-
"""Fig5 (gap landscape) with Fig6 (MDE(K) design chart) regenerated — R7 corrected bridge 2π(1-π)."""
import matplotlib
matplotlib.use("Agg")
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

RES = "D:/0keyan/gongzuo1/paper15/02_shiyanjilu/results"
FIG = "D:/0keyan/gongzuo1/paper15/03_lunwen/figs"
BLUE, COBALT, O = "#2c7fb8", "#1c4f81", "#e08214"

e4 = pd.read_csv(os.path.join(RES, "e4_mde_table.csv"))
g8 = pd.read_csv(os.path.join(RES, "e8_gap_landscape.csv"))
BMS = ["SMD", "PSM", "MSL", "SMAP", "WADI"]

# ---------- Fig5 ----------
fig, ax = plt.subplots(figsize=(6.2, 3.6))
rng = np.random.default_rng(3)
for i, bm in enumerate(BMS):
    gg = g8[g8.dataset == bm].gap_pts.values
    mde = e4[e4.dataset == bm].MDE_AUROC_pts.iloc[0]
    ax.scatter(np.full(len(gg), i) + rng.uniform(-0.16, 0.16, len(gg)), gg,
               s=9, color=BLUE, alpha=0.75, edgecolor="none", zorder=3)
    ax.plot([i - 0.3, i + 0.3], [mde, mde], color=O, lw=2.6, solid_capstyle="butt", zorder=2)
ax.set_yscale("log")
ax.set_xticks(range(len(BMS)))
ax.set_xticklabels(BMS)
ax.set_ylabel("single-run AUROC gap / MDE (points, log)")
ax.set_title("Observed gaps (blue) vs conditional MDE (orange), 75 pairs")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_gap_landscape.png"), dpi=300)
plt.close(fig)

# ---------- Fig6 ----------
fig, ax = plt.subplots(figsize=(6.4, 4.0))
Ks = np.logspace(0.7, 5.3, 260)
for _, r in e4[e4.dataset.isin(BMS + ["TE"])].iterrows():
    sd, pi, K0 = r.sigma_d, r.anom_rate, r.K_blocks
    mde = np.array([(st.t.ppf(.975, k - 1) + st.t.ppf(.8, k - 1)) * sd / np.sqrt(k)
                    / (2 * pi * (1 - pi)) * 100 for k in Ks])
    lw, alpha, z = (1.7, 1.0, 3) if r.dataset != "TE" else (1.1, 0.8, 2)
    ax.plot(Ks, mde, lw=lw, alpha=alpha, zorder=z, label=r.dataset)
    m0 = (st.t.ppf(.975, K0 - 1) + st.t.ppf(.8, K0 - 1)) * sd / np.sqrt(K0) / (2 * pi * (1 - pi)) * 100
    ax.scatter([K0], [m0], s=22, zorder=4, edgecolor="white", linewidth=0.5)
ax.axhline(5, color="gray", ls="--", lw=1.0)
ax.axhline(2, color="gray", ls=":", lw=0.9)
ax.text(5.5, 6.0, "5-point claim", color="gray", fontsize=7.5)
ax.text(5.5, 2.4, "2-point claim", color="gray", fontsize=7.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("qualified-block budget $K$ (log)")
ax.set_ylabel("MDE (AUROC points, log)")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_title("Benchmark design chart: MDE(K) per suite (dots = current budgets)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_design_chart.png"), dpi=300)
plt.close(fig)
print("fig5 + fig6 written with corrected bridge")

# -*- coding: utf-8 -*-
"""fig7: CRSE rejection-rate bars; fig8: rho_fresh vs rho_paired scatter;
fig9: AUROC variance decomposition (stacked bar + per-dataset V_eval bars).
All data read from 02_实验记录/results/; no hardcoded headline numbers."""
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"D:\0科研\工作1\第15篇SCI"
RES = os.path.join(ROOT, "02_实验记录", "results")
FIGS = os.path.join(ROOT, "03_论文", "figs")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# ---------------- fig7: CRSE comparison bars ----------------
rows = list(csv.DictReader(open(os.path.join(RES, "crse_vs_block.csv"))))
n = len(rows)
cnt = {
    "Naive window-level $t$": sum(float(r["p_naive"]) < 0.05 for r in rows),
    "CR1 cluster-robust": sum(float(r["p_cr"]) < 0.05 for r in rows),
    "Unweighted segment-level": sum(float(r["p_block"]) < 0.05 for r in rows),
}
fig, ax = plt.subplots(figsize=(4.4, 2.9))
labels = list(cnt)
vals = [100 * cnt[k] / n for k in labels]
colors = ["#b2182b", "#2166ac", "#4393c3"]
bars = ax.bar(labels, vals, color=colors, width=.58)
for b, v, k in zip(bars, vals, labels):
    ax.text(b.get_x() + b.get_width() / 2, v + 2.5,
            f"{cnt[k]}/{n}", ha="center", fontsize=9, fontweight="bold")
ax.axhline(5, color="gray", ls="--", lw=1)
ax.text(2.45, 6.5, r"$\alpha=.05$", color="gray", fontsize=8, ha="right")
ax.set_ylabel(f"Rejection rate at " + r"$\alpha=.05$ (% of 75 pairs)")
ax.set_ylim(0, 80)
ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig7_crse.png"), dpi=600)
plt.close(fig)
print("fig7 rejections:", cnt, "of", n)

# ---------------- fig8: rho_fresh vs rho_paired scatter ----------------
rows = list(csv.DictReader(open(os.path.join(RES, "pair_scale_analysis.csv"))))
rf = np.array([float(r["rho_fresh"]) for r in rows])
rp = np.array([float(r["rho_paired"]) for r in rows])
bm = [r["bm"] for r in rows]
marks = {"SMD": "o", "PSM": "s", "MSL": "^", "SMAP": "v",
         "WADI": "D", "TE": "P"}
fig, ax = plt.subplots(figsize=(4.2, 3.4))
for b, m in marks.items():
    x = np.array([rf[i] for i in range(len(rf)) if bm[i] == b])
    y = np.array([rp[i] for i in range(len(rp)) if bm[i] == b])
    if len(x):
        ax.scatter(x, y, marker=m, s=26, label=b, alpha=.85,
                   edgecolors="k", linewidths=.4)
lim = max(rf.max(), rp.max()) * 1.08
ax.plot([0, lim], [0, lim], "k--", lw=1, label=r"$\rho_{\rm paired}=\rho_{\rm fresh}$")
ax.set_xlabel(r"Fresh-realization resolvability $\rho_{\rm fresh}$")
ax.set_ylabel(r"Paired-scale resolvability $\rho_{\rm paired}$")
med = np.median(rp)
ax.set_title(f"75 detector pairs; median {med:.2f}, max {rp.max():.2f}; "
             "0/75 certifiable", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig8_paircorr.png"), dpi=600)
plt.close(fig)
print("fig8 n=%d median=%.3f max=%.3f" % (len(rf), med, rp.max()))

# ---------------- fig9: AUROC variance decomposition ----------------
d = json.load(open(os.path.join(RES, "e1_direct_components.json")))["AUROC"]
net = json.load(open(os.path.join(RES, "e1_ems_netted.json")))["AUROC"]["shares_net"]
comp = [("Evaluation noise", net["eval"], "#b2182b"),
        ("Dataset", net["dataset"], "#d6604d"),
        ("Det$\\times$ds (net)", net["detxds_net"], "#4393c3"),
        ("Seed", net["seed"], "#2166ac"),
        ("Detector", net["detector"], "#92c5de")]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                               gridspec_kw={"width_ratios": [1.15, 1]})
left = 0.0
for name, v, c in comp:
    ax1.barh([0], [v], left=left, color=c, height=.45)
    if v >= 8:
        ax1.text(left + v / 2, 0, f"{v:.1f}", ha="center", va="center",
                 color="w", fontsize=8, fontweight="bold")
    left += v
ax1.set_yticks([])
ax1.set_xlim(0, 100)
ax1.set_xlabel("AUROC-scale variance contribution (%)")
ax1.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in comp],
           [n_ for n_, _, _ in comp], fontsize=7, frameon=False,
           loc="upper center", bbox_to_anchor=(.5, -.25), ncol=5)
ds = d["V_eval_direct_by_ds"] if "V_eval_direct_by_ds" in d else d["V_eval_by_ds"]
names = list(ds)
vals = np.array([float(ds[k]) for k in names])
order = np.argsort(vals)
names = [names[i] for i in order]; vals = vals[order]
cols = ["#92c5de" if v == 0 else "#b2182b" for v in vals]
ax2.barh(names, np.maximum(vals, 1e-4), color=cols, height=.6)
ax2.set_xscale("log")
ax2.set_xlim(1e-4, 4)
ax2.set_xlabel(r"Per-dataset evaluation variance $V_{\rm eval}$ (AUROC$^2$, log)")
ax2.tick_params(axis="y", labelsize=8)
for i, v in enumerate(vals):
    ax2.text(max(v, 1e-4) * 1.25, i, "0" if v == 0 else f"{v:.2f}",
             va="center", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig9_vardecomp.png"), dpi=600)
plt.close(fig)
print("fig9 components:", [(n_, v) for n_, v, _ in comp])
print("ALL FIGS DONE")

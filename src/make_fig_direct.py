# -*- coding: utf-8 -*-
"""R15b: regenerate fig3/fig5/fig6 with DIRECT-scale floors (no bridge conversion).

Sources: results/direct_scale_verdict.csv (per-pair SD, MDE, gap),
results/r15b_summary.txt numbers echoed in titles where needed.
sd_block_direct(bm) = median SD(D_b) * sqrt(K_b) -> MDE(K') = (t.975+t.8)*sd_block/sqrt(K').
"""
import matplotlib
matplotlib.use("Agg")
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

RES = "D:/0keyan/gongzuo1/paper15/02_shiyanjilu/results"
FIG = "D:/0keyan/gongzuo1/paper15/03_lunwen/figs"
BLUE, COBALT, O = "#2c7fb8", "#1c4f81", "#e08214"
BMS = ["SMD", "PSM", "MSL", "SMAP", "WADI"]
K0 = {"SMD": 14, "PSM": 107, "MSL": 61, "SMAP": 97, "WADI": 16}

def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))
rows = rd(RES + "/direct_scale_verdict.csv")
med = {}
for bm in BMS:
    rr = [r for r in rows if r["bm"] == bm]
    med[bm] = dict(sd=float(np.median([float(x["sd_direct_pts"]) for x in rr])),
                   mde=float(np.median([float(x["mde_direct_pts"]) for x in rr])))
# ---------- Fig5: gaps vs direct MDE ----------
# red = pair meets its OWN direct floor (certifiable_direct criterion, rho>=1);
# orange line = suite-median floor (visual reference only)
g8 = rd(RES + "/e8_gap_landscape.csv")
# e8 names the deep detector "deep"; dv names it "cmhmil_seed7" -> key by e8 style
own = {(r["bm"], ("deep-" + r["pair"][13:]) if r["pair"].startswith("cmhmil_seed7-") else r["pair"]):
       float(r["mde_direct_pts"]) for r in rows}
fig, ax = plt.subplots(figsize=(6.2, 3.6))
rng = np.random.default_rng(3)
ncert = 0
for i, bm in enumerate(BMS):
    rr8 = [r for r in g8 if r["dataset"] == bm]
    gg = [float(r["gap_pts"]) for r in rr8]
    fl = [own[(bm, r["pair"])] for r in rr8]
    mde = med[bm]["mde"]
    nc = sum(1 for g, f in zip(gg, fl) if g >= f)
    ncert += nc
    xs = np.full(len(gg), i) + rng.uniform(-0.16, 0.16, len(gg))
    ax.scatter(xs, gg, s=9, color=BLUE, alpha=0.75, edgecolor="none", zorder=3)
    over = [(x, g) for x, g, f in zip(xs, gg, fl) if g >= f]
    if over:
        ax.scatter([o[0] for o in over], [o[1] for o in over], s=22, color="#b2182b",
                   edgecolor="white", linewidth=0.4, zorder=4)
    ax.plot([i - 0.3, i + 0.3], [mde, mde], color=O, lw=2.6, solid_capstyle="butt", zorder=2)
ax.set_yscale("log")
ax.set_xticks(range(len(BMS)))
ax.set_xticklabels(BMS)
ax.set_ylabel("single-run AUROC gap / direct MDE (points, log)")
ax.set_title("Observed gaps vs direct-scale floors: %d/75 certifiable (red)" % ncert)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_gap_landscape.png"), dpi=600)
plt.close(fig)

# ---------- Fig6: MDE(K) direct ----------
fig, ax = plt.subplots(figsize=(6.4, 4.0))
Ks = np.logspace(0.7, 4.6, 240)
for bm in BMS:
    sd_blk = med[bm]["sd"] * np.sqrt(K0[bm])
    mde = np.array([(st.t.ppf(.975, k - 1) + st.t.ppf(.8, k - 1)) * sd_blk / np.sqrt(k)
                    for k in Ks])
    ax.plot(Ks, mde, lw=1.7, zorder=3, label=bm)
    m0 = (st.t.ppf(.975, K0[bm] - 1) + st.t.ppf(.8, K0[bm] - 1)) * sd_blk / np.sqrt(K0[bm])
    ax.scatter([K0[bm]], [m0], s=22, zorder=4, edgecolor="white", linewidth=0.5)
ax.axhline(5, color="gray", ls="--", lw=1.0)
ax.axhline(2, color="gray", ls=":", lw=0.9)
ax.text(5.5, 6.0, "5-point claim", color="gray", fontsize=7.5)
ax.text(5.5, 2.4, "2-point claim", color="gray", fontsize=7.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("qualified-block budget $K$ (log)")
ax.set_ylabel("direct-scale MDE (AUROC points, log)")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_title("Design chart: direct MDE(K) per suite (dots = current budgets)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_design_chart.png"), dpi=600)
plt.close(fig)

# ---------- Fig3: power curves direct at a 5-point claim ----------
fig, ax = plt.subplots(figsize=(4.6, 3.4))
Ks = np.logspace(0.8, 3.6, 200)
for bm in BMS:
    sd_blk = med[bm]["sd"] * np.sqrt(K0[bm])          # points per block
    delta = 5.0
    pow_ = []
    for k in Ks:
        ncp = delta / (sd_blk / np.sqrt(k))
        tc = st.t.ppf(.975, k - 1)
        pow_.append(st.nct.sf(tc, k - 1, ncp) + st.nct.cdf(-tc, k - 1, ncp))
    ax.plot(Ks, pow_, lw=1.6, label=bm)
    p0 = st.nct.sf(st.t.ppf(.975, K0[bm] - 1), K0[bm] - 1,
                   delta / (sd_blk / np.sqrt(K0[bm]))) + \
        st.nct.cdf(-st.t.ppf(.975, K0[bm] - 1), K0[bm] - 1,
                   delta / (sd_blk / np.sqrt(K0[bm])))
    ax.scatter([K0[bm]], [p0], s=20, zorder=4, edgecolor="white", linewidth=0.5)
ax.axhline(0.8, color="gray", ls="--", lw=0.9)
ax.text(8, 0.82, "power 0.8", color="gray", fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("qualified-block budget $K$ (log)")
ax.set_ylabel("power at a 5-point AUROC difference")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("Segment-level paired $t$: direct-scale power at 5 points", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_power_curves.png"), dpi=600)
plt.close(fig)
print("fig3/5/6 regenerated on the direct scale")

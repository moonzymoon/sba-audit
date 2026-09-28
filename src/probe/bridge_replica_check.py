# -*- coding: utf-8 -*-
"""STAGE 1 (feedback-95 C1): empirical adjudication of the AUROC bridge at replica level.

For each (benchmark, scorer): draw B paired-design replicas (same rng state for g and y),
compute per replica:
  u_direct[b] = roc_auc_score(y*, |g*|)                    (replica's true AUROC)
  u_bridge[b] = (mean(g*) - (pi^2-1/2) - (pi-1/2)/T) / (2*pi*(1-pi))
Pairwise (deep-pca, deep-iforest, pca-iforest per benchmark):
  dAUC_direct vs dAUC_bridge = (mean(g*A-g*B))/(2*pi*(1-pi))
Decision inputs: |u_direct-u_bridge|, Var ratio, slope(dAUC_direct ~ dbar)/(2pi(1-pi)),
max pair error vs the benchmark's MDE. Saves results/bridge_replica_check.csv
"""
import csv
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "D:/0keyan/gongzuo1/paper 15SCI/src")
sys.path.insert(0, "D:/0keyan/gongzuo1/paper 15SCI/src/bootstrap")
from common.blocks import build_blocks, resample_blocks  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from bootstrap.e3_clean import load_goodness  # noqa: E402

R = "D:/0keyan/gongzuo1/paper 15SCI/02_shiyanjilu/results/"
SC = ["cmhmil_seed7", "pca", "iforest"]
BMS = {"SMD": 70.6, "PSM": 26.6, "MSL": 43.1, "SMAP": 49.2, "WADI": 610.1}
B = 400
rng = np.random.default_rng(20260905)

out = []
for bm, mde_pts in BMS.items():
    data = {}
    for sc in SC:
        g, y = load_goodness(sc, bm)
        data[sc] = (g, y)
    y0 = data[SC[0]][1]
    T = len(y0)
    blocks = build_blocks(events_from_binary(y0), T)
    ud = {sc: np.empty(B) for sc in SC}
    ub = {sc: np.empty(B) for sc in SC}
    dbar = {pair: np.empty(B) for pair in
            [(SC[i], SC[j]) for i in range(len(SC)) for j in range(i + 1, len(SC))]}
    for b in range(B):
        reps = {}
        for sc in SC:
            g, _ = data[sc]
            state = rng.bit_generator.state
            gs, _, _ = resample_blocks(g, blocks, rng)
            rng.bit_generator.state = state
            ys, _, _ = resample_blocks(y0.astype(float), blocks, rng)
            ys = ys.astype(int)
            reps[sc] = (gs, ys)
        for sc in SC:
            gs, ys = reps[sc]
            pi = ys.mean()
            ud[sc][b] = roc_auc_score(ys, np.abs(gs))
            ub[sc][b] = (gs.mean() - (pi ** 2 - 0.5) - (pi - 0.5) / T) / (2 * pi * (1 - pi))
        for (a, bb) in dbar:
            dbar[(a, bb)][b] = reps[a][0].mean() - reps[bb][0].mean()
    for sc in SC:
        vr = ud[sc].var() / max(ub[sc].var(), 1e-12)
        out.append(dict(bm=bm, quantity="scorer", name=sc, B=B,
                        med_abs_err_pts=100 * np.median(np.abs(ud[sc] - ub[sc])),
                        max_abs_err_pts=100 * np.max(np.abs(ud[sc] - ub[sc])),
                        var_ratio=vr, slope_ratio=np.nan, mde_pts=mde_pts,
                        err_over_mde=100 * np.max(np.abs(ud[sc] - ub[sc])) / mde_pts))
    pi = y0.mean()
    bridge_factor = 2 * pi * (1 - pi)
    for (a, bb) in dbar:
        dB = dbar[(a, bb)] / bridge_factor
        dD = ud[a] - ud[bb]
        slope = np.polyfit(dbar[(a, bb)], dD, 1)[0]
        err = np.abs(dD - dB) * 100
        out.append(dict(bm=bm, quantity="pair", name=f"{a}-{bb}", B=B,
                        med_abs_err_pts=np.median(err), max_abs_err_pts=err.max(),
                        var_ratio=np.nan, slope_ratio=slope / bridge_factor,
                        mde_pts=mde_pts, err_over_mde=err.max() / mde_pts))
    print(bm, "done")

with open(R + "bridge_replica_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)

sc_rows = [r for r in out if r["quantity"] == "scorer"]
pr_rows = [r for r in out if r["quantity"] == "pair"]
vr = np.array([r["var_ratio"] for r in sc_rows])
sr = np.array([r["slope_ratio"] for r in pr_rows if not np.isnan(r["slope_ratio"])])
eo = np.array([r["err_over_mde"] for r in out])
print("var_ratio range: %.3f - %.3f | slope_ratio range: %.3f - %.3f"
      % (vr.min(), vr.max(), sr.min(), sr.max()))
print("max pair err as %% of MDE: %.4f" % max(
    r["err_over_mde"] for r in pr_rows))
verd = ("PASS_A" if (0.75 <= vr.min() and vr.max() <= 1.33
                     and 0.8 <= sr.min() and sr.max() <= 1.25
                     and max(r["err_over_mde"] for r in pr_rows) < 0.1) else "TRIGGER_B")
print("DECISION:", verd)

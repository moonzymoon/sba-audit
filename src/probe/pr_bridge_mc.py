# -*- coding: utf-8 -*-
"""PR/AP bridge existence MC (deep-derivation item 6).

Question: no exact per-window linear bridge to AP exists (adjacent-transposition
argument: AP increments depend on the running anomaly count, a global path
quantity, while any sum_t h(rank_t, y_t) has increments depending only on the
two swapped ranks). Empirical fallback: across segment-block replicas (paired
scale — both detectors see the same drawn blocks), is delta-AP an approximately
linear function of the exact delta-AUROC bridge, with residual small relative
to the cell's AUROC-scale MDE?

For each replica b (same rng state for A, B, and y -> identical block draws):
  dbar_b   = mean(gA* - gB*)                      (rank-margin paired mean)
  dAUC_b   = dbar_b / (2*pi*(1-pi))               (exact bridge, R7)
  dAP_b    = AP(|gA*|, y*) - AP(|gB*|, y*)        (direct, no bridge)
Regress dAP on dAUC -> slope, R^2, residual SD.
"""
import sys

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, "D:/0科研/工作1/第15篇SCI/src")
sys.path.insert(0, "D:/0科研/工作1/第15篇SCI/src/bootstrap")
from common.blocks import build_blocks, resample_blocks  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from bootstrap.e3_clean import load_goodness  # noqa: E402

R = "D:/0科研/工作1/第15篇SCI/02_实验记录/results/"
CELLS = [("SMD", "cmhmil_seed7", "pca", 70.6),
         ("PSM", "cmhmil_seed7", "iforest", 26.6),
         ("WADI", "cmhmil_seed7", "pca", 610.1)]
B = 400
rng = np.random.default_rng(20260904)

out = []
for bm, sa, sb, mde_auc in CELLS:
    gA, y = load_goodness(sa, bm)
    gB, yb = load_goodness(sb, bm)
    assert np.array_equal(y, yb)
    blocks = build_blocks(events_from_binary(y), len(y))
    Tn, pi = len(y), y.mean()
    sA = np.abs(gA)  # = rank/T, the within-run score
    sB = np.abs(gB)
    dAUC, dAP = np.empty(B), np.empty(B)
    for b in range(B):
        state = rng.bit_generator.state
        gAs, _, _ = resample_blocks(gA, blocks, rng)
        rng.bit_generator.state = state
        gBs, _, _ = resample_blocks(gB, blocks, rng)
        rng.bit_generator.state = state
        ys, _, _ = resample_blocks(y.astype(float), blocks, rng)
        ys = ys.astype(int)
        dbar = (gAs - gBs).mean()
        dAUC[b] = dbar / (2 * pi * (1 - pi))
        dAP[b] = (average_precision_score(ys, np.abs(gAs)) -
                  average_precision_score(ys, np.abs(gBs)))
    x, yy = dAUC, dAP
    slope, icept = np.polyfit(x, yy, 1)
    resid = yy - (slope * x + icept)
    r2 = 1 - resid.var() / yy.var()
    print(f"{bm} {sa[:6]}-{sb}: slope={slope:.3f} R2={r2:.4f} "
          f"residSD={resid.std()*100:.3f} APpts | slope*MDE_AUOC="
          f"{abs(slope)*mde_auc:.1f} APpts | SD(dAUC)={x.std():.4f}")
    out.append(dict(bm=bm, pair=f"{sa}-{sb}", B=B, slope=slope, r2=r2,
                    resid_sd_ap=resid.std(), sd_dauc=x.std(),
                    slope_x_mde=abs(slope) * mde_auc))

import csv  # noqa: E402
with open(R + "pr_bridge_mc.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print("saved pr_bridge_mc.csv")

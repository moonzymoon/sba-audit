# -*- coding: utf-8 -*-
"""STAGE 2 adjudication (feedback-95 C1): DIRECT-scale evaluation noise and verdict.

For each benchmark x 6 scorers, B=400 replicas (paired rng state):
  u_direct[b] = roc_auc_score(y*, |g*|)
Per pair (75): SD(D_b) with D_b = u_A - u_B  -> MDE_direct = 2.8*SD(D)*100 pts
Actual single-run gap per pair from original caches.
Verdict: gap >= MDE_direct (certifiable on the direct scale).
Saves results/direct_scale_verdict.csv; prints summary.
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
SC = ["cmhmil_seed7", "iforest", "pca", "gmm", "ocsvm", "lof"]
BMS = ["SMD", "PSM", "MSL", "SMAP", "WADI"]
B = 400
rng = np.random.default_rng(20260906)

out = []
cert = 0
tot = 0
for bm in BMS:
    gs, y = {}, None
    for sc in SC:
        g, y_ = load_goodness(sc, bm)
        y = y_
        gs[sc] = g
    T = len(y)
    blocks = build_blocks(events_from_binary(y), T)
    ud = {sc: np.empty(B) for sc in SC}
    for b in range(B):
        # save state ONCE per replicate so all scorers AND labels share the same block indices
        state = rng.bit_generator.state
        for sc in SC:
            rng.bit_generator.state = state
            g_star, _, _ = resample_blocks(gs[sc], blocks, rng)
            rng.bit_generator.state = state
            ys, _, _ = resample_blocks(y.astype(float), blocks, rng)
            ud[sc][b] = roc_auc_score(ys.astype(int), np.abs(g_star))
    aucs = {sc: roc_auc_score(y, np.abs(gs[sc])) for sc in SC}
    for i in range(len(SC)):
        for j in range(i + 1, len(SC)):
            a, b2 = SC[i], SC[j]
            D = ud[a] - ud[b2]
            sd = D.std(ddof=1)
            mde_direct = 2.8 * sd * 100
            gap = abs(aucs[a] - aucs[b2]) * 100
            c = gap >= mde_direct
            cert += int(c)
            tot += 1
            out.append(dict(bm=bm, pair=f"{a}-{b2}", B=B, sd_direct_pts=sd * 100,
                            mde_direct_pts=mde_direct, gap_pts=gap,
                            certifiable_direct=c))
    print(bm, "done")

with open(R + "direct_scale_verdict.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
for bm in BMS:
    rows = [r for r in out if r["bm"] == bm]
    print("%s: median SD=%.2f pts, median MDE_direct=%.1f pts, certifiable %d/%d"
          % (bm, np.median([r["sd_direct_pts"] for r in rows]),
             np.median([r["mde_direct_pts"] for r in rows]),
             sum(r["certifiable_direct"] for r in rows), len(rows)))
print("TOTAL certifiable on direct scale: %d/%d" % (cert, tot))

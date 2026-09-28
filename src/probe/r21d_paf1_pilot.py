# -*- coding: utf-8 -*-
"""Plan A Card 3d: PA-F1 direct floors, two pilot cells.

Mirrors direct_scale_verdict.py conventions exactly (paired rng state, B=400,
MDE = 2.8*SD*100 in F1 percentage points). PA = point-adjust (Xu et al. 2018):
within each true anomaly segment, if >=1 position is flagged, set the whole
true segment to flagged; then window-level P/R/F1. Detection rate per replica
side = anomaly rate of the replica labels (top-quantile threshold).
Cells: (SMD, cmhmil_seed7 vs pca), (MSL, cmhmil_seed7 vs pca).
"""
import sys
import numpy as np

sys.path.insert(0, r"D:\0科研\工作1\第15篇SCI\src")
sys.path.insert(0, r"D:\0科研\工作1\第15篇SCI\src\bootstrap")
from common.blocks import build_blocks, resample_blocks  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from bootstrap.e3_clean import load_goodness  # noqa: E402

B = 400
CELLS = [("SMD", "cmhmil_seed7", "pca"), ("MSL", "cmhmil_seed7", "pca")]
rng = np.random.default_rng(20260910)


def segments(mask):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(idx[::2], idx[1::2]))


def pa_f1(score, y, rate):
    thr = np.quantile(score, 1.0 - rate)
    p = score >= thr
    adj = p.copy()
    for s, e in segments(y):
        if p[s:e].any():
            adj[s:e] = True
    tp = float((adj & (y == 1)).sum())
    fp = float((adj & (y == 0)).sum())
    fn = float(((~adj) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 200.0 * prec * rec / (prec + rec) if prec + rec else 0.0


out = []
for bm, sa, sb in CELLS:
    gA, y = load_goodness(sa, bm)      # g is rank-signed margin; |g| ordering == score ordering
    gB, _ = load_goodness(sb, bm)
    # recover ascending score order from rank margin: rank r -> |g| monotone in score
    sA = np.abs(gA); sB = np.abs(gB)
    T = len(y)
    blocks = build_blocks(events_from_binary(y.astype(np.int8)), T)
    pi = float(y.mean())
    fA = np.empty(B); fB = np.empty(B)
    for b in range(B):
        state = rng.bit_generator.state
        sA_star, _, _ = resample_blocks(sA, blocks, rng)
        rng.bit_generator.state = state
        ysA, _, _ = resample_blocks(y.astype(float), blocks, rng)
        rate = float(ysA.mean())
        fA[b] = pa_f1(sA_star, np.asarray(ysA), rate)
        state = rng.bit_generator.state
        sB_star, _, _ = resample_blocks(sB, blocks, rng)
        rng.bit_generator.state = state
        ysB, _, _ = resample_blocks(y.astype(float), blocks, rng)
        fB[b] = pa_f1(sB_star, np.asarray(ysB), float(ysB.mean()))
    D = fA - fB
    sd = D.std(ddof=1)
    mde = 2.8 * sd
    base_a = pa_f1(sA, y, pi); base_b = pa_f1(sB, y, pi)
    rec = dict(bm=bm, pair="%s-%s" % (sa, sb), B=B,
               sd_PA_F1_pts=round(float(sd), 3), MDE_PA_F1_pts=round(float(mde), 2),
               PA_F1_A=round(float(base_a), 4), PA_F1_B=round(float(base_b), 4),
               gap_pts=round(float(abs(base_a - base_b)), 2))
    out.append(rec)
    print(rec)

import json
json.dump(out, open(r"D:\0科研\工作1\第15篇SCI\02_实验记录\results\r21d_paf1_pilot.json", "w"), indent=1)
print("saved r21d_paf1_pilot.json")

# -*- coding: utf-8 -*-
"""R32-1: full direct pricing for PA-F1 and AP over ALL 75 canonical pairs
(upgrades the 4-cell pilot). Same conventions as r21d/r26b: paired rng state,
B=400, MDE = 2.8*SD*100 (metric points). AP via sklearn on |g|; PA-F1 via
per-replica anomaly-rate threshold + point-adjust. Output r32_full_pricing.json."""
import os, sys, json, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, r"D:\0keyan\gongzuo1\paper 15SCI\src")
from bootstrap.e3_clean import load_goodness  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from common.blocks import build_blocks, resample_blocks  # noqa: E402

RES = r"D:\0keyan\gongzuo1\paper 15SCI\02_shiyanjilu\results"
B = 400
d = pd.read_csv(os.path.join(RES, "direct_scale_verdict.csv"), encoding="utf-8-sig")

def segments(mask):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(idx[::2].tolist(), idx[1::2].tolist()))

def pa_f1(score, y, rate):
    thr = np.quantile(score, 1.0 - rate)
    p = score >= thr
    adj = p.copy()
    for s, e in segments(y):
        if p[s:e].any():
            adj[s:e] = True
    tp = float((adj & (y == 1)).sum()); fp = float((adj & (y == 0)).sum()); fn = float(((~adj) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 200.0 * pr * rc / (pr + rc) if pr + rc else 0.0

out = []
groups = {}
for _, r in d.iterrows():
    groups.setdefault(r["bm"], []).append(r["pair"].split("-"))

for bm, pairs in groups.items():
    scorers = sorted({s for p in pairs for s in p})
    gs, y = {}, None
    for sc in scorers:
        g, y_ = load_goodness(sc, bm)
        y = y_.astype(float)
        gs[sc] = g
    T = len(y)
    blocks = build_blocks(events_from_binary(y.astype(np.int8)), T)
    ap_d = {sc: np.empty(B) for sc in scorers}
    pf_d = {sc: np.empty(B) for sc in scorers}
    master_rng = np.random.default_rng(777000)
    for b in range(B):
        # single shared realization: one state, all scorers AND labels use same block indices
        state = master_rng.bit_generator.state
        master_rng.bit_generator.state = state
        ys, _, _ = resample_blocks(y, blocks, master_rng)
        yi = ys.astype(int)
        rate = float(ys.mean())
        for sc in scorers:
            master_rng.bit_generator.state = state
            g_star, _, _ = resample_blocks(gs[sc], blocks, master_rng)
            ap_d[sc][b] = average_precision_score(yi, np.abs(g_star))
            pf_d[sc][b] = pa_f1(np.abs(g_star), yi, rate)
        if (b + 1) % 100 == 0:
            print(bm, b + 1, flush=True)
    aps = {sc: float(average_precision_score(y.astype(int), np.abs(gs[sc]))) for sc in scorers}
    pfs = {sc: pa_f1(np.abs(gs[sc]), y.astype(int), float(y.mean())) for sc in scorers}
    for a, b2 in itertools.combinations(scorers, 2):
        rec = dict(bm=bm, pair=f"{a}-{b2}")
        for met, dd, base in [("AP", ap_d, aps), ("PAF1", pf_d, pfs)]:
            D = dd[a] - dd[b2]
            sd = float(D.std(ddof=1))
            gap = abs(base[a] - base[b2])
            rec[met] = dict(sd=round(sd, 4), mde=round(2.8 * sd * 100, 1),
                            gap=round(gap * 100, 1), cert=bool(gap * 100 >= 2.8 * sd * 100))
        out.append(rec)
    print(bm, "done:", len(out), flush=True)

json.dump(out, open(os.path.join(RES, "r32_full_pricing.json"), "w"), indent=1)
for met in ("AP", "PAF1"):
    certs = [r for r in out if r[met]["cert"]]
    mdes = [r[met]["mde"] for r in out]
    gaps = [r[met]["gap"] for r in out]
    above = sum(1 for r in out if r[met]["gap"] >= 5.0)
    print(met, "pairs:", len(out), "| certifiable:", len(certs), "| median MDE:", round(float(np.median(mdes)), 1),
          "| gaps>=5pts:", above)
print("saved r32_full_pricing.json")

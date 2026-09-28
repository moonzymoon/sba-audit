

def _orient(rows):
    """Orientation consistency: if the point estimate falls outside [lo,hi]
    The containment test is an orientation invariant for this test: the
    acceptance set of a sign-flip inversion of the (weighted) mean statistic
    is always centered on the weighted mean, so a non-containing interval
    signals an orientation/scale flip, not genuine exclusion.
    but inside the flipped interval, flip the interval to the pair's reported
    (positive-gap) orientation (R56 post-processing, made durable here)."""
    for r in rows:
        pt = r.get('point_pts', r.get('point'))
        if 'CI_pts' in r:
            lo, hi = r['CI_pts']
        else:
            lo, hi = r.get('lo'), r.get('hi')
        if pt is None or lo is None or hi is None:
            continue
        if not (lo <= pt <= hi) and (-hi <= pt <= -lo):
            if 'CI_pts' in r:
                r['CI_pts'] = [-hi, -lo]
                for k in ('set_envelope', 'set_envelope_full'):
                    if k in r:
                        e = r[k]; r[k] = [-e[1], -e[0]]
            else:
                r['lo'], r['hi'] = -hi, -lo
    return rows


# -*- coding: utf-8 -*-
"""R30-3: exact/MC confidence intervals for ALL audited pairs (75 canonical +
10 EMG) by sign-permutation inversion on the length-weighted block-mean
statistic. Exact enumeration K<=20; vectorized MC (m=999, fixed sign matrix
per pair for a stable p-function) beyond. Report per pair: point estimate,
CI (principal component envelope), covers-zero, width. Output r30_exact_ci_all.json."""
import os, sys, json, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\0科研\工作1\第15篇SCI\src")
from bootstrap.e3_clean import load_goodness  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from common.blocks import build_blocks, block_stats  # noqa: E402

ALPHA, M = 0.05, 999
RES = r"D:\0科研\工作1\第15篇SCI\02_实验记录\results"
d = pd.read_csv(os.path.join(RES, "direct_scale_verdict.csv"), encoding="utf-8-sig")
pairs = [(r["bm"], *r["pair"].split("-")) for _, r in d.iterrows()]
import json as _j
emg = _j.load(open(os.path.join(RES, "r24_emg_summary.json")))
pairs += [("EMG", *p["pair"].split("-")) for p in emg["floors"]]
print("total pairs:", len(pairs))

def setup(bm, sa, sb):
    ga, y = load_goodness(sa, bm)
    gb, _ = load_goodness(sb, bm)
    blocks = build_blocks(events_from_binary(y.astype(np.int8)), len(ga))
    dd = ga - gb
    bs = block_stats(dd, blocks, min_seg=10)
    dj = np.array([b["m"] for b in bs])
    wj = np.array([b["n"] for b in bs], float)
    w = wj / wj.sum()
    pi = float(y.mean())
    return dj, w, 2 * pi * (1 - pi)

def ci_for(bm, sa, sb, rng):
    dj, w, scale = setup(bm, sa, sb)
    K = len(dj)
    center = float((w * dj).sum())
    if K <= 20:
        import itertools as it
        S = np.array(list(it.product([-1.0, 1.0], repeat=K)))
        def pfun(delta):
            x = (w * (dj - delta))
            obs = abs(x.sum())
            dist = np.abs(S @ x)
            return float(np.mean(dist >= obs - 1e-12))
    else:
        S = rng.choice([-1.0, 1.0], size=(M, K))
        def pfun(delta):
            x = (w * (dj - delta))
            obs = abs(x.sum())
            dist = np.abs(S @ x)
            return float((np.sum(dist >= obs - 1e-12) + 1) / (M + 1))
    half = 3.0 * dj.std(ddof=1) / np.sqrt(K)
    hi = center + half
    while pfun(hi) > ALPHA and hi < center + 200 * half:
        hi *= 1.5 - 0.5 * center / hi if hi != 0 else 1
        hi = hi + half
        half *= 1.3
    lo = center - half
    while pfun(lo) > ALPHA and lo > center - 200 * half:
        lo = lo - half
        half *= 1.3
    def bisect(a, b):
        fa, fb = pfun(a) > ALPHA, pfun(b) > ALPHA
        if fa == fb:
            return b if fa else a
        for _ in range(22):
            mmid = 0.5 * (a + b)
            if (pfun(mmid) > ALPHA) == fa:
                a = mmid
            else:
                b = mmid
        return 0.5 * (a + b)
    lo_b = bisect(lo, center)
    hi_b = bisect(center, hi)
    ci = sorted([lo_b / scale * 100, hi_b / scale * 100])
    return dict(bm=bm, pair=f"{sa}-{sb}", K=K, exact=bool(K <= 20),
                point=round(abs(center) / scale * 100, 1),
                lo=round(ci[0], 1), hi=round(ci[1], 1),
                width=round(ci[1] - ci[0], 1), covers0=bool(ci[0] <= 0 <= ci[1]))

rng = np.random.default_rng(20260911)
out = []
for bm, sa, sb in pairs:
    out.append(ci_for(bm, sa, sb, rng))
    if len(out) % 15 == 0:
        print(len(out), "done", flush=True)

json.dump(_orient(out), open(os.path.join(RES, "r30_exact_ci_all.json"), "w"), indent=1)
cov = sum(r["covers0"] for r in out)
w_med = float(np.median([r["width"] for r in out]))
print("covers zero: %d/%d | median width %.1f pts | saved" % (cov, len(out), w_med))



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
"""R27d v3: exact CIs by test inversion — search outward from the weighted
point estimate (where p is maximal), doubling then bisecting on each side.
Weighted (length) sign-flip statistic; enumeration K<=20 else MC m=999."""
import os, sys, json, itertools
import numpy as np

sys.path.insert(0, r"D:\0科研\工作1\第15篇SCI\src")
from bootstrap.e3_clean import load_goodness  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from common.blocks import build_blocks  # noqa: E402

ALPHA, M = 0.05, 999
PAIRS = [("SMD", "cmhmil_seed7", "pca"), ("EMG", "gmm", "lof"), ("MSL", "cmhmil_seed7", "pca")]

def setup(ds, sa, sb):
    ga, y = load_goodness(sa, ds)
    gb, _ = load_goodness(sb, ds)
    T = len(ga)
    blocks = build_blocks(events_from_binary(y.astype(np.int8)), T)
    d = ga - gb
    dj, wj = [], []
    for k, s, e in blocks:
        if k == "event" and e - s < 10:
            continue
        dj.append(d[s:e].mean())
        wj.append(e - s)
    dj, wj = np.array(dj), np.array(wj, float)
    w = wj / wj.sum()
    pi = float(y.mean())
    return dj, w, 2 * pi * (1 - pi), len(dj)

def make_pfun(dj, w):
    center = float((w * dj).sum())
    K = len(dj)
    signs = (np.array(list(itertools.product([-1.0, 1.0], repeat=K)))
             if K <= 20 else None)
    rng = np.random.default_rng(20260910)
    def p(delta):
        x = dj - delta
        obs = abs((w * x).sum())
        if signs is not None:
            return float(np.mean(np.abs(signs @ (w * x)) >= obs - 1e-12))
        cnt = sum(1 for _ in range(M)
                  if abs((rng.choice([-1.0, 1.0], K) * w * x).sum()) >= obs - 1e-12)
        return (cnt + 1) / (M + 1)
    return p, center

def bisect(p, a, b, want_inside):
    # a inside, b outside (or vice versa); find boundary
    fa, fb = p(a) > ALPHA, p(b) > ALPHA
    if fa == fb:
        return b if fa else a
    for _ in range(24):
        m = 0.5 * (a + b)
        if (p(m) > ALPHA) == fa:
            a = m
        else:
            b = m
    return 0.5 * (a + b)

out = []
for ds, sa, sb in PAIRS:
    dj, w, scale, K = setup(ds, sa, sb)
    p, center = make_pfun(dj, w)
    step = 2 * dj.std(ddof=1) / np.sqrt(len(dj)) if K > 2 else abs(center) + 1
    # upper
    hi = center + step
    while p(hi) > ALPHA and hi < center + 200 * step:
        hi += step
        step *= 1.5
    hi_b = bisect(p, hi - step, hi, True) if hi > center + step else hi
    # lower
    lo = center - step
    while p(lo) > ALPHA and lo > center - 200 * step:
        lo -= step
        step *= 1.5
    lo_b = bisect(p, center - (hi_b - center) * 3, lo, False) if lo < center else center - (hi_b - center)
    # ensure lo below center: lo was outside -> boundary between lo and center side
    lo_b = bisect(p, lo, center, False) if p(lo) <= ALPHA else bisect(p, lo, center, False)
    # connected component of the confidence set that contains the point estimate
    grid_in = np.linspace(lo_b, hi_b, 401)
    flags = [p(g) > ALPHA for g in grid_in]
    ci_raw = sorted({lo_b, hi_b})
    gaps = sum(1 for i in range(1, len(flags)) if flags[i - 1] and not flags[i])
    comp_l = lo_b / scale * 100
    comp_r = hi_b / scale * 100
    c0 = center
    i = min(range(len(grid_in)), key=lambda k: abs(grid_in[k] - c0))
    l = i
    while l > 0 and flags[l - 1]:
        l -= 1
    r = i
    while r < len(flags) - 1 and flags[r + 1]:
        r += 1
    comp_l, comp_r = grid_in[l] / scale * 100, grid_in[r] / scale * 100
    ci = sorted({comp_l, comp_r})
    rec = dict(bm=ds, pair=f"{sa}-{sb}", K=K, exact=bool(K <= 20),
               point_pts=round(abs(center) / scale * 100, 1),
               CI_pts=[round(ci[0], 1), round(ci[1], 1)],
               set_envelope=[round(min(comp_l, comp_r), 1), round(max(comp_l, comp_r), 1)],
               set_envelope_full=[round(min(lo_b, hi_b) / scale * 100, 1), round(max(lo_b, hi_b) / scale * 100, 1)],
               excursions=int(gaps),
               covers_zero=bool(ci[0] <= 0 <= ci[1]))
    out.append(rec)
    print(rec, flush=True)

json.dump(_orient(out), open(r"D:\0科研\工作1\第15篇SCI\02_实验记录\results\r27d_exact_ci.json", "w"), indent=1)
print("saved")

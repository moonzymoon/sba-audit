# -*- coding: utf-8 -*-
"""Plan A Card 3e: block-length factor for the synthetic grid (L in {50,200,800}).

One representative cell (K=12, phi=0.5, sigma_b=0.5), B=2000 per L.
Checks that the r -> size mapping 2*Phi(-1.96/sqrt(r)) is block-length invariant.
Mirrors synthetic_grid.py machinery (R6) without modifying the frozen original.
"""
import numpy as np
from scipy import stats
from scipy.signal import lfilter

SEED = 20260910
B = 2000
K, PHI, SB = 12, 0.5, 0.5

def run_L(L, seed):
    rng = np.random.default_rng(seed)
    T = 2 * K * L
    e = rng.normal(0, np.sqrt(1 - PHI**2), T)
    x = lfilter([1.0], [1.0, -PHI], e)
    be = rng.normal(0, SB, K)
    bn = rng.normal(0, SB, K)
    g = x.reshape(2 * K, L).copy()
    for j in range(2 * K):
        g[j] += be[j // 2] if j % 2 == 1 else bn[j // 2]
    g = g.reshape(-1)
    ev = np.arange(2 * K)[1::2]
    no = np.arange(2 * K)[0::2]
    v_g = g.var(ddof=1)
    rej_win = 0
    gbar_star = np.empty(B)
    for b in range(B):
        seq_e = rng.choice(ev, K, replace=True)
        seq_n = rng.choice(no, K, replace=True)
        gs = np.concatenate([np.concatenate([g[s * L:(s + 1) * L], g[t * L:(t + 1) * L]])
                             for s, t in zip(seq_n, seq_e)])
        d = g - gs
        rej_win += int(stats.ttest_1samp(d, 0.0).pvalue < .05)
        gbar_star[b] = gs.mean()
    emp = rej_win / B
    r_mean = float(T * gbar_star.var(ddof=1) / (2 * v_g))
    pred = float(2 * stats.norm.cdf(-1.96 / np.sqrt(max(r_mean, 1e-9))))
    return L, round(r_mean, 2), round(emp, 3), round(pred, 3)

rows = []
for i, L in enumerate((50, 200, 800)):
    r = run_L(L, SEED + i)
    rows.append(r)
    print("L=%d  r_mean=%.2f  empirical size=%.3f  predicted=%.3f" % r)

import json
json.dump([{"L": r[0], "r_mean": r[1], "size_emp": r[2], "size_pred": r[3]} for r in rows],
          open(r"D:\0keyan\gongzuo1\paper 15SCI\02_shiyanjilu\results\r21e_length_scan.json", "w"), indent=1)
print("saved r21e_length_scan.json")

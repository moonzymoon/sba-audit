# -*- coding: utf-8 -*-
"""R27f: inverse-variance block weighting vs unweighted vs length-weighted
segment-level tests. For 5 canonical suites (cmhmil_seed7 vs pca):
- B=1000 paired replicas; per-replica qualified-block mean differences d*_j.
- sigma_hat_j from the first half of replicas; tests evaluated on second half
  (no sigma optimism). Statistics: unweighted mean / length-weighted mean /
  IV-weighted mean. Size at delta=0; power at delta = 5 AUROC points
  (via the exact full-run bridge scale 2 pi (1-pi)).
"""
import os, sys, json
import numpy as np
from scipy import stats as st

sys.path.insert(0, r"D:\0keyan\gongzuo1\paper 15SCI\src")
from bootstrap.e3_clean import load_goodness  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from common.blocks import build_blocks  # noqa: E402

B = 1000

def _resample(g, blocks, rng):
    import numpy as np
    parts = []
    ev = [(s, e) for k, s, e in blocks if k == "event"]
    no = [(s, e) for k, s, e in blocks if k == "normal"]
    for k, s, e in blocks:
        pool = ev if k == "event" else no
        j = pool[rng.integers(0, len(pool))]
        parts.append(g[j[0]:j[1]])
    gs = np.concatenate(parts)
    return gs[:len(g)] if len(gs) >= len(g) else np.pad(gs, (0, len(g) - len(gs)))

out = []
for ds in ["SMD", "PSM", "MSL", "SMAP", "WADI"]:
    ga, y = load_goodness("cmhmil_seed7", ds)
    gb, _ = load_goodness("pca", ds)
    T = len(ga)
    blocks = build_blocks(events_from_binary(y.astype(np.int8)), T)
    slots, kinds = [], []
    for k, s, e in blocks:
        if k == "event" and e - s < 10:
            continue
        slots.append((s, e)); kinds.append(k)
    K = len(slots)
    lens = np.array([e - s for s, e in slots], float)
    wl = lens / lens.sum()
    pi = float(y.mean())
    scale = 2 * pi * (1 - pi)
    D = np.empty((B, K))
    for b in range(B):
        rng = np.random.default_rng(20260910 + b)
        state = rng.bit_generator.state
        ga_s = _resample(ga, blocks, rng)
        rng.bit_generator.state = state
        gb_s = _resample(gb, blocks, rng)
        d = ga_s - gb_s
        for j, (s, e) in enumerate(slots):
            D[b, j] = d[s:e].mean()
    h = B // 2
    sig2 = D[:h].var(axis=0, ddof=1)
    w_iv = 1.0 / (sig2 + 1e-12)
    w_iv = w_iv / w_iv.sum()
    w_u = np.ones(K) / K
    D2 = D[h:]
    def rej(w, delta_pts=0.0):
        delta = delta_pts / 100.0 * scale
        stat = (D2 - delta) @ w
        # t-style studentization with replica-based SE is optimistic; use the
        # design convention: normal deviate at alpha .05, one-sample
        se = float(np.sqrt(np.sum((w ** 2) * sig2)))  # design SE from first-half sigma
        return float(np.mean(np.abs(stat) / se > 1.96))
    row = dict(dataset=ds, K=K,
               size_unw=round(rej(w_u), 3), size_len=round(rej(wl), 3), size_iv=round(rej(w_iv), 3),
               pow5_unw=round(rej(w_u, 5.0), 3), pow5_len=round(rej(wl, 5.0), 3),
               pow5_iv=round(rej(w_iv, 5.0), 3))
    out.append(row)
    print(row, flush=True)

json.dump(out, open(r"D:\0keyan\gongzuo1\paper 15SCI\02_shiyanjilu\results\r27f_weights.json", "w"), indent=1)
print("saved r27f_weights.json")

# -*- coding: utf-8 -*-
"""R27b: composition-fixed null via length-stratified redraw.

Each slot (event or normal, chronological) draws a same-kind donor block whose
length is within +-20% of the slot's original length (nearest-length fallback),
so block composition (count and length profile) is held ~fixed while within-kind
selection remains -> the induced window-level size isolates the selection share.
Cells: cmhmil_seed7 on the five canonical benchmarks, B=2000.
"""
import os, sys, json
import numpy as np
from scipy import stats as st

sys.path.insert(0, r"D:\0keyan\gongzuo1\paper 15SCI\src")
from bootstrap.e3_clean import load_goodness  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from common.blocks import build_blocks  # noqa: E402

B = 2000

def redraw_stratified(g, blocks, rng, tol=0.2):
    ev = [(s, e) for k, s, e in blocks if k == "event"]
    no = [(s, e) for k, s, e in blocks if k == "normal"]
    parts = []
    for slots, kind in ((ev, "event"), (no, "normal")):
        lens = np.array([e - s for s, e in slots])
        arrs = [g[s:e] for s, e in slots]
        for (s, e) in slots:
            L = e - s
            tol_abs = max(2, int(tol * L))
            cand = [i for i, (a, b) in enumerate(slots) if abs((b - a) - L) <= tol_abs]
            if not cand:
                cand = [int(np.argmin(np.abs(lens - L)))]
            j = cand[rng.integers(0, len(cand))]
            parts.append(arrs[j])
    gs = np.concatenate(parts)
    return gs[:len(g)] if len(gs) >= len(g) else np.pad(gs, (0, len(g) - len(gs)))

out = []
for ds in ["SMD", "PSM", "MSL", "SMAP", "WADI"]:
    g, y = load_goodness("cmhmil_seed7", ds)
    T = len(g)
    blocks = build_blocks(events_from_binary(y.astype(np.int8)), T)
    v_g = g.var(ddof=1)
    gbar = np.empty(B)
    rej = 0
    for b in range(B):
        rng = np.random.default_rng(20260910 + b)
        gs = redraw_stratified(g, blocks, rng)
        d = g - gs
        if st.ttest_1samp(d, 0.0).pvalue < .05:
            rej += 1
        gbar[b] = gs.mean()
    r = T * gbar.var(ddof=1) / (2 * v_g)
    size = rej / B
    out.append(dict(dataset=ds, r_compfixed=round(float(r), 2),
                    size_compfixed=round(float(size), 3)))
    print(ds, out[-1], flush=True)

json.dump(out, open(r"D:\0keyan\gongzuo1\paper 15SCI\02_shiyanjilu\results\r27b_compfixed.json", "w"), indent=1)
print("saved r27b_compfixed.json")

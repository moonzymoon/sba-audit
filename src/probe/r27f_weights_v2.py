# -*- coding: utf-8 -*-
"""R27f v2: inverse-variance weighting inside the suite's own R2 machinery.
The battery's block-level tests pair two independent draws from the within-kind
segment-mean pools (block_mean_pools). Here three weight vectors are applied to
the K drawn positions: unweighted, length-proportional, inverse-pool-variance.
Design SE from the pools' own variances (design quantities, Prop 2).
Size at delta=0 and power at delta=5 AUROC points (bridge scale).
Cells: cmhmil_seed7 vs pca on the five canonical suites."""
import os, sys, json
import numpy as np

sys.path.insert(0, r"D:\0科研\工作1\第15篇SCI\src")
from bootstrap.e3_clean import load_goodness, block_mean_pools  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from common.blocks import build_blocks  # noqa: E402

B = 4000
out = []
for ds in ["SMD", "PSM", "MSL", "SMAP", "WADI"]:
    ga, y = load_goodness("cmhmil_seed7", ds)
    gb, _ = load_goodness("pca", ds)
    T = len(ga)
    blocks = build_blocks(events_from_binary(y.astype(np.int8)), T)
    d = ga - gb
    st = __import__("common.blocks", fromlist=["block_stats"]).block_stats(d, blocks, min_seg=10)
    qk = np.array([b["kind"] for b in st])
    ql = np.array([b["n"] for b in st], float)
    m = np.array([b["m"] for b in st])
    ev = m[qk == "event"]; no = m[qk == "normal"]
    pi = float(y.mean())
    scale = 2 * pi * (1 - pi)
    sig = np.where(qk == "event", ev.std(ddof=1), no.std(ddof=1))
    w_iv = 1.0 / sig ** 2; w_iv /= w_iv.sum()
    w_len = ql / ql.sum()
    w_u = np.ones(len(qk)) / len(qk)
    rng = np.random.default_rng(20260910)
    ie = qk == "event"

    def trial(w, delta_pts):
        delta = delta_pts / 100.0 * scale
        A = np.where(ie, ev[rng.integers(0, len(ev), len(qk))],
                     no[rng.integers(0, len(no), len(qk))])
        Bv = np.where(ie, ev[rng.integers(0, len(ev), len(qk))],
                      no[rng.integers(0, len(no), len(qk))])
        x = (A - Bv) - delta
        se = float(np.sqrt(2.0 * np.sum((w ** 2) * sig ** 2)))  # A-B difference: 2 sigma^2
        return abs((w * x).sum()) / se > 1.96

    row = dict(dataset=ds, K=len(qk))
    for name, w in [("unw", w_u), ("len", w_len), ("iv", w_iv)]:
        row[f"{name}_size"] = round(sum(trial(w, 0.0) for _ in range(B)) / B, 3)
        row[f"{name}_pow5"] = round(sum(trial(w, 5.0) for _ in range(B)) / B, 3)
    out.append(row)
    print(row, flush=True)

json.dump(out, open(r"D:\0科研\工作1\第15篇SCI\02_实验记录\results\r27f_weights.json", "w"), indent=1)
print("saved r27f_weights.json")

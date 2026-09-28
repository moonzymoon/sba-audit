# -*- coding: utf-8 -*-
"""R30-1: power calibration -- realized vs nominal power at injected effects.

For each canonical suite (cmhmil_seed7 vs pca), use the R2 pool machinery:
two independent draws from within-kind block-mean pools; inject a shift delta
(AUROC points via bridge scale) into side A's drawn means; segment-level test
studentized by the design SE sqrt(2 * sum w_j^2 sigma_j^2) with unweighted w.
Realized power = rejection rate over B=2000 trials at each delta on a grid.
Nominal power = two-sided normal: Phi(d/se - 1.96) + Phi(-d/se - 1.96).
Key check: realized power at each suite's own median MDE ~ 0.80.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, r"D:\0keyan\gongzuo1\paper 15SCI\src")
from bootstrap.e3_clean import load_goodness  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from common.blocks import build_blocks, block_stats  # noqa: E402
from scipy import stats as st

B = 2000
out = []
for ds in ["SMD", "PSM", "MSL", "SMAP", "WADI"]:
    ga, y = load_goodness("cmhmil_seed7", ds)
    gb, _ = load_goodness("pca", ds)
    T = len(ga)
    blocks = build_blocks(events_from_binary(y.astype(np.int8)), T)
    d = ga - gb
    bs = block_stats(d, blocks, min_seg=10)
    qk = np.array([b["kind"] for b in bs])
    m = np.array([b["m"] for b in bs])
    ev, no = m[qk == "event"], m[qk == "normal"]
    pi = float(y.mean())
    scale = 2 * pi * (1 - pi)
    K = len(qk)
    sig = np.where(qk == "event", ev.std(ddof=1), no.std(ddof=1))
    se = float(np.sqrt(2 * sig.mean() ** 2 / K))  # unweighted: Var(mean diff)
    mde = 2.8 * se / scale * 100
    rng = np.random.default_rng(20260911)
    ie = qk == "event"
    deltas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]  # in units of MDE
    rows = {}
    for f in deltas:
        delta_pts = f * mde
        delta = delta_pts / 100.0 * scale
        cnt = 0
        for _ in range(B):
            A = np.where(ie, ev[rng.integers(0, len(ev), K)], no[rng.integers(0, len(no), K)]) + delta
            Bv = np.where(ie, ev[rng.integers(0, len(ev), K)], no[rng.integers(0, len(no), K)])
            xbar = (A - Bv).mean()
            if abs(xbar) / se > 1.96:
                cnt += 1
        real = cnt / B
        nom = float(st.norm.cdf(delta / se - 1.96) + st.norm.cdf(-delta / se - 1.96))
        rows[round(f, 2)] = [round(real, 3), round(nom, 3)]
    rec = dict(dataset=ds, K=K, se_pts=round(se / scale * 100, 2), mde_pts=round(mde, 1),
               realized_vs_nominal=rows)
    out.append(rec)
    print(ds, "MDE=%.1f pts | at MDE: realized %.3f vs nominal %.3f" %
          (mde, rows[1.0][0], rows[1.0][1]), flush=True)

json.dump(out, open(r"D:\0keyan\gongzuo1\paper 15SCI\02_shiyanjilu\results\r30_powercal.json", "w"), indent=1)
mx = max(abs(r["realized_vs_nominal"][1.0][0] - r["realized_vs_nominal"][1.0][1]) for r in out)
print("max |realized-nominal| at MDE across suites: %.3f" % mx)
print("saved r30_powercal.json")

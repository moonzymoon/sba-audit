# -*- coding: utf-8 -*-
"""Deep-derivation item 1 reproducibility: block-length ratio-estimator effects.

Computes, per benchmark (pca cell, qualified blocks):
  lbar, E[1/l], harmonic-arithmetic gap lbar*E[1/l]  (over-prediction ceiling)
  CV_l, (1+CV^2)                                      (under-predication factor)
Saved to results/length_effects.csv; numbers quoted in Appendix A (R11).
"""
import csv
import sys

import numpy as np

sys.path.insert(0, "D:/0keyan/gongzuo1/paper 15SCI/src")
sys.path.insert(0, "D:/0keyan/gongzuo1/paper 15SCI/src/bootstrap")
from common.blocks import build_blocks  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from bootstrap.e3_clean import load_goodness  # noqa: E402

R = "D:/0keyan/gongzuo1/paper 15SCI/02_shiyanjilu/results/"
rows = []
for bm in ["WADI", "SMD", "PSM", "MSL", "SMAP"]:
    g, y = load_goodness("pca", bm)
    blocks = build_blocks(events_from_binary(y), len(y))
    ln = np.array([e - s for (k, s, e) in blocks if k == "normal"], float)
    lbar, einv = ln.mean(), (1 / ln).mean()
    cv = ln.std(ddof=1) / lbar
    rows.append(dict(bm=bm, n_normal=len(ln), lbar=lbar, E_inv_l=einv,
                     harm_gap=lbar * einv, CV=cv, one_plus_cv2=1 + cv ** 2,
                     lmin=ln.min(), lmax=ln.max()))
    print(rows[-1])
with open(R + "length_effects.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("saved length_effects.csv")

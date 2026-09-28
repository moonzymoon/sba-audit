# -*- coding: utf-8 -*-
"""R27e v2: benchmark-portfolio guidance on the direct scale (no new conventions).
Atoms: per-suite median per-pair SD sigma_i (AUROC points); each independent
suite contributes precision 1/sigma_i^2. Pooled floor = 2.8 / sqrt(sum 1/sigma^2).
Questions answered: (a) floor after adding n suites of a given class;
(b) how many MSL-class suites to reach the 5-point bar; (c) marginal value by class."""
import json
import numpy as np
import pandas as pd

RES = r"D:\0keyan\gongzuo1\paper 15SCI\02_shiyanjilu\results"
d = pd.read_csv(RES + r"\direct_scale_verdict.csv", encoding="utf-8-sig")
sig = d.groupby("bm")["sd_direct_pts"].median()[["SMD", "PSM", "MSL", "SMAP", "WADI"]]
zz = 2.8
base = float((1.0 / sig.values ** 2).sum())
base_floor = zz / np.sqrt(base)
print("sigma:", {k: round(float(v), 2) for k, v in sig.items()})
print("current 5-suite pooled floor = %.1f pts" % base_floor)

def floor_with(n, sigma):
    return float(zz / np.sqrt(base + n / sigma ** 2))

out = {"sigma_pts": {k: round(float(v), 2) for k, v in sig.items()},
       "base_floor_pts": round(base_floor, 1)}
for name, s in [("MSL-class", sig["MSL"]), ("SMD-class", sig["SMD"]), ("WADI-class", sig["WADI"])]:
    n5 = (zz / 5.0) ** 2 - base
    n5 = int(np.ceil(n5 * s ** 2)) if n5 > 0 else 0
    out[f"add_{name}"] = {"n_to_reach_5pts": n5,
                          "floor_after_1": round(floor_with(1, s), 1),
                          "floor_after_n": round(floor_with(n5, s), 1) if n5 else None}
    print(name, "-> +1 suite:", out[f"add_{name}"]["floor_after_1"],
          "| n for 5 pts:", n5, "| floor then:", out[f"add_{name}"]["floor_after_n"])
json.dump(out, open(RES + r"\r27e_portfolio.json", "w"), indent=1)
print("saved r27e_portfolio.json")

# -*- coding: utf-8 -*-
"""R41: recompute rho bootstrap CI with shared-index data."""
import os, json
import numpy as np
import pandas as pd

RES = r"D:\0keyan\gongzuo1\paper 15SCI\02_shiyanjilu\results"
d = pd.read_csv(os.path.join(RES, "direct_scale_verdict.csv"), encoding="utf-8-sig")
d["rho"] = d["gap_pts"] / d["mde_direct_pts"]
obs_below = int((d["gap_pts"] < d["mde_direct_pts"]).sum())
obs_med = float(d["rho"].median())

rng = np.random.default_rng(20260926)
B = 2000
bms = d["bm"].unique()
cnt, meds = [], []
for _ in range(B):
    parts = [d[d.bm == b].sample(frac=1.0, replace=True, random_state=rng.integers(1 << 30)) for b in bms]
    db = pd.concat(parts)
    cnt.append(int((db["gap_pts"] < db["mde_direct_pts"]).sum()))
    meds.append(float(db["rho"].median()))
cnt, meds = np.array(cnt), np.array(meds)
out = {"below_of_75": obs_below, "below_CI": [int(np.percentile(cnt, 2.5)), int(np.percentile(cnt, 97.5))],
       "median_rho": round(obs_med, 2),
       "rho_CI": [round(float(np.percentile(meds, 2.5)), 2), round(float(np.percentile(meds, 97.5)), 2)]}
print(out)
json.dump(out, open(os.path.join(RES, "r41_rho_ci.json"), "w"), indent=1)
print("saved r41_rho_ci.json")

# -*- coding: utf-8 -*-
"""Plan A Card 3b: qualified-block threshold sensitivity (event segment >= {5,10,20} pts).

K(threshold) recomputed from the cached labels; floors recomputed by the MDE(K)
formula with sigma_d and pi held at the audited (>=10) values, so the table
isolates the K channel. Mirrors e4_mde.py conventions.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, r"D:\0科研\工作1\第15篇SCI\src")
from bootstrap.e3_clean import load_goodness  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from common.blocks import build_blocks  # noqa: E402

BMS = ["SMD", "PSM", "MSL", "SMAP", "WADI", "TE", "MetroPT3", "BATADAL"]
e4 = pd.read_csv(r"D:\0科研\工作1\第15篇SCI\02_实验记录\results\e4_mde_table.csv").set_index("dataset")
OUT = r"D:\0科研\工作1\第15篇SCI\02_实验记录\results\r21b_threshold_sensitivity.csv"

rows = []
for ds in BMS:
    try:
        _, y = load_goodness("cmhmil_seed7", ds)
    except Exception:
        print("skip", ds)
        continue
    T = len(y)
    blocks = build_blocks(events_from_binary(y.astype(np.int8)), T)
    ev = [e - s for kind, s, e in blocks if kind == "event"]
    no = [e - s for kind, s, e in blocks if kind == "normal"]
    pi = float(y.mean())
    sd = float(e4.loc[ds, "sigma_d"]) if ds in e4.index else np.nan
    rec = {"dataset": ds, "pi": round(pi, 4), "sigma_d": sd}
    for th in (5, 10, 20):
        K = sum(1 for x in ev if x >= th) + len(no)
        rec["K_%d" % th] = K
        if not np.isnan(sd) and K > 1:
            mde = (stats.t.ppf(.975, K - 1) + stats.t.ppf(.8, K - 1)) * sd / np.sqrt(K) \
                  / (2 * pi * (1 - pi)) * 100
            rec["MDE_%d" % th] = round(float(mde), 1)
        else:
            rec["MDE_%d" % th] = ""
    rows.append(rec)
    print(rec)

pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print("saved", OUT)

# -*- coding: utf-8 -*-
"""Deep-derivation item 8 reproducibility: cross-benchmark power synthesis.

FE pooled MDE from e4_mde_table.csv (per-suite V_b = (MDE_pts/100/t_mult)^2);
RE floor from e1_full_components.json interaction (tau_pair = sqrt(2*sigma_i));
suites needed for 5-pt certification; item 9 allocation numbers.
Outputs results/cross_bench_synthesis.csv; numbers quoted in sec 5.4 / sec 8 (R14).
"""
import csv
import json

import numpy as np
from scipy import stats as st

R = "D:/0keyan/gongzuo1/paper 15SCI/02_shiyanjilu/results/"
j = json.load(open(R + "e1_full_components.json"))["AUROC"]
si = np.sqrt(j["V_detxds_resid"])           # detector x dataset interaction SD (AUROC)
tau = np.sqrt(2) * si                        # pair-level heterogeneity
z = st.norm.ppf(.975) + st.norm.ppf(.8)
e4 = {"SMD": (14, 70.6), "PSM": (107, 26.6), "MSL": (61, 43.1),
      "SMAP": (97, 49.2), "WADI": (16, 610.1)}
inv = sum(1 / ((m / 100 / (st.t.ppf(.975, K - 1) + st.t.ppf(.8, K - 1))) ** 2)
          for K, m in e4.values())
Vp = 1 / inv
rows = [dict(quantity="sigma_i_pts", value=100 * si),
        dict(quantity="tau_pair_pts", value=100 * tau),
        dict(quantity="FE_pooled_MDE_pts", value=z * np.sqrt(Vp) * 100),
        dict(quantity="RE_floor_5suites_pts", value=z * tau / np.sqrt(5) * 100),
        dict(quantity="suites_for_5pts", value=(z * tau / 0.05) ** 2),
        dict(quantity="pair_singleseed_noise_pts", value=np.sqrt(2) * 100 * np.sqrt(j["V_seed"]))]
for r in rows:
    print(r)
with open(R + "cross_bench_synthesis.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["quantity", "value"])
    w.writeheader()
    w.writerows(rows)
print("saved cross_bench_synthesis.csv")

# -*- coding: utf-8 -*-
"""R26a: cluster bootstrap CI for netted %GRR/NDC using e1_full's OWN loader
and MoM (build_table + mom_two_way), guaranteeing pipeline identity.
Bootstrap: resample 6 datasets and 6 detectors (cluster) of the balanced
sub-table with replacement; V_seed/V_eval held fixed; interaction netted."""
import sys, json
import numpy as np

sys.path.insert(0, r"D:\0keyan\gongzuo1\paper 15SCI\src\mixed")
sys.path.insert(0, r"D:\0keyan\gongzuo1\paper 15SCI\src")
from e1_full import build_table, mom_two_way  # noqa: E402
from bootstrap.e3_clean import cache_file  # noqa: E402
from e1_full import auroc_of  # noqa: E402
import os

RES = r"D:\0keyan\gongzuo1\paper 15SCI\02_shiyanjilu\results"
CORE6 = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI"]
DET6 = ["cmhmil", "iforest", "pca", "gmm", "ocsvm", "lof"]
CMH_SEEDS = {"SMD": 10, "PSM": 10, "SWaT": 10, "MSL": 11, "SMAP": 11, "WADI": 11}
SEEDLIST = [0, 1, 2, 3, 31, 42, 97, 123, 2024, 4, 5, 6, 8, 9, 10]

df = build_table()
comp = json.load(open(RES + r"\e1_direct_components.json", encoding="utf-8"))["AUROC"]
V_seed, V_eval = comp["V_seed"], comp["V_eval_direct"]

# rebuild the balanced matrix exactly as mom_two_way does (cmhmil -> seed mean)
d2 = df.copy()
cmh = df[df.detector == "cmhmil"]
d2.loc[d2.detector == "cmhmil", "AUROC"] = (
    cmh.groupby("dataset")["AUROC"].transform("mean"))
d2 = d2.drop_duplicates(["dataset", "detector"])
sub = d2[d2.dataset.isin(CORE6) & d2.detector.isin(DET6)]
piv = sub.pivot_table(index="dataset", columns="detector", values="AUROC")
piv = piv.loc[CORE6, DET6]
assert not piv.isna().any().any()
X = piv.values

def grr_ndc(Xm):
    a, b = Xm.shape
    gm = Xm.mean()
    ms_d = b * ((Xm.mean(1) - gm) ** 2).sum() / (a - 1)
    ms_v = a * ((Xm.mean(0) - gm) ** 2).sum() / (b - 1)
    resid = Xm - Xm.mean(1, keepdims=True) - Xm.mean(0, keepdims=True) + gm
    ms_e = (resid ** 2).sum() / ((a - 1) * (b - 1))
    V_ds, V_det = max(0.0, (ms_d - ms_e) / b), max(0.0, (ms_v - ms_e) / a)
    net = max(ms_e - V_seed - V_eval, 0.0)
    V_grr = V_seed + V_eval + net
    V_tot = V_ds + V_det + V_seed + V_eval + net
    return 100 * np.sqrt(V_grr / V_tot), 1.41 * np.sqrt(V_det / V_grr), net

g0, n0, _ = grr_ndc(X)
print("point: GRR=%.1f NDC=%.2f (expect 49.9/0.51)" % (g0, n0))
rng = np.random.default_rng(20260910)
B = 2000
gs, ns, neg = [], [], 0
for _ in range(B):
    ri = rng.integers(0, 6, 6)
    ci = rng.integers(0, 6, 6)
    g, n, net = grr_ndc(X[np.ix_(ri, ci)])
    gs.append(g); ns.append(n); neg += int(net == 0)
gs, ns = np.array(gs), np.array(ns)
out = {"point_GRR": round(g0, 1), "point_NDC": round(n0, 2),
       "GRR_CI95": [round(float(np.percentile(gs, 2.5)), 1), round(float(np.percentile(gs, 97.5)), 1)],
       "NDC_CI95": [round(float(np.percentile(ns, 2.5)), 2), round(float(np.percentile(ns, 97.5)), 2)],
       "truncated_net_freq": round(neg / B, 3), "B": B, "V_seed": V_seed, "V_eval": V_eval}
print(out)
prev = json.load(open(RES + r"\r26_bootstrap_ci.json"))
prev["grr"] = out
json.dump(prev, open(RES + r"\r26_bootstrap_ci.json", "w"), indent=1)
print("updated r26_bootstrap_ci.json (A replaced with pipeline-identical version)")

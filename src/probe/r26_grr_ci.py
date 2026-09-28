# -*- coding: utf-8 -*-
"""R26: (A) cluster bootstrap CI for netted %GRR / NDC -- resample detectors and
datasets of the balanced 6x6 table (MoM per replicate, V_seed/V_eval held fixed
at their separately-identified values, interaction netted as in R20);
(B) benchmark-stratified pair bootstrap CI for the 73/75 uncertifiable count
and the median resolvability ratio rho.
Mirrors e1_full.mom_two_way formulas exactly."""
import glob, os, sys, json, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"D:\0keyan\gongzuo1\paper 15SCI\src")
sys.path.insert(0, r"D:\0keyan\gongzuo1\paper 6SCI\src")
from cadms.scorers import get_scorer  # noqa: E402

CORE6 = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI"]
DET6 = ["cmhmil", "iforest", "pca", "gmm", "ocsvm", "lof"]
CACHE = r"D:\0keyan\gongzuo1\paper 6SCI\src\_score_cache"
RES = r"D:\0keyan\gongzuo1\paper 15SCI\02_shiyanjilu\results"

def cell_auc(path):
    d = np.load(path)
    s, y = d["scores"], d["labels"]
    return float(roc_auc_score(y, np.abs(s)))

def seedmean_auc(ds, seeds):
    vals = [cell_auc(os.path.join(CACHE, f"cmhmil_{ds}_seed{s}.npz")) for s in seeds
            if os.path.exists(os.path.join(CACHE, f"cmhmil_{ds}_seed{s}.npz"))]
    return float(np.mean(vals)), float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0

SEEDMAP = {"SMD": 10, "PSM": 10, "SWaT": 10, "MSL": 11, "SMAP": 11, "WADI": 11}
SEEDLIST = [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024, 4, 5, 6, 8, 9, 10]

X = np.zeros((6, 6))
vs_cells = []
for i, ds in enumerate(CORE6):
    n = SEEDMAP[ds]
    m, v = seedmean_auc(ds, SEEDLIST[:n])
    vs_cells.append(v)
    X[i, 0] = m
    for j, det in enumerate(DET6[1:], start=1):
        X[i, j] = cell_auc(os.path.join(CACHE, f"{det}_{ds}.npz"))
V_seed = float(np.mean(vs_cells))
comp = json.load(open(os.path.join(RES, "e1_direct_components.json")))["AUROC"]
V_eval = comp["V_eval_direct"]
print("rebuilt 6x6; V_seed=%.5f V_eval=%.5f" % (V_seed, V_eval))

def mom(Xm):
    a, b = Xm.shape
    gm = Xm.mean()
    ms_d = b * ((Xm.mean(1) - gm) ** 2).sum() / (a - 1)
    ms_v = a * ((Xm.mean(0) - gm) ** 2).sum() / (b - 1)
    resid = Xm - Xm.mean(1, keepdims=True) - Xm.mean(0, keepdims=True) + gm
    ms_e = (resid ** 2).sum() / ((a - 1) * (b - 1))
    return max(0.0, (ms_d - ms_e) / b), max(0.0, (ms_v - ms_e) / a), ms_e

def grr_ndc(Xm):
    V_ds, V_det, ms_e = mom(Xm)
    net = max(ms_e - V_seed - V_eval, 0.0)
    V_grr = V_seed + V_eval + net
    V_tot = V_ds + V_det + V_seed + V_eval + net
    return 100 * np.sqrt(V_grr / V_tot), 1.41 * np.sqrt(V_det / V_grr), net

g0, n0, _ = grr_ndc(X)
print("point check: GRR=%.1f NDC=%.2f (expect ~49.9/0.51)" % (g0, n0))

rng = np.random.default_rng(20260910)
B = 2000
gs, ns, neg = [], [], 0
for _ in range(B):
    ri = rng.integers(0, 6, 6)
    ci = rng.integers(0, 6, 6)
    g, n, net = grr_ndc(X[np.ix_(ri, ci)])
    gs.append(g); ns.append(n)
    neg += int(net == 0)
gs, ns = np.array(gs), np.array(ns)
out = {"point_GRR": round(g0, 1), "point_NDC": round(n0, 2),
       "GRR_CI": [round(float(np.percentile(gs, 2.5)), 1), round(float(np.percentile(gs, 97.5)), 1)],
       "NDC_CI": [round(float(np.percentile(ns, 2.5)), 2), round(float(np.percentile(ns, 97.5)), 2)],
       "net_negative_freq": round(neg / B, 3), "B": B}
print("A:", out)

# ---- B: 73/75 and median rho, benchmark-stratified pair bootstrap ----
d = pd.read_csv(os.path.join(RES, "direct_scale_verdict.csv"), encoding="utf-8-sig")
d["rho"] = d["gap_pts"] / d["mde_direct_pts"]
obs_below = int((d["gap_pts"] < d["mde_direct_pts"]).sum())
obs_med = float(d["rho"].median())
bms = d["bm"].unique()
cnt, meds = [], []
for _ in range(B):
    parts = [d[d.bm == b].sample(frac=1.0, replace=True, random_state=rng.integers(1 << 30)) for b in bms]
    db = pd.concat(parts)
    cnt.append(int((db["gap_pts"] < db["mde_direct_pts"]).sum()))
    meds.append(float(db["rho"].median()))
cnt, meds = np.array(cnt), np.array(meds)
outB = {"below_of_75": obs_below, "below_CI": [int(np.percentile(cnt, 2.5)), int(np.percentile(cnt, 97.5))],
        "median_rho": round(obs_med, 2),
        "rho_CI": [round(float(np.percentile(meds, 2.5)), 2), round(float(np.percentile(meds, 97.5)), 2)]}
print("B:", outB)
json.dump({"grr": out, "rho": outB}, open(os.path.join(RES, "r26_bootstrap_ci.json"), "w"), indent=1)
print("saved r26_bootstrap_ci.json")

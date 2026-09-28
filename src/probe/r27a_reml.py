# -*- coding: utf-8 -*-
"""R27a: REML on the partially-replicated panel (run-level AUROCs).

Model: y_ijk = mu + a_ds + b_det + (ab)_cell + eps_run, all random.
Panel: cmhmil 6 ds x 10-11 seeds (63), at2 2 ds x 5 seeds (10),
       shallow 5 det x 6 ds single runs (30).  n = 103, 38 cells.
Direct Gaussian REML on the n x n covariance (4 variance components, >=0).
Deliverable: does the interaction hit the zero boundary (confirming the R20
netted MoM reading) and does run variance match Table 6's seed component?
"""
import os, json
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

CACHE = r"D:\0keyan\gongzuo1\paper 6SCI\src\_score_cache"
RES = r"D:\0keyan\gongzuo1\paper 15SCI\02_shiyanjilu\results"
CORE6 = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI"]
CMH_SEEDS = {"SMD": [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024],
             "PSM": [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024],
             "SWaT": [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024],
             "MSL": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "SMAP": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "WADI": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
AT_SEEDS = [7, 42, 123, 202, 999]
SHALLOW = ["iforest", "pca", "gmm", "ocsvm", "lof"]

def auc(f):
    d = np.load(f)
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(d["labels"], d["scores"]))

obs = []  # (ds, det, cell_id, value)
for ds in CORE6:
    for s in CMH_SEEDS[ds]:
        f = os.path.join(CACHE, f"cmhmil_{ds}_seed{s}.npz")
        if os.path.exists(f):
            obs.append((ds, "cmhmil", f"cmhmil|{ds}", auc(f)))
for ds in ["SMD", "PSM"]:
    for s in AT_SEEDS:
        f = os.path.join(CACHE, f"at2_{ds}_seed{s}.npz")
        if os.path.exists(f):
            obs.append((ds, "at", f"at|{ds}", auc(f)))
for det in SHALLOW:
    for ds in CORE6:
        f = os.path.join(CACHE, f"{det}_{ds}.npz")
        if os.path.exists(f):
            obs.append((ds, det, f"{det}|{ds}", auc(f)))

n = len(obs)
y = np.array([o[3] for o in obs])
ds_idx = {d: i for i, d in enumerate(sorted(set(o[0] for o in obs)))}
det_idx = {d: i for i, d in enumerate(sorted(set(o[1] for o in obs)))}
cell_ids = sorted(set(o[2] for o in obs))
cell_idx = {c: i for i, c in enumerate(cell_ids)}
I = np.array([ds_idx[o[0]] for o in obs])
J = np.array([det_idx[o[1]] for o in obs])
C = np.array([cell_idx[o[2]] for o in obs])
print("n=%d cells=%d datasets=%d detectors=%d" % (n, len(cell_ids), len(ds_idx), len(det_idx)))

def reml_nll(theta):
    sds, sdt, sint, srun = np.exp(theta)
    V = (sds * (I[:, None] == I[None, :]) + sdt * (J[:, None] == J[None, :])
         + sint * (C[:, None] == C[None, :]) + srun * np.eye(n))
    c = cho_factor(V + 1e-10 * np.eye(n), lower=True)
    Vi = cho_solve(c, np.eye(n))
    one = np.ones(n)
    Vi1 = cho_solve(c, one)
    logdet = 2 * np.sum(np.log(np.diag(c[0])))
    yvy = y @ cho_solve(c, y)
    return 0.5 * (logdet + np.log(one @ Vi1) + yvy - (y @ Vi1) ** 2 / (one @ Vi1))

best = None
for x0 in ([np.log(v) for v in (0.03, 0.001, 0.001, 0.006)],
           [np.log(v) for v in (0.02, 0.005, 0.003, 0.004)],
           [np.log(v) for v in (0.01, 0.0005, 0.0001, 0.01)]):
    r = minimize(reml_nll, x0, method="L-BFGS-B", bounds=[(-20, 0)] * 4)
    if best is None or r.fun < best.fun:
        best = r
th = np.exp(best.x)
lab = ["V_ds", "V_det", "V_int", "V_run"]
est = dict(zip(lab, [round(float(v), 5) for v in th]))
print("REML:", est, "| converged:", best.success)

out = {"n": n, "cells": len(cell_ids), "reml": est,
       "note": "V_int at/below ~1e-5 => boundary (confirms netted reading); V_run vs Table6 V_seed=0.00586; V_ds vs 0.02819; V_det vs 0.00128",
       "table6_reference": {"V_ds": 0.02819, "V_det": 0.00128, "V_seed(run)": 0.00586},
       "converged": bool(best.success)}
json.dump(out, open(os.path.join(RES, "r27a_reml.json"), "w"), indent=1)
print("saved r27a_reml.json")

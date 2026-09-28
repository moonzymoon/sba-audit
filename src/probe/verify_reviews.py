# -*- coding: utf-8 -*-
"""外部审查核验脚本: 逐条验证三个AI提出的定量主张."""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "D:/0科研/工作1/第15篇SCI/src")
sys.path.insert(0, "D:/0科研/工作1/第15篇SCI/src/bootstrap")

R = "D:/0科研/工作1/第15篇SCI/02_实验记录/results/"

print("=== 1) bridge (1-pi) 数值验证 ===")
rng = np.random.default_rng(7)
try:
    from sklearn.metrics import roc_auc_score
    for pi in [0.0946, 0.2214, 0.01]:
        N = 20000
        y = (rng.random(N) < pi).astype(int)
        s = rng.random(N) * (1 + 0.5 * y)
        g = stats.rankdata(s) / N * (2 * y - 1)
        Au = roc_auc_score(y, s)
        pred = 2 * pi * (1 - pi) * Au + pi**2 - 0.5 + (pi - 0.5) / N
        print("  pi=%.4f gbar=%.5f pred=%.5f diff=%.2e" % (pi, g.mean(), pred, abs(g.mean() - pred)))
except Exception as e:
    print("  skip:", e)

print("\n=== 2) e4 桥接修正 ===")
e4 = pd.read_csv(R + "e4_mde_table.csv")
e4["MDE_new"] = e4["MDE_AUROC_pts"] / (1 - e4["anom_rate"])
f_old = stats.t.ppf(0.975, e4.K_blocks - 1) + stats.t.ppf(0.8, e4.K_blocks - 1)
e4["K_req_new"] = (f_old * e4.MDE_new / 5.0) ** 2
print(e4[["dataset", "anom_rate", "K_blocks", "MDE_AUROC_pts", "MDE_new", "K_needed_for_5AUROCpts", "K_req_new"]].round(1).to_string())

print("\n=== 3) sigma_d 定义核查 ===")
f14 = stats.t.ppf(0.975, 13) + stats.t.ppf(0.8, 13)
print("  f(13)=%.4f  f*sd/sqrt14=%.5f (MDE_dbar=0.12095)" % (f14, f14 * 0.14934 / 14**0.5))

print("\n=== 4) TSB170/gmm CI ===")
e3 = pd.read_csv(R + "e3_clean_summary.csv")
r17 = e3[(e3.dataset == "TSB170") & (e3.scorer == "gmm")]
print(r17[["size_blk_t", "ci_blk_t", "size_win_t"]].to_string())

print("\n=== 5) e1 成分 ===")
j = json.load(open(R + "e1_full_components.json", encoding="utf-8"))
print(json.dumps(j, indent=1, ensure_ascii=False)[:1000])

print("\n=== 6) tau SMD ===")
tau = pd.read_csv(R + "e2_tau_map.csv")
print(tau[(tau.dataset == "SMD")].round(3).to_string())

print("\n=== 7) r_win 双抽取窗口列 canonical ===")
canon = e3[e3.dataset.isin(["SMD", "PSM", "MSL", "SMAP", "WADI"])]
print("  r_win_t: %.3f-%.3f  r_blk_t: %.3f-%.3f  win_t: %.3f-%.3f" % (
    canon.size_r_win_t.min(), canon.size_r_win_t.max(),
    canon.size_r_blk_t.min(), canon.size_r_blk_t.max(),
    canon.size_win_t.min(), canon.size_win_t.max()))

print("\n=== 8) AT 行 ===")
at = e3[e3.scorer.str.contains("AT|at2", case=False, na=False)]
print(at[["dataset", "scorer", "size_win_t", "size_win_wilcox", "size_blk_t", "size_blk_wilcox"]].round(3).to_string())

print("\n=== 9) NEweather CI ===")
nw = e3[(e3.dataset == "NEweather")]
print(nw[["scorer", "size_win_t", "ci_win_t"]].to_string())

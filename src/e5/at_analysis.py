"""E5b: 第二深度族 (Anomaly Transformer) 种子方差分析.

问题 (limits iv): R=V_seed/V_clean≪1 是否限于 CMH-MIL 单族?
方法: 5 种子 × {SMD, PSM} 的 at2_* 缓存, 10 对/数据集;
  dbar_k = 配对差均值 (秩尺度, 与 e5 同构); V_run = Var_k(dbar);
  R = max(0, V_run − V_clean) / V_clean  (V_clean 来自 at2_seed7 的 E3-clean 格).

用法: python e5/at_analysis.py
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = r"D:/0科研/工作1/第6篇SCI/src/_score_cache"
RCACHE = r"D:/0科研/工作1/第15篇SCI/_resample_cache"
RES = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results"
SEEDS = [7, 42, 123, 202, 999]


def auroc(s, y):
    r = rankdata(s)
    n1, n0 = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def g_of(s, y):
    return (rankdata(s) / len(s)) * (2.0 * y - 1.0)


def main():
    rows = []
    for ds in ["SMD", "PSM"]:
        gs, au = {}, {}
        for sd in SEEDS:
            d = np.load(os.path.join(CACHE, f"at2_{ds}_seed{sd}.npz"))
            gs[sd] = g_of(d["scores"].astype(np.float64), d["labels"].astype(np.int8))
            au[sd] = auroc(d["scores"].astype(np.float64), d["labels"].astype(np.int8))
        dbars = [np.mean(gs[a] - gs[b]) for a, b in itertools.combinations(SEEDS, 2)]
        v_run = float(np.var(dbars, ddof=1))
        f = os.path.join(RCACHE, f"e3clean_{ds}_at2_seed7.npz")
        if os.path.exists(f):
            v_clean = float(np.var(np.load(f)["dbar"], ddof=1))
        else:
            v_clean = np.nan
            print(f"[warn] {ds}: V_clean 缓存未找到, R 记 nan")
        r = max(0.0, v_run - v_clean) / v_clean if np.isfinite(v_clean) else np.nan
        rows.append(dict(dataset=ds, n_seeds=len(SEEDS), n_pairs=len(dbars),
                         auroc_min=min(au.values()), auroc_max=max(au.values()),
                         auroc_spread_pts=100 * (max(au.values()) - min(au.values())),
                         V_run=v_run, V_clean=v_clean, R=r))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e5b_at_seed_panel.csv"), index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

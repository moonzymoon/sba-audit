"""R 比值 (预注册 §7.3): R = V_seed-pair / V_clean.

V_clean:  N2 (s, s*) 的 10000 次 d̄* 经验方差 (E3-clean 落盘缓存);
V_seed-pair: 同数据集内真实种子对 d̄ 的经验方差 (3 对时为描述性初值, 预注册明示).
测量点 = 块级配对差均值的方差 (秩尺度, 修订 R1).
"""
from __future__ import annotations

import os

import numpy as np

RCACHE = r"D:/0科研/工作1/第15篇SCI/_resample_cache"


def compute_R(dbar_clean, dbar_seeds):
    """R = Var(d̄_seed-pair) / Var(d̄*_clean). 输入为两组均值统计量数组."""
    dbar_clean = np.asarray(dbar_clean, float)
    dbar_seeds = np.asarray(dbar_seeds, float)
    if len(dbar_seeds) < 2 or len(dbar_clean) < 2:
        return float("nan"), float("nan"), float("nan")
    v_seed = float(np.var(dbar_seeds, ddof=1))
    v_clean = float(np.var(dbar_clean, ddof=1))
    r = v_seed / v_clean if v_clean > 0 else float("inf")
    return r, v_seed, v_clean


def load_clean_dbar(dataset, scorer="cmhmil_seed7"):
    f = os.path.join(RCACHE, f"e3clean_{dataset}_{scorer}.npz")
    if not os.path.exists(f):
        return None
    return np.load(f)["dbar"]


def r_report(pairs_dbar, dataset, scorer="cmhmil_seed7"):
    dbar_clean = load_clean_dbar(dataset, scorer)
    if dbar_clean is None:
        return dict(dataset=dataset, R=np.nan, V_seed=np.nan, V_clean=np.nan,
                    note="E3-clean 缓存缺失")
    r, vs, vc = compute_R(dbar_clean, pairs_dbar)
    return dict(dataset=dataset, R=r, V_seed=vs, V_clean=vc, n_pairs=len(pairs_dbar),
                note="3 对为描述性初值" if len(pairs_dbar) == 3 else "")

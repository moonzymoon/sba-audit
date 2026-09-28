"""E2 τ 图谱 (R3 扩充): 全数据集 × stride {1,4,16} 的事件段级 τ + 全序列 τ.

配对: cmhmil_seed7 vs pca (每数据集都有缓存; AT/iforest 对已有阶段0探针).
输出 02_实验记录/results/e2_tau_map.csv —— 论文核心图的数值底座.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from probe.tau_probe import load_pair, zscore, ips_tau, cluster_ips_tau  # noqa: E402

OUT = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results/e2_tau_map.csv"
DATASETS = ["SMD", "PSM", "MSL", "SMAP", "WADI", "TE", "MetroPT3", "BATADAL", "NEweather"]
PAIRS = [("cmhmil_SMD_seed7", "iforest_SMD", "SMD")]  # 阶段0 已有, 跳过重算标记


def main():
    rows = []
    for ds in DATASETS:
        n1 = f"cmhmil_{ds}_seed7"
        n2 = f"pca_{ds}"
        f1 = os.path.join(r"D:/0科研/工作1/第6篇SCI/src/_score_cache", f"{n1}.npz")
        f2 = os.path.join(r"D:/0科研/工作1/第6篇SCI/src/_score_cache", f"{n2}.npz")
        if not (os.path.exists(f1) and os.path.exists(f2)):
            print(f"[skip] {ds}: 缓存不全")
            continue
        s1, s2, y = load_pair(n1, n2)
        from common.events import events_from_binary
        events = events_from_binary(y)
        D = zscore(s1) - zscore(s2)
        T = len(y)
        for stride in (1, 4, 16):
            tau_full = ips_tau(D[::stride])
            tau_seg, Nseg = cluster_ips_tau(D, events, T, stride)
            rows.append(dict(dataset=ds, pair="cmhmil-pca", stride=stride,
                             tau_win_full=round(float(tau_full), 2),
                             tau_seg=round(float(tau_seg), 3) if np.isfinite(tau_seg) else np.nan,
                             N_seg_windows=Nseg, n_events=len(events)))
        print(f"{ds}: τ_seg[1/4/16] = {[r['tau_seg'] for r in rows[-3:]]}")
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print("已写出", OUT)


if __name__ == "__main__":
    main()

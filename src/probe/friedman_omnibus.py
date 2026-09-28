# -*- coding: utf-8 -*-
"""Friedman 全向检验 (R5 增补分析): 6 全覆盖检测器 x 5 典型基准,
合格块(事件段>=10 + 正常间隔)的秩裕量块均值上做 friedmanchisquare.
复用 e3_clean 的缓存加载与块机制, 无重采样, 确定性计算."""
import os
import sys

import numpy as np
from scipy.stats import friedmanchisquare

sys.path.insert(0, r"D:/0科研/工作1/第15篇SCI/src")
sys.path.insert(0, r"D:/0科研/工作1/第15篇SCI/src/bootstrap")
from common.blocks import build_blocks, block_stats  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from e3_clean import load_goodness  # noqa: E402

BMS = ["SMD", "PSM", "MSL", "SMAP", "WADI"]
SC = ["cmhmil_seed7", "iforest", "pca", "gmm", "ocsvm", "lof"]

if __name__ == "__main__":
    rows = []
    for bm in BMS:
        y = None
        cols = []
        for sc in SC:
            g, y_ = load_goodness(sc, bm)
            y = y_
            cols.append(g)
        events = events_from_binary(y)
        blocks = build_blocks(events, len(y))
        mats = []
        for g in cols:
            st = block_stats(g, blocks, min_seg=10)
            mats.append(np.array([b["m"] for b in st]))
        K = mats[0].shape[0]
        assert all(m.shape == (K,) for m in mats), "block count mismatch"
        chi2, p = friedmanchisquare(*mats)
        rows.append((bm, K, float(chi2), float(p)))
        print("%-6s K=%3d  chi2(5)=%8.2f  p=%.4f" % (bm, K, chi2, p))
    out = os.path.join(r"D:/0科研/工作1/第15篇SCI/02_实验记录/results", "friedman_omnibus.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("dataset,K,chi2_df5,p\n")
        for bm, K, c, p in rows:
            f.write(f"{bm},{K},{c:.4f},{p:.5f}\n")
    print("saved:", out)

"""E3-realistic (pre-registered §7.2): samedetectornotsame seed  realpaired — zh:zh_7063assumption.

cmhmil ∈ {(7,42),(7,123),(42,123)} × {SMD,PSM,SWaT}; zh:zh_4307auditzh:zh_1063same E3-clean ①–④.
SWaT segmentzh:zh_4092 (N_seg=1) -> zh:zh_7674windowzh:zh_4542 (①②), blockzh:zh_9363notzh:zh_9412.
samezh:zh_2567 R zh:zh_8961valuezh_2440  d̄ (seed pair) — with E3-clean   d̄* cachepairedzh:use.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.blocks import build_blocks, block_stats  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from bootstrap.e3_clean import load_goodness  # noqa: E402
from bootstrap.r_ratio import r_report  # noqa: E402


def real_block_pvalues(d, blocks, min_seg=10):
    """realistic zh:zh_5429blockzh:zh_4542: realzh:seedpairzh:zh_3757samezh:zh_6443windowzh:zh_6814paired, d  zh:zh_7452blockmeanzh:i.e.zh_6767
    (no zh:zh_3847, zh:zh_6924 R2 zh:zh_6650and clean zh:zh_5429  original-vs-reshuffled paired)."""
    m = np.array([b["m"] for b in block_stats(d, blocks, min_seg=min_seg)])
    if len(m) < 3:
        return np.nan, np.nan
    p3 = float(stats.ttest_1samp(m, 0.0).pvalue)
    try:
        p4 = float(stats.wilcoxon(m).pvalue) if len(m) >= 6 else np.nan
    except ValueError:
        p4 = 1.0
    return p3, p4

RCACHE = r"D:/0科研/工作1/第15篇SCI/_resample_cache"
OUT = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results/e3_realistic.csv"
SEEDS = [7, 42, 123]
DATASETS = ["SMD", "PSM", "SWaT"]


def main():
    rows, rrows = [], []
    for ds in DATASETS:
        gs = {}
        for s in SEEDS:
            g, y = load_goodness(f"cmhmil_seed{s}", ds)
            gs[s] = g
        T = len(gs[7])
        blocks = build_blocks(events_from_binary(y), T)
        dbar_by_ds = []
        for i, a in enumerate(SEEDS):
            for b in SEEDS[i + 1:]:
                d = gs[a] - gs[b]
                p1 = float(stats.ttest_1samp(d, 0.0).pvalue)
                try:
                    p2 = float(stats.wilcoxon(d).pvalue)
                except ValueError:
                    p2 = 1.0
                p3, p4 = real_block_pvalues(d, blocks)
                if ds == "SWaT":  # segmentzh:zh_4092: blockzh:zh_4542notzh:zh_9412
                    p3 = p4 = np.nan
                dbar = float(d.mean())
                dbar_by_ds.append(dbar)
                rows.append(dict(dataset=ds, pair=f"{a}-{b}", T=T,
                                 dbar=dbar, p_win_t=p1, p_win_wilcox=p2,
                                 p_blk_t=p3, p_blk_wilcox=p4,
                                 AUROC_a=np.nan, AUROC_b=np.nan))
        rep = r_report(dbar_by_ds, ds)
        rep["dbar_pairs"] = str([round(v, 5) for v in dbar_by_ds])
        rrows.append(rep)
        print(f"{ds}: d̄(zh:seedpair)={[round(v,5) for v in dbar_by_ds]}, R={rep['R']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    pd.DataFrame(rrows).to_csv(OUT.replace(".csv", "_R.csv"), index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    print("\nempiricalzh:false positiverate (α=0.05, 9 cell):")
    for c in ["p_win_t", "p_win_wilcox", "p_blk_t", "p_blk_wilcox"]:
        v = df[c].dropna()
        print(f"  {c}: {(v < 0.05).mean():.3f} ({(v < 0.05).sum()}/{len(v)})")


if __name__ == "__main__":
    main()

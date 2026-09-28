"""aggregation _resample_cache   e3clean_*.npz -> E3-clean main table CSV (Type I + 95%Wilson CI)."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

RCACHE = r"D:/0科研/工作1/第15篇SCI/_resample_cache"
OUT = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results/e3_clean_summary.csv"
ALPHA = 0.05


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(RCACHE, "e3clean_*.npz"))):
        d = np.load(f)
        tag = os.path.basename(f)[len("e3clean_"):-len(".npz")]
        ds, scorer = tag.split("_", 1)  # dataset name without underscore: SMD/cmhmil_seed7
        row = dict(dataset=ds, scorer=scorer, B=int(d["B"]), T=int(d["T"]),
                   n_blocks=int(d["n_blocks"]),
                   V_clean=float(np.var(d["dbar"], ddof=1)))
        for k in ["win_t", "win_wilcox", "blk_t", "blk_wilcox", "n1_signperm",
                  "r_win_t", "r_win_wilcox", "r_blk_t", "r_blk_wilcox"]:
            v = d[k]
            v = v[~np.isnan(v)]
            n = len(v)
            kk = int((v < ALPHA).sum())
            lo, hi = wilson_ci(kk, n)
            row[f"size_{k}"] = kk / n if n else np.nan
            row[f"ci_{k}"] = f"[{lo:.3f},{hi:.3f}]" if n else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    cols = ["dataset", "scorer", "B", "n_blocks", "size_win_t", "size_win_wilcox",
            "size_blk_t", "size_blk_wilcox", "size_n1_signperm", "V_clean"]
    print(df[cols].to_string(index=False))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()

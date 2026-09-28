"""E5b: paper zh:zh_2315 (Anomaly Transformer) zh:seedzh_6254differencezh:zh_7069.

zh:zh_5535 (limits iv): R=V_seed/V_clean≪1 zh:zh_1737 CMH-MIL zh:zh_6452?
zh:method: 5 zh:seed × {SMD, PSM}   at2_* cache, 10 pair/dataset;
  dbar_k = paireddifferencemean (rankzh:zh_405, with e5 samezh:zh_2058); V_run = Var_k(dbar);
  R = max(0, V_run − V_clean) / V_clean  (V_clean zh:zh_7902 at2_seed7   E3-clean cell).

zh:zh_966: python e5/at_analysis.py
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = r"D:/0keyan/gongzuo1/paper 6SCI/src/_score_cache"
RCACHE = r"D:/0keyan/gongzuo1/paper 15SCI/_resample_cache"
RES = r"D:/0keyan/gongzuo1/paper 15SCI/02_shiyanjilu/results"
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
            print(f"[warn] {ds}: V_clean cachezh:zh_9954, R zh:zh_1152 nan")
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

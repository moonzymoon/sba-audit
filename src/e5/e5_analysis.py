"""E5 zh:zh_7069: zh:zh_9092seedzh_4219valuezh_8488 + zh:zh_8228 (pre-registered §12 + R3).

zh:data: cmhmil × {SMD,PSM,SWaT}(10zh:seed) ∪ {MSL,SMAP,WADI}(11zh:seed) = 63  zh:zh_6802 run.
zh:zh_7185:
1) zh:zh_8228: each datasetallzh:seedpair (45/55 pair)   d̄ withwindowzh:zh_4542/blockzh:zh_4542 p zh:value -> empiricalzh:false positiverate;
2) V_seed zh:zh_6898: run zh:zh_4542meanzh:zh_6254difference − V_clean(bootstrap zh:zh_8441) = σ²_seed (zh:zh_64 0);
3) R zh:zh_7221value: R = 2(σ²_seed + V_clean)  zh:zh_6331pairzh:zh_9388 —— zh:zh_6523: zh:seedrandomzh_746 A/B zh:zh_418,
   σ²_seed from  A zh:zh_8611, zh:zh_5717differencezh:zh_6701 A×B zh:zh_2568 25 pair d̄ (zh:zh_5820), log zh:zh_6254differencezh:zh_8961 95%CI + TOST[0.67,1.5];
4) SWaT/MSL/SMAP/WADI blockzh:zh_4542perzh:zh_9925segmentzh:zh_7202 (SWaT onlywindowzh:zh_4542).
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.blocks import build_blocks, block_stats  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from bootstrap.e3_clean import load_goodness, block_mean_pools  # noqa: E402
from bootstrap.r_ratio import load_clean_dbar  # noqa: E402

OUT = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results"
GLOBAL_SEED = 20260902
SEEDS = {"SMD": [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024],
         "PSM": [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024],
         "SWaT": [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024],
         "MSL": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         "SMAP": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         "WADI": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}


def tost(log_ratios, lo=0.67, hi=1.5, alpha=0.05):
    """TOST zh:etc.zh_3812test on zh:zh_6254differencezh:zh_8961 (zh:zh_6507is zh:zh_8961valuearray)."""
    r = np.asarray(log_ratios, float)
    r = r[np.isfinite(r) & (r > 0)]
    if len(r) < 3:
        return dict(n=len(r), p=np.nan, equiv=np.nan)
    # TOST: two-zh:zh_333 t zh:test, mean pairzh:zh_8362
    lr = np.log(r)
    se = lr.std(ddof=1) / np.sqrt(len(lr))
    if se == 0:
        return dict(n=len(r), p=np.nan, equiv=np.nan)
    t1 = (np.log(hi) - lr.mean()) / se
    t2 = (lr.mean() - np.log(lo)) / se
    p = max(stats.t.sf(t1, len(lr) - 1), stats.t.sf(t2, len(lr) - 1))
    return dict(n=len(r), p=float(p), equiv=bool(p < alpha),
                ci95=[float(np.exp(lr.mean() - 1.96 * se)), float(np.exp(lr.mean() + 1.96 * se))])


def main():
    rows, calib = [], []
    for ds, seeds in SEEDS.items():
        gs = {}
        for s in seeds:
            g, y = load_goodness(f"cmhmil_seed{s}", ds)
            gs[s] = g
        T = len(g)
        blocks = build_blocks(events_from_binary(y), T)
        # run zh:zh_4542windowmean m̄_r
        mbar = {s: float(gs[s].mean()) for s in seeds}
        pairs = list(itertools.combinations(seeds, 2))
        pw_rej = bw_rej = 0
        nvalid_b = 0
        dbars = []
        for a, b in pairs:
            d = gs[a] - gs[b]
            dbars.append(float(d.mean()))
            p1 = float(stats.ttest_1samp(d, 0.0).pvalue)
            pw_rej += p1 < 0.05
            if ds != "SWaT":
                m = np.array([x["m"] for x in block_stats(d, blocks, min_seg=10)])
                if len(m) >= 3:
                    p3 = float(stats.ttest_1samp(m, 0.0).pvalue)
                    bw_rej += p3 < 0.05
                    nvalid_b += 1
        dbars = np.array(dbars)
        # V_clean (E3-clean cache; no cache zh:zh_6434datasetzh:zh_8215blockmeanzh:zh_7421;
        # SWaT K_event=1 zh:zh_3558 -> notzh:zh_8450, zh:zh_1152 NaN)
        dbar_clean = load_clean_dbar(ds)
        n_ev = len([1 for k, _, _ in blocks if k == "event"])
        if dbar_clean is not None:
            v_clean = float(np.var(dbar_clean, ddof=1))
            v_clean_src = "e3clean-cache"
        elif n_ev >= 2:
            ev, no = block_mean_pools(g, blocks)
            rng = np.random.default_rng(GLOBAL_SEED)
            sims = []
            for _ in range(2000):
                e1, e2 = rng.choice(ev, len(ev)), rng.choice(ev, len(ev))
                n1_, n2_ = rng.choice(no, len(no)), rng.choice(no, len(no))
                sims.append((len(ev) * e1.mean() + len(no) * n1_.mean()) / (len(ev) + len(no))
                            - (len(ev) * e2.mean() + len(no) * n2_.mean()) / (len(ev) + len(no)))
            v_clean = float(np.var(sims, ddof=1))
            v_clean_src = "blockmean-pool"
        else:
            v_clean, v_clean_src = float("nan"), "zh:zh_3558(K_event<2)notzh:zh_8450"
        v_run = float(np.var(list(mbar.values()), ddof=1))  # σ²_seed + V_eval_run
        v_seed = max(0.0, v_run - v_clean) if np.isfinite(v_clean) else float("nan")
        # zh:zh_6523 (zh:zh_980test): A zh:zh_8303 run zh:zh_1852difference -> zh:prediction A×B zh:zh_2568pair d̄ zh:zh_6254difference = 2·vA
        # zh:zh_2317: zh:zh_4053 vA zh:directis  run zh:zh_1852difference (zh:seed+notzh:zh_865), notzh:zh_9936 V_clean —— zh:zh_3729is 
        # realpairedzh:zh_1787segmentzh:zh_7202, segmentzh:componentzh_2070; V_clean zh:zh_6816componentzh_3521pairedzh:design zh:zh_9513 (E7 zh:zh_0).
        rng = np.random.default_rng(GLOBAL_SEED + 1)
        ratios = []
        for rep in range(20):
            perm = rng.permutation(seeds)
            A, B = perm[:len(perm) // 2], perm[len(perm) // 2:]
            vA = float(np.var([mbar[s] for s in A], ddof=1))
            v_pred_pair = 2 * vA
            obs = [mbar[a] - mbar[b] for a in A for b in B]
            v_obs = float(np.var(obs, ddof=1))
            if v_pred_pair > 0 and v_obs > 0:
                ratios.append(v_pred_pair / v_obs)
        t = tost(ratios)
        R = (2 * (v_seed + v_clean)) / max(2 * v_run, 1e-12)
        rows.append(dict(dataset=ds, n_seeds=len(seeds), n_pairs=len(pairs),
                          win_fp=f"{pw_rej}/{len(pairs)}", blk_fp=f"{bw_rej}/{nvalid_b}" if nvalid_b else "NA",
                          V_run=v_run, V_clean=v_clean, V_clean_src=v_clean_src,
                          V_seed=v_seed, dbar_sd_across_pairs=float(dbars.std(ddof=1)),
                          R_seed_over_clean=(v_seed / v_clean) if (np.isfinite(v_clean) and v_clean > 0 and np.isfinite(v_seed)) else np.nan,
                          tost_p=t["p"], tost_equiv=t["equiv"], tost_ci=t.get("ci95"),
                          ratio_pred_over_obs=float(np.mean(ratios)) if ratios else np.nan))
        print(f"{ds}: {len(seeds)}zh:seed {len(pairs)}pair | windowzh:zh_870 {pw_rej}/{len(pairs)} | "
              f"blockzh:zh_4542 {rows[-1]['blk_fp']} | V_seed={v_seed:.2e} V_clean={v_clean:.2e} "
              f"R={rows[-1]['R_seed_over_clean']:.3f} | TOST p={t['p']:.3f} zh:etc.zh_3812={t['equiv']}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e5_analysis.csv"), index=False, encoding="utf-8-sig")
    print("\nwritten e5_analysis.csv")


if __name__ == "__main__":
    main()

"""E4 MDE withrowszh:zh_9625powercurve (pre-registered §11: t zh:zh_2806 + zh:parameter bootstrap, zh:zh_2144table).

zh:zh_350: blockzh:zh_4542paireddifference d_j = m1*_j − m2*_j (rankzh:zh_405). each datasetfrom  E3-clean  blockmeanzh:zh_5414estimate
σ_d (blockzh:zh_4542paireddifferencezh:zh_7947difference), K = zh:zh_7974cellblockzh:zh_4930; MDE = (t_{0.975,K-1} + t_{0.8,K-1}) · σ_d/√K.
AUROC zh:zh_7892: zh:zh_9131etc. E[d̄] = 2π·ΔAUROC (π=zh:anomalyrate) => ΔA = MDE/(2π)
(U zh:statisticszh_9498, zh:zh_3017/zh:zh_1003correctedzh:zh_3845 — zh:methodzh_7775).
zh:parameter bootstrap zh:powercurve: d_j ~ N(Δ, σ_d²), K block paired-t, 2000 zh:zh_4885/zh:point.
zh:zh_9272: zh:zh_5226datasetzh:zh_3757current K / 2K / 4K zh:zh_2071 ΔA=0.05 (power≥0.8).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.blocks import build_blocks, block_stats  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from bootstrap.e3_clean import load_goodness, block_mean_pools, GLOBAL_SEED  # noqa: E402

OUT = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results"
ALPHA, POWER = 0.05, 0.8
DA_TARGET = 0.05  # AUROC zh:pointdifferencezh:zh_3829


def sigma_d_from_pool(g, blocks, n_draw=2000, rng_seed=GLOBAL_SEED):
    """N2 blockmeanzh:zh_3207paired  σ_d withblockzh:zh_4930."""
    ev, no = block_mean_pools(g, blocks)
    if len(ev) < 2 or len(no) < 2:
        return np.nan, 0
    rng = np.random.default_rng(rng_seed)
    K = len(ev) + len(no)
    sds = []
    for _ in range(n_draw):
        e1, e2 = rng.choice(ev, len(ev)), rng.choice(ev, len(ev))
        n1_, n2_ = rng.choice(no, len(no)), rng.choice(no, len(no))
        sds.append(np.concatenate([e1 - e2, n1_ - n2_]).std(ddof=1))
    return float(np.mean(sds)), K


def mde(sigma_d, K):
    if not np.isfinite(sigma_d) or K < 3:
        return np.nan
    df = K - 1
    return float((stats.t.ppf(1 - ALPHA / 2, df) + stats.t.ppf(POWER, df)) * sigma_d / np.sqrt(K))


def power_curve(sigma_d, K, deltas, n_sim=2000, seed=GLOBAL_SEED):
    """zh:parameter bootstrap zh:powercurve (paired-t @ α)."""
    rng = np.random.default_rng(seed)
    out = []
    for dl in deltas:
        rej = 0
        for _ in range(n_sim):
            d = rng.normal(dl, sigma_d, K)
            if stats.ttest_1samp(d, 0.0).pvalue < ALPHA:
                rej += 1
        out.append(rej / n_sim)
    return np.array(out)


def k_needed(sigma_d, delta):
    """zh:zh_848 delta (d̄ zh:zh_405) zh:zh_2440blockzh:zh_4930 (zh:zh_9525 t≈z zh:zh_3235scan)."""
    for K in range(3, 400000):
        if mde(sigma_d, K) <= delta:
            return K
    return np.inf


def main(datasets=None, scorer="cmhmil_seed7"):
    rows, curves = [], {}
    for ds in (datasets or ["SMD", "PSM", "MSL", "SMAP", "WADI", "TE", "BATADAL", "NEweather"]):
        try:
            g, y = load_goodness(scorer, ds)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {ds}: {e}")
            continue
        T = len(g)
        blocks = build_blocks(events_from_binary(y), T)
        pi = float(y.mean())
        sd_d, K = sigma_d_from_pool(g, blocks)
        m = mde(sd_d, K)
        da = m / (2 * pi * (1 - pi)) if 0 < pi < 1 else np.nan
        k2 = k_needed(sd_d, 2 * pi * (1 - pi) * DA_TARGET)
        rows.append(dict(dataset=ds, T_test=T, anom_rate=round(pi, 4), K_blocks=K,
                          sigma_d=round(sd_d, 5), MDE_dbar=round(m, 5) if np.isfinite(m) else np.nan,
                          MDE_AUROC_pts=round(da * 100, 1) if np.isfinite(da) else np.nan,
                          K_needed_for_5AUROCpts=(k2 if k2 < 400000 else ">400000") if np.isfinite(k2) else ">400000",
                          verdict=("zh:zh_8190(ΔA=5zh:point)" if np.isfinite(da) and da <= DA_TARGET
                                   else "notzh:zh_8190" if np.isfinite(da) else "NA")))
        deltas = np.linspace(0, max(m * 2, 2 * pi * (1 - pi) * DA_TARGET), 15)
        pw = power_curve(sd_d, K, deltas)
        curves[ds] = dict(deltas=deltas.tolist(), power=pw.tolist(),
                          delta_AUROC_pts=(deltas / (2 * pi * (1 - pi)) * 100).tolist())
        print(f"{ds}: K={K} MDE(dbar)={m:.4f} MDE(dA)={da*100:.1f}pt K*={k2}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e4_mde_table.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT, "e4_power_curves.json"), "w") as f:
        json.dump(curves, f)
    print("\n== zh:zh_9272 (ΔA=0.05, power≥0.8) ==")
    print(df[["dataset", "K_blocks", "MDE_AUROC_pts", "K_needed_for_5AUROCpts", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()

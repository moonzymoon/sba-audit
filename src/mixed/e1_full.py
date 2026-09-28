"""E1 zh:zh_1343: 8 detector × 11 dataset zh:zh_7258sourcezh:zh_6254differencezh:decomposition + NDC (zh:readzh_2614B, R3 extended).

detectorzh:zh_7860: MIL(cmhmil) / zh:zh_1105(AT) / zh:zh_3107(iforest) / zh:zh_579(pca) / zh:zh_7158(gmm+ocsvm) / zh:zh_4807(lof)
dataset: SMD PSM MSL SMAP SWaT WADI TE MetroPT3 BATADAL NEweather(windowzh:zh_4542) [+TSB zh:zh_7813table]

zh:zh_7185:
1) zh:zh_7020table: (detector, dataset, seed) -> AUROC/AP;
2) zh:zh_7258sourcezh:zh_6254difference (AUROC zh:zh_405): V_dataset / V_detector / V_det×ds (MoM-ANOVA, no replicateszh:zh_5104interactionzh_468difference;
   zh:zh_6379 cmhmil zh:seedreplicateszh:zh_9168 V_seed);
3) V_eval zh:zh_897: V_clean(rankzh:zh_405 d̄*)/(2π)² -> AUROC zh:zh_405 (U zh:statisticszh_9498);
4) MSA zh:zh_624 %Contribution zh:table + NDC = 1.41·σ_det/σ_GRR (zh:zh_2348=detector, pre-registered §13 zh:readzh_2614B),
   zh:zh_826 %GRR withzh:zh_8365 AIAG zh:zh_5335 pairzh:zh_9388.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bootstrap.e3_clean import cache_file  # noqa: E402
from bootstrap.r_ratio import load_clean_dbar  # noqa: E402

CACHE = r"D:/0科研/工作1/第6篇SCI/src/_score_cache"
OUT = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results"

DATASETS = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI", "TE", "MetroPT3", "BATADAL", "NEweather"]

DETECTORS = ["cmhmil_seed7", "AT", "iforest", "pca", "gmm", "ocsvm", "lof"]
# cmhmil zh:seedreplicates (V_seed zh:zh_8958)
CMH_SEEDS = {"SMD": 10, "PSM": 10, "SWaT": 10, "MSL": 11, "SMAP": 11, "WADI": 11}


def auroc_of(f):
    d = np.load(f)
    return roc_auc_score(d["labels"], d["scores"])


def ap_of(f):
    d = np.load(f)
    return average_precision_score(d["labels"], d["scores"])


def build_table():
    rows = []
    for ds in DATASETS:
        for det in DETECTORS:
            f = cache_file(det, ds)
            if os.path.exists(f):
                rows.append(dict(dataset=ds, detector=det.replace("_seed7", ""),
                                 seed=7, AUROC=auroc_of(f), AP=ap_of(f)))
        # cmhmil zh:seedreplicates
        n_seed = CMH_SEEDS.get(ds, 1)
        for s in [0, 1, 2, 3, 31, 42, 97, 123, 2024, 4, 5, 6, 8, 9, 10][:n_seed - 1]:
            f = cache_file(f"cmhmil_seed{s}", ds)
            if os.path.exists(f):
                rows.append(dict(dataset=ds, detector="cmhmil", seed=s,
                                 AUROC=auroc_of(f), AP=ap_of(f)))
    return pd.DataFrame(rows)


def mom_two_way(df, metric="AUROC"):
    """zh:zh_9599 MoM; no replicates -> zh:interactionzh_468difference; cmhmil zh:zh_909seedzh_9617seedmean.

    zh:zh_8529decompositionzh_7488table: 6 zh:coredataset × 6 detector {cmhmil, iforest, pca, gmm, ocsvm, lof}
    (AT only 2 dataset、newdatasetdetectorzh:zh_8518notzh:zh_9405 -> zh:zh_2516table, notzh:zh_3207 MoM, zh:zh_5018record).
    """
    cmh = df[df.detector == "cmhmil"]
    v_seed_cells = cmh.groupby("dataset")[metric].var(ddof=1)
    v_seed = float(np.nanmean(v_seed_cells)) if v_seed_cells.notna().any() else 0.0
    d2 = df.copy()
    d2.loc[d2.detector == "cmhmil", metric] =         d2[d2.detector == "cmhmil"].groupby("dataset")[metric].transform("mean")
    d2 = d2.drop_duplicates(["dataset", "detector"])
    core6 = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI"]
    det6 = ["cmhmil", "iforest", "pca", "gmm", "ocsvm", "lof"]
    sub = d2[d2.dataset.isin(core6) & d2.detector.isin(det6)]
    piv = sub.pivot_table(index="dataset", columns="detector", values=metric)
    assert not piv.isna().any().any(), "zh:zh_717tablezh_9168cell: %s" % piv.isna().sum().to_dict()
    a, b, X = piv.shape[0], piv.shape[1], piv.values
    gm = X.mean()
    ms_d = b * ((X.mean(1) - gm) ** 2).sum() / (a - 1)
    ms_v = a * ((X.mean(0) - gm) ** 2).sum() / (b - 1)
    resid = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm
    ms_e = (resid ** 2).sum() / ((a - 1) * (b - 1))
    return dict(V_dataset=max(0.0, (ms_d - ms_e) / b),
                V_detector=max(0.0, (ms_v - ms_e) / a),
                V_detxds_resid=ms_e,
                V_seed=v_seed,
                n_ds=a, n_det=b)


def v_eval_auroc(ds):
    """rankzh:zh_405 V_clean -> AUROC zh:zh_405: Var(ΔA) = V_clean/(2π(1-π))². R7zh:zh_1259corrected."""
    db = load_clean_dbar(ds)
    if db is None:
        return np.nan
    y = np.load(cache_file("cmhmil_seed7", ds))["labels"]
    pi = float(y.mean())
    return float(np.var(db, ddof=1)) / (2 * pi * (1 - pi)) ** 2


def main():
    df = build_table()
    df.to_csv(os.path.join(OUT, "e1_full_metric_table.csv"), index=False, encoding="utf-8-sig")
    print(f"zh:zh_7020table: {df.groupby(['dataset', 'detector']).ngroups} cell, {len(df)} runs")
    out = {}
    for metric in ["AUROC", "AP"]:
        comps = mom_two_way(df, metric)
        ve = {ds: v_eval_auroc(ds) for ds in DATASETS}
        ve = {k: v for k, v in ve.items() if np.isfinite(v)}
        v_eval_mean = float(np.mean(list(ve.values()))) if ve else np.nan
        # MSA %Contribution (AUROC): zh:zh_7037difference = dataset + detector + zh:interactionzh_869difference + seed + eval
        tot = comps["V_dataset"] + comps["V_detector"] + comps["V_detxds_resid"] \
            + comps["V_seed"] + v_eval_mean
        tab = {
            "V_dataset": comps["V_dataset"], "V_detector": comps["V_detector"],
            "V_detxds_resid": comps["V_detxds_resid"], "V_seed": comps["V_seed"],
            "V_eval(mean)": v_eval_mean, "V_total": tot,
        }
        tab_pct = {k + "_pct": round(100 * v / tot, 1) if tot > 0 else np.nan
                   for k, v in tab.items()}
        # NDC zh:readzh_2614B: zh:zh_2348=detector; GRR = seed + eval + det×ds
        grr2 = comps["V_seed"] + v_eval_mean + comps["V_detxds_resid"]
        ndc = 1.41 * np.sqrt(comps["V_detector"] / grr2) if grr2 > 0 else np.inf
        grr_pct_sv = np.sqrt(grr2 / tot) * 100  # %Study Var
        out[metric] = dict(**{k: round(v, 8) for k, v in tab.items()}, **tab_pct,
                           NDC=round(float(ndc), 2), GRR_pctSV=round(float(grr_pct_sv), 1),
                           V_eval_by_ds={k: f"{v:.2e}" for k, v in ve.items()},
                           n_ds=comps["n_ds"], n_det=comps["n_det"])
        print(f"\n== {metric} ==")
        for k, v in out[metric].items():
            if not isinstance(v, dict):
                print(f"  {k}: {v}")
    import json
    with open(os.path.join(OUT, "e1_full_components.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nwritten e1_full_components.json + e1_full_metric_table.csv")


if __name__ == "__main__":
    main()

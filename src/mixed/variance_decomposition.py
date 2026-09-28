"""E1 zh:zh_564 (pre-registered §7.4): zh:zh_7020table + zh:zh_7258sourcezh:zh_6254differencezh:decomposition.

1) zh:zh_7020table: zh:zh_4028allcache run -> (dataset, detector, seed, AUROC, AP) zh:zh_1935table;
2) MixedLM: metric ~ 1, groups=dataset, vc={detector, seed} (REML, zh:zh_1873;
   zh:zh_3603——seed zh:zh_1309(only cmhmil×3 dataset×3 seed), E5 zh:zh_9092seedzh_9177, zh:zh_2125);
3) MoM-ANOVA: cmhmil zh:zh_7211block (3 dataset × 3 seed)  zh:zh_4253 EMS zh:zh_2387 (with MSA zh:zh_129 §3 samezh:zh_624);
4) V_residual: zh:zh_7902 E3-clean N2   d̄* zh:zh_6254differencecache (zh:componentpairzh:zh_4248, v2 zh:zh_6446 2).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE = r"D:/0科研/工作1/第6篇SCI/src/_score_cache"
RCACHE = r"D:/0科研/工作1/第15篇SCI/_resample_cache"
OUTDIR = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results"

# zh:zh_4028cache run zh:zh_7558 (zh:zh_5303segment1; E5 zh:zh_6535)
RUNS = []
for ds in ["SMD", "PSM", "SWaT"]:
    for seed in [7, 42, 123]:
        RUNS.append(("cmhmil", ds, seed))
for ds in ["MSL", "SMAP", "WADI", "TE", "MetroPT3"]:
    RUNS.append(("cmhmil", ds, 7))
for ds in ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI"]:
    RUNS.append(("iforest", ds, 0))
for ds in ["SMD", "SWaT"]:
    RUNS.append(("AT", ds, 0))


def file_of(detector, ds, seed):
    if detector == "cmhmil":
        return os.path.join(CACHE, f"cmhmil_{ds}_seed{seed}.npz")
    return os.path.join(CACHE, f"{detector}_{ds}.npz")


def metric_table():
    from sklearn.metrics import roc_auc_score, average_precision_score
    rows = []
    for det, ds, seed in RUNS:
        f = file_of(det, ds, seed)
        if not os.path.exists(f):
            continue
        d = np.load(f)
        s, y = d["scores"].astype(np.float64), d["labels"].astype(int)
        rows.append(dict(dataset=ds, detector=det, seed=seed,
                         AUROC=roc_auc_score(y, s), AP=average_precision_score(y, s)))
    return pd.DataFrame(rows)


def mixedlm_components(df, metric="AUROC"):
    """MixedLM: groups=dataset zh:randomzh_3426; vc: detector(zh:zh_5991), seed(zh:zh_6551)."""
    import statsmodels.formula.api as smf
    sub = df.copy()
    sub["detector"] = sub.detector.astype(str)
    sub["seed"] = sub.seed.astype(int).astype(str)
    try:
        md = smf.mixedlm(
            f"{metric} ~ 1", sub, groups=sub.dataset,
            vc_formula={"detector": "0 + C(detector)", "seed": "0 + C(seed)"},
            re_formula="1")
        fit = md.fit(reml=True, method="lbfgs")
        p = fit.params
        comps = {
            "V_dataset": float(p.get("Group Var", np.nan)),      # groups=dataset zh:randomzh_3426
            "V_detxds_approx": float(p.get("detector Var", np.nan)),  # zh:zh_891vc: zh:zh_4477interaction, zh:zh_1549
            "V_seed": float(p.get("seed Var", np.nan)),
            "V_residual": float(fit.scale),
            "note": "statsmodels vc is zh:zh_549(zh:zh_7501); zh:boundarynotzh:zh_8202, E5 zh:zh_9177",
        }
        return comps, fit
    except Exception as e:  # noqa: BLE001
        return dict(error=str(e)), None


def mom_anova_cmhmil(df, metric="AUROC"):
    """cmhmil zh:zh_7211block (3 dataset × 3 seed)   EMS zh:zh_1651estimate.

    zh:model X_ds = μ + D_d + S_s + ε (datasetwith seed zh:zh_6592, no replicates -> zh:interactionzh_468difference).
    MS_D / MS_S / MS_E -> σ²_dataset, σ²_seed, σ²_resid(+zh:interaction).
    """
    piv = df[(df.detector == "cmhmil") & df.seed.isin([7, 42, 123])] \
        .pivot_table(index="dataset", columns="seed", values=metric)
    piv = piv.dropna()
    if piv.shape != (3, 3):
        return dict(note=f"zh:zh_7211blocknotzh:zh_2476: {piv.shape}", n_ds=int(piv.shape[0]), n_seed=int(piv.shape[1]))
    a, b, X = piv.shape[0], piv.shape[1], piv.values
    gm = X.mean()
    ms_d = b * ((X.mean(axis=1) - gm) ** 2).sum() / (a - 1)
    ms_s = a * ((X.mean(axis=0) - gm) ** 2).sum() / (b - 1)
    resid = X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + gm
    ms_e = (resid ** 2).sum() / ((a - 1) * (b - 1))
    return dict(
        V_dataset=max(0.0, (ms_d - ms_e) / b),
        V_seed=max(0.0, (ms_s - ms_e) / a),
        V_resid_inter=ms_e,
        MS_D=ms_d, MS_S=ms_s, MS_E=ms_e,
        note="zh:interactionzh_468difference; 3x3 zh:zh_7211block, zh:zh_449value")


def v_eval_from_cache(ds, scorer="cmhmil_seed7"):
    f = os.path.join(RCACHE, f"e3clean_{ds}_{scorer}.npz")
    if not os.path.exists(f):
        return float("nan")
    d = np.load(f)["dbar"]
    return float(np.var(d, ddof=1))


def main():
    df = metric_table()
    df.to_csv(os.path.join(OUTDIR, "e1_metric_table.csv"), index=False, encoding="utf-8-sig")
    print(f"zh:zh_7020table: {len(df)} runs")
    out = {}
    for metric in ["AUROC", "AP"]:
        comps, fit = mixedlm_components(df, metric)
        out[f"MixedLM_{metric}"] = comps
        print(metric, "MixedLM:", {k: (round(v, 8) if isinstance(v, float) else v) for k, v in comps.items()})
        out[f"MoM_{metric}"] = mom_anova_cmhmil(df, metric)
        print(metric, "MoM(cmhmilzh:zh_7211block):", out[f"MoM_{metric}"])
    veval = {ds: v_eval_from_cache(ds) for ds in ["SMD", "PSM", "MSL", "SMAP", "WADI"]}
    out["V_eval_dbar_clean"] = veval
    print("V_eval (E3-clean d̄* zh:zh_6254difference, rankzh:zh_405):", {k: f"{v:.2e}" for k, v in veval.items()})
    import json
    with open(os.path.join(OUTDIR, "e1_variance_components.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("written e1_variance_components.json")


if __name__ == "__main__":
    main()

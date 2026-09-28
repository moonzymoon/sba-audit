"""E1 初版 (预注册 §7.4): 度量表 + 四源方差分解.

1) 度量表: 现有全部缓存 run -> (dataset, detector, seed, AUROC, AP) 长表;
2) MixedLM: metric ~ 1, groups=dataset, vc={detector, seed} (REML, 近似交叉结构;
   初版近似——seed 覆盖稀疏(仅 cmhmil×3 数据集×3 seed), E5 十种子后重拟, 见文档注);
3) MoM-ANOVA: cmhmil 平衡块 (3 数据集 × 3 seed) 的经典 EMS 解 (与 MSA 笔记 §3 同式);
4) V_residual: 来自 E3-clean N2 的 d̄* 方差缓存 (成分对齐, v2 必改 2).
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

# 现有缓存 run 清单 (阶段1; E5 后扩展)
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
    """MixedLM: groups=dataset 随机截距; vc: detector(交叉近似), seed(嵌套近似)."""
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
            "V_dataset": float(p.get("Group Var", np.nan)),      # groups=dataset 随机截距
            "V_detxds_approx": float(p.get("detector Var", np.nan)),  # 嵌套vc: 吸收交互, 近似
            "V_seed": float(p.get("seed Var", np.nan)),
            "V_residual": float(fit.scale),
            "note": "statsmodels vc 为组内嵌套近似(非真交叉); 边界不稳, E5 后重拟",
        }
        return comps, fit
    except Exception as e:  # noqa: BLE001
        return dict(error=str(e)), None


def mom_anova_cmhmil(df, metric="AUROC"):
    """cmhmil 平衡块 (3 数据集 × 3 seed) 的 EMS 矩估计.

    模型 X_ds = μ + D_d + S_s + ε (数据集与 seed 交叉, 无重复 -> 交互并入残差).
    MS_D / MS_S / MS_E -> σ²_dataset, σ²_seed, σ²_resid(+交互).
    """
    piv = df[(df.detector == "cmhmil") & df.seed.isin([7, 42, 123])] \
        .pivot_table(index="dataset", columns="seed", values=metric)
    piv = piv.dropna()
    if piv.shape != (3, 3):
        return dict(note=f"平衡块不完整: {piv.shape}", n_ds=int(piv.shape[0]), n_seed=int(piv.shape[1]))
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
        note="交互并入残差; 3x3 平衡块, 描述性初值")


def v_eval_from_cache(ds, scorer="cmhmil_seed7"):
    f = os.path.join(RCACHE, f"e3clean_{ds}_{scorer}.npz")
    if not os.path.exists(f):
        return float("nan")
    d = np.load(f)["dbar"]
    return float(np.var(d, ddof=1))


def main():
    df = metric_table()
    df.to_csv(os.path.join(OUTDIR, "e1_metric_table.csv"), index=False, encoding="utf-8-sig")
    print(f"度量表: {len(df)} runs")
    out = {}
    for metric in ["AUROC", "AP"]:
        comps, fit = mixedlm_components(df, metric)
        out[f"MixedLM_{metric}"] = comps
        print(metric, "MixedLM:", {k: (round(v, 8) if isinstance(v, float) else v) for k, v in comps.items()})
        out[f"MoM_{metric}"] = mom_anova_cmhmil(df, metric)
        print(metric, "MoM(cmhmil平衡块):", out[f"MoM_{metric}"])
    veval = {ds: v_eval_from_cache(ds) for ds in ["SMD", "PSM", "MSL", "SMAP", "WADI"]}
    out["V_eval_dbar_clean"] = veval
    print("V_eval (E3-clean d̄* 方差, 秩尺度):", {k: f"{v:.2e}" for k, v in veval.items()})
    import json
    with open(os.path.join(OUTDIR, "e1_variance_components.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("已写出 e1_variance_components.json")


if __name__ == "__main__":
    main()

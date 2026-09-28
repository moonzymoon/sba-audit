"""E5 分析: 十种子真值校验 + 现实轨全量 (预注册 §12 + R3).

数据: cmhmil × {SMD,PSM,SWaT}(10种子) ∪ {MSL,SMAP,WADI}(11种子) = 63 个独立 run.
产出:
1) 现实轨全量: 每数据集全部种子对 (45/55 对) 的 d̄ 与窗口级/块级 p 值 -> 经验假阳性率;
2) V_seed 识别: run 级均值方差 − V_clean(bootstrap 评估噪声) = σ²_seed (下截 0);
3) R 终值: R = 2(σ²_seed + V_clean) 的观测对照 —— 分半标定: 种子随机分 A/B 两半,
   σ²_seed 从 A 估, 观测方差取 A×B 跨组 25 对 d̄ (非循环), log 方差比 95%CI + TOST[0.67,1.5];
4) SWaT/MSL/SMAP/WADI 块级按各自段结构 (SWaT 仅窗口级).
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
    """TOST 等价检验 on 方差比 (输入为比值数组)."""
    r = np.asarray(log_ratios, float)
    r = r[np.isfinite(r) & (r > 0)]
    if len(r) < 3:
        return dict(n=len(r), p=np.nan, equiv=np.nan)
    # TOST: 双单侧 t 检验, 均值的对数尺度
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
        # run 级窗口均值 m̄_r
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
        # V_clean (E3-clean 缓存; 无缓存的非退化数据集用块均值池两次抽取近似;
        # SWaT K_event=1 退化 -> 不可估, 记 NaN)
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
            v_clean, v_clean_src = float("nan"), "退化(K_event<2)不可估"
        v_run = float(np.var(list(mbar.values()), ddof=1))  # σ²_seed + V_eval_run
        v_seed = max(0.0, v_run - v_clean) if np.isfinite(v_clean) else float("nan")
        # 分半标定 (可加性检验): A 半 run 级方差 -> 预测 A×B 跨组对 d̄ 方差 = 2·vA
        # 注: 此处 vA 直接为 run 级方差 (种子+不可抵消评估噪声), 不加 V_clean —— 因为
        # 真实配对共享段结构, 段成分噪声抵消; V_clean 含成分项故是配对设计的上界 (E7 发现).
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
        print(f"{ds}: {len(seeds)}种子 {len(pairs)}对 | 窗口级假阳 {pw_rej}/{len(pairs)} | "
              f"块级 {rows[-1]['blk_fp']} | V_seed={v_seed:.2e} V_clean={v_clean:.2e} "
              f"R={rows[-1]['R_seed_over_clean']:.3f} | TOST p={t['p']:.3f} 等价={t['equiv']}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e5_analysis.csv"), index=False, encoding="utf-8-sig")
    print("\n已写出 e5_analysis.csv")


if __name__ == "__main__":
    main()

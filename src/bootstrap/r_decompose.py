"""r 的解析分解与验证 (§6.3 升级): r = T·Var(ḡ1)/(2·Var(g)).

分解 (重抽样设计下, ḡ 为常量, Var(d̄)=Var(ḡ1)):
  ḡ1 = w·m̄*_e + (1−w)·m̄*_n   (w = 副本中事件块窗口占比)
  ≈ Term_mix    : Var(w)·(μ_e − μ_n)^2             [类混合: 副本含多少事件块]
  + Term_select : E[w²]·σ²_{m,e}·E[1/n_e] + E[(1−w)²]·σ²_{m,n}·E[1/n_n]
                                            [类内选段: 抽到哪些段]
  + residual    (长度加权、截尾、类内计数与选段耦合等)

预测只用单次运行的池统计 (μ, σ²_m) 与设计层面的 w/n 分布
(对索引数组做纯组成重抽 MC, 不触及检验机制) → r_pred 与经验 r 对比.

用法: python bootstrap/r_decompose.py
输出: results/r_decomposition.csv + 控制台摘要
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bootstrap.e3_clean import load_goodness, block_mean_pools  # noqa: E402
from common.blocks import build_blocks, resample_blocks, block_stats  # noqa: E402
from common.events import events_from_binary  # noqa: E402

RES = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results"
MC = 4000
SEED = 20260902


def decompose_cell(dataset, scorer, mc=MC):
    g, y = load_goodness(scorer, dataset)
    T = len(g)
    events = events_from_binary(y)
    blocks = build_blocks(events, T)
    # pos → kind 查找表 (索引数组重抽后可直接反查来源块)
    pos_kind = np.zeros(T, dtype=np.int8)
    pos_blk = np.full(T, -1, dtype=np.int32)
    for i, (kd, a, b) in enumerate(blocks):
        pos_kind[a:b] = 1 if kd == "event" else 0
        pos_blk[a:b] = i
    idx = np.arange(T)

    st = block_stats(g, blocks, min_seg=10)
    ev = np.array([b_["m"] for b_ in st if b_["kind"] == "event"])
    no = np.array([b_["m"] for b_ in st if b_["kind"] == "normal"])
    if len(ev) < 2 or len(no) < 2:
        return None
    mu_e, mu_n = float(ev.mean()), float(no.mean())
    v_e, v_n = float(ev.var(ddof=1)), float(no.var(ddof=1))

    # 设计层 MC: 对索引数组重抽 (与主检验同一 resample_blocks, 但 rng 独立流)
    ws = np.empty(mc)
    inv_ne = np.empty(mc)
    inv_nn = np.empty(mc)
    for b in range(mc):
        rng = np.random.default_rng(SEED + 977 * b)
        drawn = resample_blocks(idx, blocks, rng)[0]
        kat = pos_kind[drawn]
        w = kat.mean()
        # 去重块 ID 计数 (同块被抽多次仍是一个"来源段", 选段项按来源段数)
        ne = np.unique(pos_blk[drawn][kat == 1]).size
        nn_ = np.unique(pos_blk[drawn][kat == 0]).size
        ws[b] = w
        inv_ne[b] = 1.0 / max(ne, 1)
        inv_nn[b] = 1.0 / max(nn_, 1)

    term_mix = ws.var(ddof=1) * (mu_e - mu_n) ** 2
    term_sel = (ws**2).mean() * v_e * inv_ne.mean() \
        + ((1 - ws)**2).mean() * v_n * inv_nn.mean()
    var_g = float(g.var(ddof=1))
    r_pred = T * (term_mix + term_sel) / (2 * var_g)
    return dict(dataset=dataset, scorer=scorer, T=T,
                K_e=int((pos_kind == 1).sum()), K_n=int((pos_kind == 0).sum()),
                mu_diff=mu_e - mu_n, var_w=ws.var(ddof=1), w_mean=ws.mean(),
                term_mix=term_mix, term_sel=term_sel,
                share_mix=term_mix / (term_mix + term_sel),
                var_g=var_g, r_pred=r_pred)


def main():
    cells = pd.read_csv(os.path.join(RES, "inflation_factor.csv"))
    rows = []
    for _, c in cells.iterrows():
        try:
            out = decompose_cell(c.dataset, c.scorer)
        except Exception as e:  # 缺缓存/退化结构: 记录不中断
            print(f"  skip {c.dataset}/{c.scorer}: {type(e).__name__} {e}")
            out = None
        if out is not None:
            out["r_emp"] = c.r
            out["pred_size"] = c.pred_size
            out["size_win_t"] = c.size_win_t
            rows.append(out)
        print(f"  done {c.dataset}/{c.scorer}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "r_decomposition.csv"), index=False, encoding="utf-8-sig")
    lr_p, lr_e = np.log(df.r_pred), np.log(df.r_emp)
    print("\n=== r 解析分解验证 ===")
    print(f"cells: {len(df)}")
    print(f"corr(log r_pred, log r_emp) = {np.corrcoef(lr_p, lr_e)[0,1]:.3f}")
    sl = np.polyfit(lr_e, lr_p, 1)
    print(f"OLS log-log slope = {sl[0]:.3f} (1=理想), intercept = {sl[1]:.3f}")
    print(f"median share_mix (类混合占比) = {df.share_mix.median():.3f} "
          f"(IQR {df.share_mix.quantile(.25):.3f}–{df.share_mix.quantile(.75):.3f})")
    print(f"r_pred/r_emp 几何均值 = {np.exp((lr_p - lr_e).mean()):.3f}")
    print(df[["dataset", "scorer", "r_emp", "r_pred", "share_mix",
              "K_e", "K_n"]].to_string(index=False))


if __name__ == "__main__":
    main()

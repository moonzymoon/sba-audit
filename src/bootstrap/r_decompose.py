"""r  zh:parsedecompositionwithzh:verify (§6.3 zh:zh_1955): r = T·Var(ḡ1)/(2·Var(g)).

zh:decomposition (zh:resamplingdesignzh_5169, ḡ is zh:zh_465, Var(d̄)=Var(ḡ1)):
  ḡ1 = w·m̄*_e + (1−w)·m̄*_n   (w = zh:zh_8029eventblockwindowzh:zh_1750)
  ≈ Term_mix    : Var(w)·(μ_e − μ_n)^2             [zh:zh_9530: zh:zh_2194eventblock]
  + Term_select : E[w²]·σ²_{m,e}·E[1/n_e] + E[(1−w)²]·σ²_{m,n}·E[1/n_n]
                                            [zh:zh_9936segment: zh:zh_4937segment]
  + residual    (zh:lengthzh_2944、zh:zh_448、zh:zh_2173withzh:zh_3970segmentzh:zh_6308etc.)

zh:predictionzh_5648row zh:zh_5414statistics (μ, σ²_m) withzh:designzh_6945  w/n zh:zh_2806
(pairindexzh:arrayzh_9431pure zh:zh_4623 MC, notzh:zh_1226andzh:testzh_678) → r_pred withempirical r pairzh:zh_8961.

zh:zh_966: python bootstrap/r_decompose.py
writes: results/r_decomposition.csv + zh:zh_3200
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

RES = r"D:/0keyan/gongzuo1/paper 15SCI/02_shiyanjilu/results"
MC = 4000
SEED = 20260902


def decompose_cell(dataset, scorer, mc=MC):
    g, y = load_goodness(scorer, dataset)
    T = len(g)
    events = events_from_binary(y)
    blocks = build_blocks(events, T)
    # pos → kind zh:zh_9007table (indexzh:arrayzh_9151directzh_8600sourceblock)
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

    # zh:designzh_419 MC: pairindexzh:arrayzh_17 (withzh:zh_8529testsamezh:zh_6443 resample_blocks, zh:zh_3107 rng zh:zh_1073)
    ws = np.empty(mc)
    inv_ne = np.empty(mc)
    inv_nn = np.empty(mc)
    for b in range(mc):
        rng = np.random.default_rng(SEED + 977 * b)
        drawn = resample_blocks(idx, blocks, rng)[0]
        kat = pos_kind[drawn]
        w = kat.mean()
        # zh:zh_3226block ID zh:zh_4001 (sameblockzh:zh_5624 "zh:zh_1325sourcesegment", zh:zh_3970segmentzh:zh_9939perzh:zh_1325sourcesegmentzh:zh_4930)
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
        except Exception as e:  # zh:zh_1963cache/zh:zh_4068: recordnotzh:zh_6644
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
    print("\n=== r zh:parsedecompositionverify ===")
    print(f"cells: {len(df)}")
    print(f"corr(log r_pred, log r_emp) = {np.corrcoef(lr_p, lr_e)[0,1]:.3f}")
    sl = np.polyfit(lr_e, lr_p, 1)
    print(f"OLS log-log slope = {sl[0]:.3f} (1=zh:zh_5289), intercept = {sl[1]:.3f}")
    print(f"median share_mix (zh:zh_8075) = {df.share_mix.median():.3f} "
          f"(IQR {df.share_mix.quantile(.25):.3f}–{df.share_mix.quantile(.75):.3f})")
    print(f"r_pred/r_emp zh:zh_7829mean = {np.exp((lr_p - lr_e).mean()):.3f}")
    print(df[["dataset", "scorer", "r_emp", "r_pred", "share_mix",
              "K_e", "K_n"]].to_string(index=False))


if __name__ == "__main__":
    main()

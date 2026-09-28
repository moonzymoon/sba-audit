# -*- coding: utf-8 -*-
"""合成因子网格 (R6 增补): Gaussian 序列 + 受控段结构, 同一副本机制.

因子: K(事件块数, 等量正常块) x phi(块内 AR1 依赖) x sigma_b(块均值异质性).
每格 B=2000 副本: 窗口级 paired-t vs 块级 paired-t 的经验 size,
实测 r = T*Var(gbar*)/(2*Var(g)) 与解析预测 2*Phi(-1.96/sqrt(r)) 对照.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lfilter

SEED = 20260903
L = 200          # 块长(窗口)
B = 5000

def gen_series(K, phi, sigma_b, rng):
    T = 2 * K * L
    e = rng.normal(0, np.sqrt(1 - phi**2), T)
    x = lfilter([1.0], [1.0, -phi], e)          # 平稳 AR(1), var=1
    be = rng.normal(0, sigma_b, K)
    bn = rng.normal(0, sigma_b, K)
    g = x.reshape(2 * K, L).copy()
    for j in range(2 * K):
        g[j] += be[j // 2] if j % 2 == 1 else bn[j // 2]
    return g.reshape(-1)

def run_cell(K, phi, sigma_b, rng):
    g = gen_series(K, phi, sigma_b, rng)
    T = len(g)
    ev = np.arange(2 * K)[1::2]                  # 事件块索引
    no = np.arange(2 * K)[0::2]
    ev_means = g.reshape(2 * K, L)[ev].mean(axis=1)
    no_means = g.reshape(2 * K, L)[no].mean(axis=1)
    v_g = g.var(ddof=1)
    p_win = np.empty(B); p_blk = np.empty(B); rbar = np.empty(B)
    for b in range(B):
        # 副本 = 抽中块按 normal/event 交错结构拼接
        seq_e = rng.choice(ev, K, replace=True)
        seq_n = rng.choice(no, K, replace=True)
        gs = np.concatenate([np.concatenate([g[s * L:(s + 1) * L], g[t * L:(t + 1) * L]])
                             for s, t in zip(seq_n, seq_e)])
        d = g - gs
        p_win[b] = stats.ttest_1samp(d, 0.0).pvalue
        a = rng.choice(ev_means, K, replace=True); c = rng.choice(ev_means, K, replace=True)
        u = rng.choice(no_means, K, replace=True); v = rng.choice(no_means, K, replace=True)
        dblk = np.concatenate([a - c, u - v])
        p_blk[b] = stats.ttest_1samp(dblk, 0.0).pvalue
        rbar[b] = gs.mean()
    r_hat = T * rbar.var(ddof=1) / (2 * v_g)
    return r_hat, (p_win < .05).mean(), (p_blk < .05).mean()

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    rows = []
    for K in [8, 32, 64]:
        for phi in [0.0, 0.9]:
            for sb in [0.0, 0.5, 2.0]:
                r, pw, pb = run_cell(K, phi, sb, rng)
                pred = 2 * stats.norm.sf(1.96 / np.sqrt(r))
                rows.append(dict(K=K, phi=phi, sigma_b=sb, r=r, pred=pred, win_t=pw, blk_t=pb))
                print("K=%2d phi=%.1f sb=%.1f  r=%8.1f  pred=%.3f  win=%.3f  blk=%.3f"
                      % (K, phi, sb, r, pred, pw, pb))
    out = "D:/0科研/工作1/第15篇SCI/02_实验记录/results/synthetic_grid.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved:", out)

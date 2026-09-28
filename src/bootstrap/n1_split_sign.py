"""E3-clean N1: 块级随机对分 + 符号置换 (预注册 §7.1).

N1 = 同一条缓存分数序列, 把块(事件段+正常间隔)随机对分为 G1/G2 (按类型分层),
伪检测器 A=(s,G1), B=(s,G2) —— H0 按构造成立; 参考检验 = 块级配对差的符号置换.
符号置换以块为单位 (整块翻符号), 是校准参照: 其经验 size 应 ≈ α.
"""
from __future__ import annotations

import numpy as np


def split_halves(block_stat_list, rng):
    """块级统计量随机对分 (按 kind 分层, 各取一半入 G1/G2).

    block_stat_list: [{block,kind,n,m,...}] — block_stats() 的输出.
    Returns: (g1_means, g2_means) 两组块均值数组 (进入配对序列的块).
    """
    idx = {"event": [], "normal": []}
    for j, b in enumerate(block_stat_list):
        idx[b["kind"]].append(j)
    g1, g2 = [], []
    for kind, ids in idx.items():
        ids = np.asarray(ids)
        rng.shuffle(ids)
        half = len(ids) // 2
        g1.extend(ids[:half])
        g2.extend(ids[half:2 * half])  # 奇数个时余一块不入配对(双侧对称)
    m = np.array([block_stat_list[j]["m"] for j in range(len(block_stat_list))])
    return m[g1], m[g2]


def sign_perm_test(m, rng, n_perm=999):
    """块级配对差 m_k (已中心化前) 的符号置换检验.

    H0: m_k 对称分布于 0. 统计量 = mean(m); 置换 = 整体翻符号.
    返回双侧 p 值 (含观测值在内的置换分布).
    """
    m = np.asarray(m, float)
    if len(m) < 3:
        return float("nan")
    obs = m.mean()
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(m)))
    null = (signs * m).mean(axis=1)
    allv = np.concatenate(([obs], null))
    p = (np.abs(allv) >= abs(obs) - 1e-12).mean()
    return float(p)


def n1_pvalue(d, blocks, rng, min_seg=10):
    """一次 N1 重复: 分层对分 + 分层组置换 p 值.

    置换必须保持事件/正常块的组内平衡 (与对分的分层机制一致),
    否则置换零分布过宽 → 检验系统性保守 (B=500 全不拒绝的教训).
    """
    from common.blocks import block_stats
    bstat = [b for b in block_stats(d, blocks, min_seg=min_seg)]
    ev = np.array([b["m"] for b in bstat if b["kind"] == "event"])
    no = np.array([b["m"] for b in bstat if b["kind"] == "normal"])
    if len(ev) < 2 or len(no) < 2:
        return float("nan")

    def stat(e1, e2, n1v, n2v):
        return (np.concatenate([e1, n1v]).mean() - np.concatenate([e2, n2v]).mean())

    # 观测: 各 kind 内对分
    rng.shuffle(ev)
    rng.shuffle(no)
    he, ho = len(ev) // 2, len(no) // 2
    obs = stat(ev[:he], ev[he:2 * he], no[:ho], no[ho:2 * ho])
    if he < 1 or ho < 1:
        return float("nan")
    n_perm = 999
    pv = np.empty(n_perm)
    for b in range(n_perm):
        pe, pn = rng.permutation(2 * he), rng.permutation(2 * ho)
        eall = np.concatenate([ev[:he], ev[he:2 * he]])
        nall = np.concatenate([no[:ho], no[ho:2 * ho]])
        pv[b] = stat(eall[pe[:he]], eall[pe[he:2 * he]],
                     nall[pn[:ho]], nall[pn[ho:2 * ho]])
    allv = np.concatenate(([obs], pv))
    return float((np.abs(allv) >= abs(obs) - 1e-12).mean())

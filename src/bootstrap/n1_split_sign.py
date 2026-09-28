"""E3-clean N1: blockzh:zh_4542randompairzh:zh_746 + signpermutation (pre-registered §7.1).

N1 = samezh:zh_6443rowscachezh:scoreseries, zh:zh_1239block(zh:eventsegment+zh:normalzh_6966)zh:randompairzh:zh_746is  G1/G2 (perzh:typestratified),
zh:zh_2493detector A=(s,G1), B=(s,G2) —— H0 perzh:constructionzh_2725; zh:referencetest = blockzh:zh_4542paireddifference signpermutation.
signpermutationzh:zh_5665blockis zh:zh_696bit (zh:zh_8990blockzh:zh_4670sign), zh:zh_5050calibrationreference: zh:zh_7753empirical size zh:zh_2483 ≈ α.
"""
from __future__ import annotations

import numpy as np


def split_halves(block_stat_list, rng):
    """blockzh:zh_4542statisticszh_1031randompairzh:zh_746 (per kind stratified, zh:zh_2897 G1/G2).

    block_stat_list: [{block,kind,n,m,...}] — block_stats()  writes.
    Returns: (g1_means, g2_means) zh:zh_4920blockmeanzh:array (zh:zh_534pairedzh:series block).
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
        g2.extend(ids[half:2 * half])  # zh:zh_6571 zh:zh_1126blocknotzh:zh_854paired(two-zh:zh_2149pairzh:zh_3065)
    m = np.array([block_stat_list[j]["m"] for j in range(len(block_stat_list))])
    return m[g1], m[g2]


def sign_perm_test(m, rng, n_perm=999):
    """blockzh:zh_4542paireddifference m_k (zh:zh_4345)  signpermutationzh:test.

    H0: m_k pairzh:zh_2246 0. zh:statisticszh_1031 = mean(m); permutation = zh:zh_5163sign.
    returnstwo-zh:zh_2149 p zh:value (zh:zh_1640valuezh_8095 permutationzh:zh_2806).
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
    """zh:zh_3485 N1 replicates: stratifiedpairzh:zh_746 + stratifiedzh:zh_2928permutation p zh:value.

    permutationzh:mustzh_8172event/zh:normalblock zh:zh_6940 (withpairzh:zh_746 stratifiedzh:zh_678identical),
    zh:otherwisepermutationzh:zh_5472 → zh:testzh_3059 (B=500 zh:zh_9405notzh:zh_5465 zh:zh_3874).
    """
    from common.blocks import block_stats
    bstat = [b for b in block_stats(d, blocks, min_seg=min_seg)]
    ev = np.array([b["m"] for b in bstat if b["kind"] == "event"])
    no = np.array([b["m"] for b in bstat if b["kind"] == "normal"])
    if len(ev) < 2 or len(no) < 2:
        return float("nan")

    def stat(e1, e2, n1v, n2v):
        return (np.concatenate([e1, n1v]).mean() - np.concatenate([e2, n2v]).mean())

    # zh:zh_6331: zh:zh_5226 kind zh:zh_3990pairzh:zh_746
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

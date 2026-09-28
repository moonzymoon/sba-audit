"""E3-clean driver (pre-registered §7.1): pure zh:zh_9398assumptionzh_5169 empirical Type I.

zh:zh_4307auditzh:zh_1063 (each cell B=10000 zh:zh_6249replicates, recordeach zh:zh_6249 p zh:value):
  ① windowzh:zh_4542 paired-t        — d_t treated as iid (reallunwen windowzh:zh_4557)
  ② windowzh:zh_4542 Wilcoxon signed-rank
  ③ blockzh:zh_4542 paired-t          — m_k (blockzh:zh_3990 d mean) treated as iid
  ④ blockzh:zh_4542 Wilcoxon
  ⑤ blockzh:zh_4542signpermutation (calibrationreference, N1 pairzh:zh_1456)
zh:zh_8529definition: N2 paired (s, s*) — pre-registeredfrozenoriginal text;
zh:zh_538 robust: (s1*, s2*) zh:zh_222 bootstrap zh:zh_8698paired — zh:zh_3417sameblockzh:zh_17 zh:zh_6336 (zh:zh_3218, zh:zh_4704).

zh:zh_966: python bootstrap/e3_clean.py SMD cmhmil_seed7 [B]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.blocks import build_blocks, resample_blocks, block_stats  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from bootstrap.n1_split_sign import n1_pvalue  # noqa: E402

CACHE = r"D:/0keyan/gongzuo1/paper 6SCI/src/_score_cache"
RCACHE = r"D:/0keyan/gongzuo1/paper 15SCI/_resample_cache"
GLOBAL_SEED = 20260902  # pre-registered §6


def cache_file(scorer, dataset):
    """'cmhmil_seed7' -> cmhmil_SMD_seed7.npz; 'iforest' -> iforest_SMD.npz."""
    name = scorer.replace("_seed", f"_{dataset}_seed") if "_seed" in scorer else f"{scorer}_{dataset}"
    return os.path.join(CACHE, f"{name}.npz")


def load_goodness(scorer, dataset):
    """rankzh:zh_405signzh:zh_4040 (pre-registered §4 + zh:zh_6924record R1).

    R1: cachezh:zh_3710scorezh_5936 (cmhmil |s| zh:zh_9468 5e13, zh:zh_4657bitzh_4930 1.6), zh:zh_2701paireddifferencezh:zh_7495 t zh:testzh_7153、
    zh:zh_6254differencezh:zh_2616blockzh:zh_7467. zh:zh_2562 run zh:zh_3990rankzh:zh_2034 rank(s)/N (zh:zh_6700rank) —— withzh:zh_8529metric AUROC  
    U zh:statisticszh_170pairzh:zh_4248. zh:zh_6670pairzh:zh_5145, notzh:zh_8946assumptionzh_8116.
    """
    from scipy.stats import rankdata
    d = np.load(cache_file(scorer, dataset))
    s, y = d["scores"].astype(np.float64), d["labels"].astype(np.int8)
    r = rankdata(s) / len(s)
    g = r * (2.0 * y - 1.0)  # signzh:zh_4040, rankzh:zh_405
    return g, y


def block_mean_pools(g, blocks, min_seg=10):
    """zh:zh_7974cellblockmeanzh:zh_5414 (per kind zh:zh_746, zh:zh_6924 R2  blockzh:zh_4542pairedzh:zh_9968)."""
    st = block_stats(g, blocks, min_seg=min_seg)
    ev = np.array([b["m"] for b in st if b["kind"] == "event"])
    no = np.array([b["m"] for b in st if b["kind"] == "normal"])
    return ev, no


def block_level_pvalues(g, blocks, rng, min_seg=10):
    """blockzh:zh_4714 ③④⑤ (zh:zh_6924 R2): blockmeanzh:zh_419pairedzh:zh_1777——**zh:zh_3207paired**.

    d_j = m1*_j − m2*_j, zh:zh_5329from same kind zh:zh_7974cellblockmeanzh:zh_4598.
    E[mean(d_j)]=0 zh:zh_3902, andzh:zh_3711differencezh:zh_7202match (zh:zh_333fixed  original-vs-zh:zh_6966
    zh:zh_3783 t zh:zh_5953 √2 → zh:zh_9707). zh:zh_5917auditzh:zh_9542: zh:zh_1666 zh:seed = zh:zh_6429.
    ⑤ signpermutation = pair {d_j} zh:zh_5163sign (pre-registeredoriginal text zh:zh_2799).
    """
    ev, no = block_mean_pools(g, blocks, min_seg)
    if len(ev) < 2 or len(no) < 2:
        return np.nan, np.nan, np.nan
    e1 = rng.choice(ev, size=len(ev), replace=True)
    e2 = rng.choice(ev, size=len(ev), replace=True)
    n1_ = rng.choice(no, size=len(no), replace=True)
    n2_ = rng.choice(no, size=len(no), replace=True)
    dblk = np.concatenate([e1 - e2, n1_ - n2_])
    p3 = float(stats.ttest_1samp(dblk, 0.0).pvalue) if len(dblk) >= 3 else np.nan
    try:
        p4 = float(stats.wilcoxon(dblk).pvalue) if len(dblk) >= 6 else np.nan
    except ValueError:
        p4 = 1.0
    obs = dblk.mean()
    signs = rng.choice((-1.0, 1.0), size=(499, len(dblk)))
    null = (signs * dblk).mean(axis=1)
    p5 = float((np.abs(np.concatenate(([obs], null))) >= abs(obs) - 1e-12).mean())
    return p3, p4, p5


def proto_pvalues(d, blocks, rng, min_seg=10):
    """windowzh:zh_4714 ①② (blockzh:zh_5499 block_level_pvalues, zh:zh_6924 R2)."""
    p1 = float(stats.ttest_1samp(d, 0.0).pvalue)
    try:
        p2 = float(stats.wilcoxon(d).pvalue)
    except ValueError:  # zh:zh_2770difference
        p2 = 1.0
    return p1, p2


def run(dataset, scorer, B=10000, shard=None, n_shards=1):
    g, y = load_goodness(scorer, dataset)
    T = len(g)
    events = events_from_binary(y)
    blocks = build_blocks(events, T)
    ev_names = ["win_t", "win_wilcox", "blk_t", "blk_wilcox", "blk_signperm",
                "n1_signperm", "r_win_t", "r_win_wilcox", "r_blk_t", "r_blk_wilcox"]
    res = {k: np.full(B, np.nan) for k in ev_names}
    dbar = np.full(B, np.nan)
    t0 = time.time()
    lo = 0 if shard is None else B * shard // n_shards
    hi = B if shard is None else B * (shard + 1) // n_shards
    for b in range(lo, hi):
        rng = np.random.default_rng(GLOBAL_SEED + b)  # each zh:zh_6249replicateszh:zh_3030reproduction
        g1 = resample_blocks(g, blocks, rng)[0]
        d = g - g1
        p1, p2 = proto_pvalues(d, blocks, rng)                      # windowzh:zh_4542 ①②
        p3, p4, p5 = block_level_pvalues(g, blocks, rng)            # blockzh:zh_4542 ③④⑤ (zh:zh_6924 R2)
        p6 = n1_pvalue(g, blocks, rng)                              # N1 pairzh:zh_746reference
        res["win_t"][b - lo], res["win_wilcox"][b - lo] = p1, p2
        res["blk_t"][b - lo], res["blk_wilcox"][b - lo] = p3, p4
        res["blk_signperm"][b - lo] = p5
        res["n1_signperm"][b - lo] = p6
        dbar[b - lo] = d.mean()
        # robust: windowzh:zh_4542=zh:zh_6313; blockzh:zh_4542=blockmeanzh:zh_4283
        rng2 = np.random.default_rng(GLOBAL_SEED + 1000000 + b)
        ga = resample_blocks(g, blocks, rng2)[0]
        gb = resample_blocks(g, blocks, rng2)[0]
        q1, q2 = proto_pvalues(ga - gb, blocks, rng2)
        q3, q4, _ = block_level_pvalues(g, blocks, rng2)
        res["r_win_t"][b - lo], res["r_win_wilcox"][b - lo] = q1, q2
        res["r_blk_t"][b - lo], res["r_blk_wilcox"][b - lo] = q3, q4
        if (b - lo + 1) % 1000 == 0:
            el = time.time() - t0
            print(f"  [{dataset}/{scorer}] {b - lo + 1}/{hi - lo} reps, {el:.0f}s", flush=True)
    os.makedirs(RCACHE, exist_ok=True)
    tag = f"{dataset}_{scorer}" + (f"_sh{shard}" if shard is not None else "")
    np.savez_compressed(os.path.join(RCACHE, f"e3clean_{tag}.npz"),
                        B=B, T=T, dbar=dbar,
                        n_blocks=len(blocks), **res)
    # zh:zh_5435
    alpha = 0.05
    out = {"dataset": dataset, "scorer": scorer, "B": hi - lo, "T": T,
           "n_blocks": len(blocks)}
    for k in ev_names:
        v = res[k]
        v = v[~np.isnan(v)]
        out[f"size_{k}"] = float((v < alpha).mean())
        out[f"n_valid_{k}"] = len(v)
    out["V_clean_dbar"] = float(np.nanvar(dbar, ddof=1))
    return out


def main():
    dataset, scorer = sys.argv[1], sys.argv[2]
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    out = run(dataset, scorer, B)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})


if __name__ == "__main__":
    main()

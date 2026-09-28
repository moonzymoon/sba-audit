"""E3-clean 驱动 (预注册 §7.1): 纯评估噪声零假设下的经验 Type I.

被审计协议 (每格 B=10000 次重复, 记录每次 p 值):
  ① 窗口级 paired-t        — d_t 视为 iid (真实论文的窗口级做法)
  ② 窗口级 Wilcoxon 符号秩
  ③ 块级 paired-t          — m_k (块内 d 均值) 视为 iid
  ④ 块级 Wilcoxon
  ⑤ 块级符号置换 (校准参照, N1 对分实现)
主定义: N2 配对 (s, s*) — 预注册冻结原文;
附列 robust: (s1*, s2*) 两独立 bootstrap 副本配对 — 防同块重抽的零簇伪影 (文档化扩展, 非静默).

用法: python bootstrap/e3_clean.py SMD cmhmil_seed7 [B]
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

CACHE = r"D:/0科研/工作1/第6篇SCI/src/_score_cache"
RCACHE = r"D:/0科研/工作1/第15篇SCI/_resample_cache"
GLOBAL_SEED = 20260902  # 预注册 §6


def cache_file(scorer, dataset):
    """'cmhmil_seed7' -> cmhmil_SMD_seed7.npz; 'iforest' -> iforest_SMD.npz."""
    name = scorer.replace("_seed", f"_{dataset}_seed") if "_seed" in scorer else f"{scorer}_{dataset}"
    return os.path.join(CACHE, f"{name}.npz")


def load_goodness(scorer, dataset):
    """秩尺度符号裕量 (预注册 §4 + 修订记录 R1).

    R1: 缓存原始分数重尾 (cmhmil |s| 至 5e13, 中位数 1.6), 原始尺度配对差使 t 检验崩溃、
    方差被离群块主导. 改用 run 内秩变换 rank(s)/N (并列取平均秩) —— 与主指标 AUROC 的
    U 统计量结构对齐. 两臂对称施加, 不影响零假设成立性.
    """
    from scipy.stats import rankdata
    d = np.load(cache_file(scorer, dataset))
    s, y = d["scores"].astype(np.float64), d["labels"].astype(np.int8)
    r = rankdata(s) / len(s)
    g = r * (2.0 * y - 1.0)  # 符号裕量, 秩尺度
    return g, y


def block_mean_pools(g, blocks, min_seg=10):
    """合格块均值池 (按 kind 分, 修订 R2 的块级配对原料)."""
    st = block_stats(g, blocks, min_seg=min_seg)
    ev = np.array([b["m"] for b in st if b["kind"] == "event"])
    no = np.array([b["m"] for b in st if b["kind"] == "normal"])
    return ev, no


def block_level_pvalues(g, blocks, rng, min_seg=10):
    """块级协议 ③④⑤ (修订 R2): 块均值层配对自助——**两次独立抽取配对**.

    d_j = m1*_j − m2*_j, 两者皆从同 kind 合格块均值池有放回独立抽取.
    E[mean(d_j)]=0 精确成立, 且分子分母方差结构匹配 (单侧固定的 original-vs-一次重抽
    会把 t 压低 √2 → 系统保守). 两次独立抽取也更贴近被审计场景: 两个种子 = 两次实现.
    ⑤ 符号置换 = 对 {d_j} 整体翻符号 (预注册原文的字面实现).
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
    """窗口级协议 ①② (块级协议已移至 block_level_pvalues, 修订 R2)."""
    p1 = float(stats.ttest_1samp(d, 0.0).pvalue)
    try:
        p2 = float(stats.wilcoxon(d).pvalue)
    except ValueError:  # 全零差
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
        rng = np.random.default_rng(GLOBAL_SEED + b)  # 每次重复可独立复现
        g1 = resample_blocks(g, blocks, rng)[0]
        d = g - g1
        p1, p2 = proto_pvalues(d, blocks, rng)                      # 窗口级 ①②
        p3, p4, p5 = block_level_pvalues(g, blocks, rng)            # 块级 ③④⑤ (修订 R2)
        p6 = n1_pvalue(g, blocks, rng)                              # N1 对分参照
        res["win_t"][b - lo], res["win_wilcox"][b - lo] = p1, p2
        res["blk_t"][b - lo], res["blk_wilcox"][b - lo] = p3, p4
        res["blk_signperm"][b - lo] = p5
        res["n1_signperm"][b - lo] = p6
        dbar[b - lo] = d.mean()
        # robust: 窗口级=两独立副本; 块级=块均值池独立再抽
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
    # 汇总
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

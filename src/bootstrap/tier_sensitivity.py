"""zh:zh_1918blockzh:zh_1935sensitivezh_4316 runner (pre-registered §8).

pairzh:zh_1018 (dataset, scorer) zh:zh_3757 b_short / b_mid / b_long zh:zh_3016rerun N2 block bootstrap
(zh:eventsegmentblockzh:zh_443notzh:zh_311, zh:zh_5561normalzh_6966block), zh:reportzh_7644windowzh:zh_4542/blockzh:zh_4542 Type I with SE,
zh:zh_598"blockzh:zh_5736notzh:zh_7467": zh:zh_1918 Type I ∈ [0.035, 0.065] and SE zh:zh_9362pairdifference <25%.

zh:zh_966: python bootstrap/tier_sensitivity.py SMD cmhmil_seed7 [B]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.blocks import build_blocks, block_stats  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from bootstrap.e3_clean import (load_goodness, proto_pvalues,  # noqa: E402
                                block_level_pvalues, GLOBAL_SEED)
from bootstrap.block_lengths import tier_lengths, resample_blocks_tier  # noqa: E402

RCACHE = r"D:/0科研/工作1/第15篇SCI/_resample_cache"


def run_tiers(dataset, scorer, B=10000, seed_offset=5000000):
    g, y = load_goodness(scorer, dataset)
    T = len(g)
    events = events_from_binary(y)
    blocks = build_blocks(events, T)
    tiers = tier_lengths(g, events, T)
    out = {"dataset": dataset, "scorer": scorer, "B": B, **tiers}
    t0 = time.time()
    for name, b in tiers.items():
        se_dbar, rej_wt, rej_bw = [], 0, 0
        for i in range(B):
            rng = np.random.default_rng(GLOBAL_SEED + seed_offset + i)
            g1 = resample_blocks_tier(g, blocks, rng, b=b)[0]
            d = g - g1
            p1 = float(stats.ttest_1samp(d, 0.0).pvalue)
            p3, _, _ = block_level_pvalues(g, blocks, rng)  # zh:zh_6924 R2: blockmeanzh:zh_419paired
            rej_wt += p1 < 0.05
            rej_bw += (not np.isnan(p3)) and (p3 < 0.05)
            se_dbar.append(d.mean())
        out[f"size_win_t_{name}"] = rej_wt / B
        out[f"size_blk_t_{name}"] = rej_bw / B
        out[f"se_dbar_{name}"] = float(np.std(se_dbar, ddof=1))
        print(f"  {name}(b={b:.0f}): win_t={rej_wt / B:.4f} blk_t={rej_bw / B:.4f} "
              f"SE(d̄)={out[f'se_dbar_{name}']:.2e}  [{time.time() - t0:.0f}s]", flush=True)
    ses = [out[f"se_dbar_{k}"] for k in tiers]
    out["SE_relspread"] = float((max(ses) - min(ses)) / np.mean(ses))
    blk_sizes = [out[f"size_blk_t_{k}"] for k in tiers]
    out["blk_all_in_band"] = bool(all(0.035 <= s <= 0.065 for s in blk_sizes))
    out["verdict"] = "blockzh:zh_5736notzh:zh_7467" if (out["blk_all_in_band"] and out["SE_relspread"] < 0.25) \
        else "pairblockzh:zh_7202sensitive(zh:zh_9363)"
    os.makedirs(RCACHE, exist_ok=True)
    np.savez_compressed(os.path.join(RCACHE, f"tiers_{dataset}_{scorer}.npz"), **out)
    return out


def main():
    ds, sc = sys.argv[1], sys.argv[2]
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    out = run_tiers(ds, sc, B)
    print(out["verdict"], "| SE_relspread =", round(out["SE_relspread"], 3))


if __name__ == "__main__":
    main()

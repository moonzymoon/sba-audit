"""三档块长 (预注册 §8): b_short / b_mid(PWSD 自动) / b_long.

块结构硬约束优先: 事件段块永不切分; 三档块长只作用于**正常间隔块内部**的子块划分
(重采样前把正常块按长度 b 切成子块, 再与完整事件段块一起进重抽样池).
b_short = max(段长中位数, w/Δ);  b_mid = arch PWSD (Politis-White 修正版 Patton 2009);
b_long = 2·b_mid. 判定"块选择不主导": 三档 E3-clean Type I ∈ [0.035,0.065] 且 SE 相对差 <25%.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.blocks import build_blocks  # noqa: E402
from common.events import events_from_binary  # noqa: E402


def tier_lengths(g, events, T, w=16, delta=1.0):
    """返回 (b_short, b_mid, b_long)."""
    ev_lens = [e - s for s, e in events if e - s >= 10]
    b_short = max(float(np.median(ev_lens)) if ev_lens else w, w / delta)
    try:
        from arch.bootstrap import optimal_block_length
        obl = optimal_block_length(g.reshape(-1, 1))
        b_mid = float(obl.loc["g", "stationary"]) if "g" in set(obl.index.get_level_values(0)) \
            else float(obl.iloc[0, 0])
    except Exception as e:  # noqa: BLE001
        print(f"[block_lengths] PWSD 失败({e}), 回退 b_mid = b_short")
        b_mid = b_short
    b_long = 2.0 * b_mid
    return dict(b_short=b_short, b_mid=b_mid, b_long=b_long)


def subdivide_normals(blocks, b):
    """把正常块按长度 b 切子块 (事件块原样保留) — 供三档敏感性用."""
    out = []
    for kind, s, e in blocks:
        if kind == "event" or e - s <= b:
            out.append((kind, s, e))
        else:
            pos = s
            while pos < e:
                out.append((kind, pos, min(pos + int(b), e)))
                pos += int(b)
    return out


def resample_blocks_tier(x, blocks, rng, b=None):
    """带块长档位的重采样: 正常块先按 b 细分再重抽; 事件块整块进出."""
    from common.blocks import resample_blocks
    if b is None:
        return resample_blocks(x, blocks, rng)
    return resample_blocks(x, subdivide_normals(blocks, b), rng)

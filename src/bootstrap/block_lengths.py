"""zh:zh_1918blockzh:zh_1935 (pre-registered §8): b_short / b_mid(PWSD zh:automatic) / b_long.

blockzh:zh_3893approxzh_4936: zh:eventsegmentblockzh:zh_443notzh:zh_311; zh:zh_1918blockzh:zh_6956**zh:normalzh_6966blockzh:internal** zh:zh_70blockzh:zh_9448
(zh:resamplingzh_2059normalblockperzh:length b zh:zh_2073block, zh:zh_9095withzh:zh_2476eventsegmentblockzh:zh_1576resamplingzh_5414).
b_short = max(segmentzh:zh_8199bitzh_4930, w/Δ);  b_mid = arch PWSD (Politis-White correctedzh:zh_2687 Patton 2009);
b_long = 2·b_mid. zh:zh_598"blockzh:zh_5736notzh:zh_7467": zh:zh_1918 E3-clean Type I ∈ [0.035,0.065] and SE zh:zh_9362pairdifference <25%.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.blocks import build_blocks  # noqa: E402
from common.events import events_from_binary  # noqa: E402


def tier_lengths(g, events, T, w=16, delta=1.0):
    """returns (b_short, b_mid, b_long)."""
    ev_lens = [e - s for s, e in events if e - s >= 10]
    b_short = max(float(np.median(ev_lens)) if ev_lens else w, w / delta)
    try:
        from arch.bootstrap import optimal_block_length
        obl = optimal_block_length(g.reshape(-1, 1))
        b_mid = float(obl.loc["g", "stationary"]) if "g" in set(obl.index.get_level_values(0)) \
            else float(obl.iloc[0, 0])
    except Exception as e:  # noqa: BLE001
        print(f"[block_lengths] PWSD zh:fail({e}), zh:zh_4769 b_mid = b_short")
        b_mid = b_short
    b_long = 2.0 * b_mid
    return dict(b_short=b_short, b_mid=b_mid, b_long=b_long)


def subdivide_normals(blocks, b):
    """zh:zh_1239normalblockperzh:length b zh:zh_4496block (zh:eventblockzh:zh_9362) — zh:zh_7331sensitivezh_140."""
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
    """zh:zh_2864blockzh:zh_5375bit zh:resampling: zh:normalblockzh:zh_386per b zh:zh_5917; zh:eventblockzh:zh_8990blockzh:zh_2338."""
    from common.blocks import resample_blocks
    if b is None:
        return resample_blocks(x, blocks, rng)
    return resample_blocks(x, subdivide_normals(blocks, b), rng)

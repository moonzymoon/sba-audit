"""zh:eventsegmentzh:zh_9231 (paper 15allzh:zh_5348 zh:zh_5604).

zh:eventsegment = zh:zh_6840anomalylabelinterval (start, end), end exclusive. withpaper 6 cadms.eval_utils.events_from_binary
bit-exact zh:etc.zh_3812 (zh:zh_3208testzh:verify), zh:zh_3107vectorzh_2584series.

zh:zh_9806approxzh_2453 (pre-registered): zh:eventsegmentzh:zh_5050block zh:zh_6369bit —— zh:zh_8318resampling (N1/N2/block bootstrap) zh:zh_4657,
samezh:zh_6443segmentzh:zh_3990 windowsamezh:zh_3207samezh:zh_8942, zh:zh_7240segmentzh:zh_6164block.
"""
from __future__ import annotations

import numpy as np


def events_from_binary(b) -> list[tuple[int, int]]:
    """zh:zh_6840 1-segment -> [(start, end), ...], end exclusive. with cadms zh:zh_2687bit-exact zh:etc.zh_3812."""
    b = np.asarray(b, dtype=np.int8)
    if b.ndim != 1:
        raise ValueError(f"labels must be 1-D, got {b.shape}")
    diff = np.diff(np.concatenate(([0], b.view(np.int8), [0])))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def windows_in_event(start: int, end: int, T: int, w: int, stride: int = 1) -> np.ndarray:
    """segment [start,end) zh:zh_3990 zh:zh_5992point (windowzh:zh_1538/labelzh:zh_2874pointzh_1022segmentzh:zh_3990).

    window i zh:zh_8518 [i-w+1, i], labelzh:zh_2874pointis  i (with cmhmil cachezh:zh_4371identical: yw = Yt[w-1:]).
    returnszh:zh_2874pointarray. stride>1 zh:zh_5104segmentzh:zh_3990per stride zh:zh_6701.
    """
    lo = max(start, w - 1)
    hi = min(end, T)
    if hi <= lo:
        return np.empty(0, dtype=np.int64)
    return np.arange(lo, hi, stride, dtype=np.int64)


def seg_anchor_list(events, T, w=16, stride=1, min_len=0):
    """each  zh:eventsegment -> zh:thissegmentwindowzh:zh_2874pointarray. zh:zh_3043 < min_len  zh:zh_5458segment.

    Returns: [(seg_id, start, end, anchors), ...]
    """
    out = []
    for seg_id, (s, e) in enumerate(events):
        if e - s < min_len:
            continue
        out.append((seg_id, s, e, windows_in_event(s, e, T, w, stride)))
    return out

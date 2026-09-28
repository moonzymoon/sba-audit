"""事件段工具 (第15篇全部推断的骨架).

事件段 = 连续异常标签区间 (start, end), end exclusive. 与第6篇 cadms.eval_utils.events_from_binary
逐位等价 (单元测试验证), 但向量化实现以支持长序列.

硬约束 (预注册): 事件段是块的最小单位 —— 任何重采样 (N1/N2/block bootstrap) 中,
同一段内的窗口同进同出, 禁止跨段切块.
"""
from __future__ import annotations

import numpy as np


def events_from_binary(b) -> list[tuple[int, int]]:
    """连续 1-段 -> [(start, end), ...], end exclusive. 与 cadms 版逐位等价."""
    b = np.asarray(b, dtype=np.int8)
    if b.ndim != 1:
        raise ValueError(f"labels must be 1-D, got {b.shape}")
    diff = np.diff(np.concatenate(([0], b.view(np.int8), [0])))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def windows_in_event(start: int, end: int, T: int, w: int, stride: int = 1) -> np.ndarray:
    """段 [start,end) 内的滑窗锚点 (窗口右端/标签锚点落入段内).

    窗口 i 覆盖 [i-w+1, i], 标签锚点为 i (与 cmhmil 缓存口径一致: yw = Yt[w-1:]).
    返回锚点数组. stride>1 时段内按 stride 取.
    """
    lo = max(start, w - 1)
    hi = min(end, T)
    if hi <= lo:
        return np.empty(0, dtype=np.int64)
    return np.arange(lo, hi, stride, dtype=np.int64)


def seg_anchor_list(events, T, w=16, stride=1, min_len=0):
    """每个事件段 -> 该段窗口锚点数组. 过滤原始时长 < min_len 的微段.

    Returns: [(seg_id, start, end, anchors), ...]
    """
    out = []
    for seg_id, (s, e) in enumerate(events):
        if e - s < min_len:
            continue
        out.append((seg_id, s, e, windows_in_event(s, e, T, w, stride)))
    return out

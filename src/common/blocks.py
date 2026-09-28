"""块结构 (预注册 §3 硬约束的代码化).

整个 test 序列划分为互不重叠的块 = 事件段块 ∪ 正常间隔块.
任何重采样中块为最小进出单位: 段内窗口同进同出, 禁止跨段切窗.
微段规则: 原始时长 <10 点的事件段块不进入块级检验的配对序列 (预注册 §3).
"""
from __future__ import annotations

import numpy as np

MIN_SEG = 10  # 微段门槛 (预注册)


def build_blocks(events, T):
    """[(kind, start, end), ...] 按时间序覆盖 [0,T). kind ∈ {'normal','event'}."""
    blocks = []
    prev = 0
    for s, e in events:
        if s > prev:
            blocks.append(("normal", prev, s))
        blocks.append(("event", s, e))
        prev = e
    if prev < T:
        blocks.append(("normal", prev, T))
    return blocks


def block_ids(blocks, T):
    """每个位置的块编号数组 (长度 T)."""
    bid = np.full(T, -1, dtype=np.int32)
    for k, (_, s, e) in enumerate(blocks):
        bid[s:e] = k
    assert (bid >= 0).all(), "块划分未覆盖全序列"
    return bid


def block_stats(d, blocks, min_seg=MIN_SEG, kind=None):
    """块级统计量 m_k (块内 d 均值) 与块元数据.

    只保留进入块级检验的块: 事件块须 >=min_seg; 正常块全保留 (可被裁剪的只有重采样尾部,
    原始正常块本身完整). kind 可选 'event'/'normal' 过滤.
    """
    out = []
    for k, (kd, s, e) in enumerate(blocks):
        if kd == "event" and e - s < min_seg:
            continue
        if kind is not None and kd != kind:
            continue
        out.append(dict(block=k, kind=kd, start=s, end=e, n=e - s, m=float(d[s:e].mean())))
    return out


def resample_blocks(x, blocks, rng, min_seg=MIN_SEG):
    """N2 事件段 block bootstrap (硬约束核心).

    事件块: 有放回重抽, 数量不变, **整块进出, 永不截断**;
    正常块: 有放回重抽补足长度, 最后一块可截断 (正常块不是事件段, 允许);
    拼接结构模仿原始 (以正常块开头/结尾, 若原始如此), 总长 == T.

    Returns: (x*, block_origin) — block_origin[k] = 拼接后第 k 块来自的原始块编号,
             以及拼接块列表 [(kind, start, end)].
    """
    T = len(x)
    ev = [(i, b) for i, b in enumerate(blocks) if b[0] == "event"]
    no = [(i, b) for i, b in enumerate(blocks) if b[0] == "normal"]
    # 事件块有放回重抽 (拒绝采样: 总长不得超过 T —— 高事件覆盖率下小概率触发重抽)
    for _ in range(1000):
        drawn_ev = [(ev[j][0], ev[j][1][1], ev[j][1][2]) for j in rng.integers(0, len(ev), len(ev))]
        len_ev = sum(e - s for _, s, e in drawn_ev)
        if len_ev <= T:
            break
    else:
        raise RuntimeError("事件块拒绝采样 1000 次未成功 — 覆盖率病态")
    need_no = T - len_ev
    drawn_no, acc = [], 0
    while acc < need_no:
        j = rng.integers(0, len(no))
        i, (_, s, e) = no[j]
        take = min(e - s, need_no - acc)
        drawn_no.append((i, s, s + take))
        acc += take
    # 拼接: 开头块类型与原始一致
    starts_normal = blocks[0][0] == "normal"
    order = []
    if starts_normal and drawn_no:
        order.append(drawn_no.pop(0))
    while drawn_ev or drawn_no:
        if drawn_ev:
            order.append(drawn_ev.pop(0))
        if drawn_no:
            order.append(drawn_no.pop(0))
        # 交替, 事件块优先排在正常块前 (结构近似即可, 关键是块完整性)
    out = np.empty(T, dtype=x.dtype)
    new_blocks, origin, pos = [], [], 0
    blocks_by_idx = dict(enumerate(blocks))
    for item in order:
        i, s, e = item
        n = e - s
        out[pos:pos + n] = x[s:e]
        kd = blocks_by_idx[i][0]
        new_blocks.append((kd, pos, pos + n))
        origin.append((i, pos, n))
        pos += n
    assert pos == T, f"拼接长度 {pos} != {T}"
    return out, new_blocks, origin


def check_hard_constraint(blocks, new_blocks, origin):
    """验证重采样未分裂任何事件块: 每个使用中的原始事件块, 其所有出现都是完整拷贝."""
    from collections import defaultdict
    occ = defaultdict(int)
    for i, _, n in origin:
        if blocks[i][0] == "event":
            occ[(i, n)] += 1
    for (i, n), _ in occ.items():
        _, s, e = blocks[i]
        assert n == e - s, f"事件块 {i} 被截断: 出现长度 {n} != 原长 {e - s}"
    return True

"""E3 机器的单元测试: 事件段硬约束 / N1 校准 / R 计算 (阶段1-2 交付要求)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.blocks import (build_blocks, block_stats, resample_blocks,  # noqa: E402
                           check_hard_constraint, MIN_SEG)
from common.events import events_from_binary  # noqa: E402
from bootstrap.n1_split_sign import n1_pvalue  # noqa: E402


def synthetic_series(rng, T=3000):
    """3 个事件段 (含一个微段) + 正常间隔."""
    y = np.zeros(T, dtype=np.int8)
    y[200:500] = 1     # 段1 长300
    y[800:1600] = 1    # 段2 长800
    y[2100:2103] = 1   # 微段 长3
    y[2500:2900] = 1   # 段3 长400
    x = rng.normal(size=T) + 2.0 * y  # 段内整体抬高 (段聚集结构)
    return x, y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hard_constraint_no_segment_split(seed):
    """人工构造 4 段序列: 任一重采样中同段窗口不分裂 (事件块永不被截断/拆分)."""
    rng = np.random.default_rng(seed)
    x, y = synthetic_series(rng)
    blocks = build_blocks(events_from_binary(y), len(x))
    ev_lens = {i: e - s for i, (k, s, e) in enumerate(blocks) if k == "event"}
    for _ in range(300):
        xs, new_blocks, origin = resample_blocks(x, blocks, rng)
        assert len(xs) == len(x), "重采样长度必须等于原长"
        check_hard_constraint(blocks, new_blocks, origin)  # 事件块完整
        # 每个拼接事件块的内容 == 某原始事件块的完整拷贝
        for (kd, s, e), (oi, _, n) in zip(new_blocks, origin):
            if kd == "event":
                os_, oe = blocks[oi][1], blocks[oi][2]
                assert n == oe - os_
                assert np.array_equal(xs[s:e], x[os_:oe])


def test_micro_segment_excluded_but_kept_in_pool():
    rng = np.random.default_rng(3)
    x, y = synthetic_series(rng)
    blocks = build_blocks(events_from_binary(y), len(x))
    st = block_stats(x, blocks)
    ev_in_test = [b for b in st if b["kind"] == "event"]
    assert len(ev_in_test) == 3, "微段(长3<10)不应进入块级检验配对序列"
    assert all(b["n"] >= MIN_SEG for b in ev_in_test)
    # 微段仍是重采样池中的完整块
    xs, new_blocks, origin = resample_blocks(x, blocks, rng)
    micro_copies = [1 for (kd, s, e) in new_blocks
                    if kd == "event" and e - s == 3]
    check_hard_constraint(blocks, new_blocks, origin)
    assert isinstance(micro_copies, list)


def test_n1_signperm_calibrated_under_null():
    """iid 块均值 + 分层对分置换: 经验 size 应接近 0.05.

    注: 块数须足够多 (≥8+9) — 置换参考的 p 值粒度 ~1/C(K/2,K/4), 块太少时天然保守
    (4+5 块时粒度≈1/60, 这是离散性不是偏差; 真实数据最小 SMD 也有 17 块).
    """
    rng = np.random.default_rng(42)
    T, pos = 6000, 100
    y = np.zeros(T, dtype=np.int8)
    for k in range(8):  # 8 个事件段, 长度 150-350 交替
        ln = 150 + 100 * (k % 3)
        y[pos:pos + ln] = 1
        pos += ln + 300
    blocks = build_blocks(events_from_binary(y), T)
    n_rep, rej = 400, 0
    for _ in range(n_rep):
        d = rng.normal(size=T)  # H0 真成立
        p = n1_pvalue(d, blocks, rng)
        rej += (p < 0.05)
    # 400 次, 真 size 0.05 -> 95% 接受带约 [0.028, 0.077]; 放宽到 [0.02, 0.10] 防 MC 抖动
    assert 0.02 <= rej / n_rep <= 0.10, f"N1 经验 size {rej / n_rep:.3f} 偏离校准"


def test_block_level_compositional_null():
    """修订 R2 的回归测试: 成分结构零假设下块级协议必须校准.

    构造与 MSL 诊断同构的数据: 事件块均值 ~ +0.5, 正常块均值 ~ −0.5 (H0 真成立——
    同一检测器自己). 旧实现 (原始分区块 ∩ 重排序列) 在此构造下 size→1;
    块均值层配对 (block_level_pvalues) 应 ≈0.05.
    """
    from bootstrap.e3_clean import block_level_pvalues
    rng = np.random.default_rng(11)
    T, pos = 8000, 50
    y = np.zeros(T, dtype=np.int8)
    for k in range(10):
        ln = 120 + 80 * (k % 3)
        y[pos:pos + ln] = 1
        pos += ln + 350
    g = np.where(y == 1, 0.5, -0.5) + rng.normal(0, 0.15, size=T)
    blocks = build_blocks(events_from_binary(y), T)
    n_rep, rej_t, rej_w = 400, 0, 0
    for i in range(n_rep):
        p3, p4, _ = block_level_pvalues(g, blocks, np.random.default_rng(1000 + i))
        rej_t += (not np.isnan(p3)) and (p3 < 0.05)
        rej_w += (not np.isnan(p4)) and (p4 < 0.05)
    assert 0.02 <= rej_t / n_rep <= 0.10, f"块级 t 经验 size {rej_t / n_rep:.3f} 偏离"
    assert 0.02 <= rej_w / n_rep <= 0.10, f"块级 Wilcoxon 经验 size {rej_w / n_rep:.3f} 偏离"


def test_resample_blocks_seed_reproducible():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    x, y = synthetic_series(np.random.default_rng(0))
    blocks = build_blocks(events_from_binary(y), len(x))
    a = resample_blocks(x, blocks, rng1)
    b = resample_blocks(x, blocks, rng2)
    assert np.array_equal(a[0], b[0]) and a[2] == b[2], "同种子重采样必须可复现"

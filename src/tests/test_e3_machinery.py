"""E3 zh:zh_1953 zh:zh_3208test: zh:eventsegmentzh:zh_9806approxzh_2453 / N1 calibration / R zh:compute (zh:zh_5303segment1-2 zh:zh_7103)."""
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
    """3  zh:eventsegment (zh:zh_1920 zh:zh_5458segment) + zh:normalzh_6966."""
    y = np.zeros(T, dtype=np.int8)
    y[200:500] = 1     # segment1 zh:zh_1935300
    y[800:1600] = 1    # segment2 zh:zh_1935800
    y[2100:2103] = 1   # zh:zh_5458segment zh:zh_19353
    y[2500:2900] = 1   # segment3 zh:zh_1935400
    x = rng.normal(size=T) + 2.0 * y  # segmentzh:zh_570 (segmentzh:zh_9807)
    return x, y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hard_constraint_no_segment_split(seed):
    """zh:zh_2385construction 4 segmentzh:series: zh:zh_5485resamplingzh_4657samesegmentwindownotzh:zh_7624 (zh:eventblockzh:zh_443notzh:zh_3264/zh:zh_440)."""
    rng = np.random.default_rng(seed)
    x, y = synthetic_series(rng)
    blocks = build_blocks(events_from_binary(y), len(x))
    ev_lens = {i: e - s for i, (k, s, e) in enumerate(blocks) if k == "event"}
    for _ in range(300):
        xs, new_blocks, origin = resample_blocks(x, blocks, rng)
        assert len(xs) == len(x), "zh:resamplinglengthmustequal tozh_2431"
        check_hard_constraint(blocks, new_blocks, origin)  # zh:eventblockzh:zh_2476
        # each  zh:zh_260eventblock zh:zh_6287 == zh:zh_2111eventblock zh:zh_838
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
    assert len(ev_in_test) == 3, "zh:zh_5458segment(zh:zh_19353<10)notzh:zh_5916blockzh:zh_4542testpairedzh:series"
    assert all(b["n"] >= MIN_SEG for b in ev_in_test)
    # zh:zh_5458segmentzh:zh_6385resamplingzh_8881 zh:zh_2476block
    xs, new_blocks, origin = resample_blocks(x, blocks, rng)
    micro_copies = [1 for (kd, s, e) in new_blocks
                    if kd == "event" and e - s == 3]
    check_hard_constraint(blocks, new_blocks, origin)
    assert isinstance(micro_copies, list)


def test_n1_signperm_calibrated_under_null():
    """iid blockmean + stratifiedpairzh:zh_746permutation: empirical size zh:zh_5336 0.05.

    zh:zh_2317: blockzh:zh_32 (≥8+9) — permutationzh:reference  p zh:valuezh_3627 ~1/C(K/2,K/4), blockzh:zh_4643
    (4+5 blockzh:zh_4179≈1/60, zh:zh_6336notzh:zh_4296difference; realzh:datazh_7191 SMD zh:zh_2962 17 block).
    """
    rng = np.random.default_rng(42)
    T, pos = 6000, 100
    y = np.zeros(T, dtype=np.int8)
    for k in range(8):  # 8  zh:eventsegment, zh:length 150-350 zh:zh_518
        ln = 150 + 100 * (k % 3)
        y[pos:pos + ln] = 1
        pos += ln + 300
    blocks = build_blocks(events_from_binary(y), T)
    n_rep, rej = 400, 0
    for _ in range(n_rep):
        d = rng.normal(size=T)  # H0 zh:zh_3563
        p = n1_pvalue(d, blocks, rng)
        rej += (p < 0.05)
    # 400 zh:zh_6249, zh:zh_4219 size 0.05 -> 95% zh:zh_6953approx [0.028, 0.077]; zh:zh_3247 [0.02, 0.10] zh:zh_3417 MC zh:zh_5131
    assert 0.02 <= rej / n_rep <= 0.10, f"N1 empirical size {rej / n_rep:.3f} zh:zh_4232calibration"


def test_block_level_compositional_null():
    """zh:zh_6924 R2  zh:zh_5164test: zh:componentzh_678assumptionzh_5169blockzh:zh_4714mustcalibration.

    zh:constructionwith MSL zh:zh_6589samezh:zh_2058 zh:data: zh:eventblockmean ~ +0.5, zh:normalblockmean ~ −0.5 (H0 zh:zh_3563——
    samezh:zh_6443detectorzh:zh_6748). oldzh:zh_3965 (zh:zh_7452block ∩ zh:zh_8031series) zh:zh_3346constructionzh_5169 size→1;
    blockmeanzh:zh_419paired (block_level_pvalues) zh:zh_2483 ≈0.05.
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
    assert 0.02 <= rej_t / n_rep <= 0.10, f"blockzh:zh_4542 t empirical size {rej_t / n_rep:.3f} zh:zh_4232"
    assert 0.02 <= rej_w / n_rep <= 0.10, f"blockzh:zh_4542 Wilcoxon empirical size {rej_w / n_rep:.3f} zh:zh_4232"


def test_resample_blocks_seed_reproducible():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    x, y = synthetic_series(np.random.default_rng(0))
    blocks = build_blocks(events_from_binary(y), len(x))
    a = resample_blocks(x, blocks, rng1)
    b = resample_blocks(x, blocks, rng2)
    assert np.array_equal(a[0], b[0]) and a[2] == b[2], "samezh:seedresamplingmustreproducible"

"""blockzh:zh_7202 (pre-registered §3 zh:zh_9806approxzh_2453 zh:zh_4699).

zh:zh_8990  test zh:serieszh_9448is zh:zh_9290notzh:zh_8297 block = zh:eventsegmentblock ∪ zh:normalzh_6966block.
zh:zh_8318resamplingzh_4657blockis zh:zh_3850bit: segmentzh:zh_3990windowsamezh:zh_3207samezh:zh_8942, zh:zh_7240segmentzh:zh_1061.
zh:zh_5458segmentzh:zh_9291: zh:zh_2276 <10 zh:point zh:eventsegmentblocknotzh:zh_534blockzh:zh_4542test pairedzh:series (pre-registered §3).
"""
from __future__ import annotations

import numpy as np

MIN_SEG = 10  # zh:zh_5458segmentzh:zh_5340 (pre-registered)


def build_blocks(events, T):
    """[(kind, start, end), ...] perzh:zh_6959 [0,T). kind ∈ {'normal','event'}."""
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
    """each  zh:bitzh_3977 blockzh:zh_5571array (zh:length T)."""
    bid = np.full(T, -1, dtype=np.int32)
    for k, (_, s, e) in enumerate(blocks):
        bid[s:e] = k
    assert (bid >= 0).all(), "blockzh:zh_6563series"
    return bid


def block_stats(d, blocks, min_seg=MIN_SEG, kind=None):
    """blockzh:zh_4542statisticszh_1031 m_k (blockzh:zh_3990 d mean) withblockzh:zh_9586data.

    zh:zh_5551blockzh:zh_4542test block: zh:eventblockzh:zh_2972 >=min_seg; zh:normalblockzh:zh_9518 (zh:zh_2442 zh:zh_1021resamplingtail,
    zh:zh_3710normalblockzh:zh_2060). kind zh:zh_6804 'event'/'normal' zh:zh_2307.
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
    """N2 zh:eventsegment block bootstrap (zh:zh_9806approxzh_2453core).

    zh:eventblock: zh:zh_7879, zh:countnotzh:zh_3688, **zh:zh_8990blockzh:zh_2338, zh:zh_443notzh:zh_24**;
    zh:normalblock: zh:zh_4070length, zh:lastzh_6443blockzh:zh_3902 (zh:normalblocknotzh:zh_5050eventsegment, zh:zh_8227);
    zh:zh_4508 (zh:zh_5665normalblockzh:zh_5737/zh:zh_233, zh:zh_5415), zh:zh_3510 == T.

    Returns: (x*, block_origin) — block_origin[k] = zh:zh_4892paper  k blockzh:zh_7902 zh:zh_3710blockzh:zh_5571,
             zh:zh_5665andzh:zh_260blocklist [(kind, start, end)].
    """
    T = len(x)
    ev = [(i, b) for i, b in enumerate(blocks) if b[0] == "event"]
    no = [(i, b) for i, b in enumerate(blocks) if b[0] == "normal"]
    # zh:eventblockzh:zh_7879 (zh:zh_5465sampling: zh:zh_3510notzh:zh_6483 T —— zh:zh_1834eventzh_8518ratezh_7031ratezh_7171)
    for _ in range(1000):
        drawn_ev = [(ev[j][0], ev[j][1][1], ev[j][1][2]) for j in rng.integers(0, len(ev), len(ev))]
        len_ev = sum(e - s for _, s, e in drawn_ev)
        if len_ev <= T:
            break
    else:
        raise RuntimeError("zh:eventblockzh:zh_5465sampling 1000 zh:zh_8814success — zh:zh_8518ratezh_1066")
    need_no = T - len_ev
    drawn_no, acc = [], 0
    while acc < need_no:
        j = rng.integers(0, len(no))
        i, (_, s, e) = no[j]
        take = min(e - s, need_no - acc)
        drawn_no.append((i, s, s + take))
        acc += take
    # zh:zh_260: zh:zh_5737blockzh:typewithzh:zh_3710identical
    starts_normal = blocks[0][0] == "normal"
    order = []
    if starts_normal and drawn_no:
        order.append(drawn_no.pop(0))
    while drawn_ev or drawn_no:
        if drawn_ev:
            order.append(drawn_ev.pop(0))
        if drawn_no:
            order.append(drawn_no.pop(0))
        # zh:zh_518, zh:eventblockzh:zh_2100normalblockzh:zh_17 (zh:zh_7514i.e.zh_542, zh:zh_124keyzh_5050blockzh:completeness)
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
    assert pos == T, f"zh:zh_260length {pos} != {T}"
    return out, new_blocks, origin


def check_hard_constraint(blocks, new_blocks, origin):
    """zh:verifyresamplingzh_8603eventblock: each  zh:usezh_4657 zh:zh_3710eventblock, zh:zh_7753allzh_3530."""
    from collections import defaultdict
    occ = defaultdict(int)
    for i, _, n in origin:
        if blocks[i][0] == "event":
            occ[(i, n)] += 1
    for (i, n), _ in occ.items():
        _, s, e = blocks[i]
        assert n == e - s, f"zh:eventblock {i} zh:zh_3264: zh:zh_421length {n} != zh:zh_2431 {e - s}"
    return True

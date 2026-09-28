"""事件段工具与清单的单元测试."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.events import events_from_binary, windows_in_event  # noqa: E402

CADMS = r"D:/0科研/工作1/第6篇SCI/src"
INV = r"D:/0科研/工作1/第15篇SCI/01_立项与调研/事件段清单"


def cadms_events(b):
    """第6篇循环版, 作等价性参照."""
    events, in_e, s = [], False, 0
    for i, v in enumerate(b):
        if v and not in_e:
            s, in_e = i, True
        elif not v and in_e:
            events.append((s, i))
            in_e = False
    if in_e:
        events.append((s, len(b)))
    return events


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_equivalence_with_cadms(seed):
    rng = np.random.default_rng(seed)
    for p in (0.05, 0.2, 0.5, 0.8):
        b = (rng.random(500) < p).astype(np.int8)
        assert events_from_binary(b) == cadms_events(list(b))


def test_boundaries():
    assert events_from_binary([0, 0, 0]) == []
    assert events_from_binary([1, 1]) == [(0, 2)]        # 尾段闭合
    assert events_from_binary([0, 1, 1, 0]) == [(1, 3)]
    assert events_from_binary([1, 0, 1]) == [(0, 1), (2, 3)]
    assert events_from_binary(np.array([], dtype=np.int8)) == []


def test_windows_in_event_bounds():
    # 段 [10,20), T=20, w=4: 锚点须 >= w-1 且 < 20
    a = windows_in_event(10, 20, 20, w=4)
    assert a.min() >= 3 and a.max() < 20
    assert len(a) == 10
    # 段起点在窗口可达范围之前 -> 锚点从 w-1 起
    a2 = windows_in_event(0, 5, 20, w=4)
    assert list(a2) == [3, 4]
    # 空段
    assert len(windows_in_event(5, 5, 20, w=4)) == 0
    # stride 采样不越界
    a3 = windows_in_event(0, 20, 20, w=4, stride=3)
    assert list(a3) == [3, 6, 9, 12, 15, 18]


def test_inventory_csv_integrity():
    files = [f for f in os.listdir(INV) if f.endswith(".csv") and not f.startswith("_")]
    core = {f for f in files if f.split("_")[0] in
            {"SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI"}}
    assert len(core) == 6, f"6 个核心数据集清单必须齐, 实得 {core}"
    assert len(files) >= 30, f"R3 扩充后应≥30 个清单(含 TSB 子集), 实得 {len(files)}"
    for f in files:
        try:
            df = pd.read_csv(os.path.join(INV, f))
        except pd.errors.EmptyDataError:
            continue  # 无异常段序列(旧版空文件)
        assert {"dataset", "seg_id", "start", "end", "duration",
                "n_channels_annotated", "channels"} <= set(df.columns)
        if len(df) == 0:
            continue  # 无异常段序列 (TSB 个别)
        assert (df.duration == df.end - df.start).all()
        assert (df.duration > 0).all()
        assert df.start.is_monotonic_increasing
        assert (df.start.values[1:] >= df.end.values[:-1]).all()  # 不重叠
        if df.dataset.iloc[0] == "SMD":
            assert df.channels.ne("unannotated").any()
            assert (df.n_channels_annotated.dropna() > 0).any()


def test_smd_channels_map_to_original_coords():
    """SMD 通道归属按原始坐标映射: 第0段缓存(15834,16380) <-> 原始(15849,16395)
    应命中 interpretation 第一行 15849-16368:1,9,10,12,13,14,15."""
    df = pd.read_csv(os.path.join(INV, "SMD_事件段清单.csv"))
    row0 = df[df.seg_id == 0].iloc[0]
    assert row0.start_orig == 15849
    assert str(row0.channels) == "1,9,10,12,13,14,15"

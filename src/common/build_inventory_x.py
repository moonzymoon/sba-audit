"""新数据集事件段清单构建 (loader 标签模式, R3 扩充).

TE / MetroPT3 / BATADAL / NEweather / 5ECK2022 / TSB-AD-MV 分层子集(~26条):
直接用 loader 的 Y -> 35/15/50 切分 -> test 标签 -> 事件段 (与打分缓存口径一致,
TE/MetroPT3 已对既有缓存逐位验证).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.events import events_from_binary  # noqa: E402
from scorers_extra import (load_any, load_tsb, tsb_select, _split, W)  # noqa: E402

OUT_DIR = r"D:/0科研/工作1/第15篇SCI/01_立项与调研/事件段清单"


def test_labels(Y):
    _, n2, _ = _split(len(Y))
    return Y[n2:][W - 1:]


def build(name, Y, src):
    y = test_labels(Y)
    ev = events_from_binary(y)
    rows = [dict(dataset=name, seg_id=k, start=s, end=e, duration=e - s,
                 start_orig=s + W - 1, end_orig=e + W - 1,
                 n_channels_annotated=np.nan, channels="aggregate", label_source=src,
                 note="R3扩充") for k, (s, e) in enumerate(ev)]
    return pd.DataFrame(rows), len(y)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    jobs = [("TE", "Rieth扩展TE(8h注入约定,自包含实现已对缓存逐位验证)"),
            ("MetroPT3", "MetroPT3(气漏事件窗,ds=5,已对缓存逐位验证)"),
            ("BATADAL", "BATADAL_training2"),
            ("NEweather", "NEweather data+class")]
    for name, src in jobs:
        _, Y = load_any(name)
        df, T = build(name, Y, src)
        df.reindex(columns=["dataset","seg_id","start","end","duration","start_orig","end_orig","n_channels_annotated","channels","label_source","note"]).to_csv(os.path.join(OUT_DIR, f"{name}_事件段清单.csv"), index=False, encoding="utf-8-sig")
        big = df[df["duration"] >= 10] if len(df) else df
        summary.append(dict(dataset=name, T_test=T, n_segments=len(df), n_ge10=len(big),
                            median_dur=big.duration.median() if len(big) else np.nan,
                            anom_ratio=round(df.duration.sum() / T, 4)))
        print(f"{name}: T={T} 段数={len(df)} (>=10: {len(big)})")
    # TSB 子集
    tsb_rows = []
    for idx, fname in tsb_select():
        _, Y = load_tsb(idx)
        name = f"TSB{idx:03d}"
        df, T = build(name, Y, f"TSB-AD-MV/{fname}")
        df.reindex(columns=["dataset","seg_id","start","end","duration","start_orig","end_orig","n_channels_annotated","channels","label_source","note"]).to_csv(os.path.join(OUT_DIR, f"{name}_事件段清单.csv"), index=False, encoding="utf-8-sig")
        big = df[df["duration"] >= 10] if len(df) else df
        summary.append(dict(dataset=name, T_test=T, n_segments=len(df), n_ge10=len(big),
                            median_dur=big.duration.median() if len(big) else np.nan,
                            anom_ratio=round(df["duration"].sum() / max(T, 1), 4) if len(df) else 0.0))
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT_DIR, "_R3扩充汇总.csv"), index=False, encoding="utf-8-sig")
    print(sm.to_string(index=False))
    print(f"TSB 子集: {len(tsb_select())} 条; 总数据集覆盖 = 6 核心 + 5 工业 + {len(tsb_select())} TSB")


if __name__ == "__main__":
    main()

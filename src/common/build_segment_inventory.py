"""事件段清单构建 (阶段0-4, 全部推断的骨架).

每数据集输出 CSV: 段ID/起止(缓存坐标+原始test坐标)/时长/涉及channel/标注来源.

坐标口径:
- 缓存坐标: cmhmil_{ds}_seed7.npz 的 labels 索引 (本文全部推断实际使用的序列);
  它 = 35/15/50 时序切分的 test 段, 再前移 w-1=15 (滑窗修剪, 标签锚点=窗口右端).
- 原始坐标: 未修剪的 test 段坐标 (与 SMD interpretation_label / iTrust 攻击表对齐用).
- 验证: 用第2篇 loaders+chronological_split 重建 test 标签, 校验 缓存labels == testY[15:].
  (SWaT/WADI loader 解析大 CSV 较慢, 允许 --no-verify 跳过并在 notes 标注.)

channel 归属:
- SMD: interpretation_label/machine-1-1.txt 逐段根因通道 (1-based), 按原始坐标重叠映射;
- MSL/SMAP: 本仓库 NASA AllInOne 已合并为单流且 info.json 为空 -> aggregate (逐通道区间不可复原);
- PSM/SWaT/WADI: 标签即整体流标签 -> aggregate.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.events import events_from_binary  # noqa: E402

CACHE = r"D:/0科研/工作1/第6篇SCI/src/_score_cache"
DS_ROOT = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/datasets"
OUT_DIR = r"D:/0科研/工作1/第15篇SCI/01_立项与调研/事件段清单"
W_TRIM = 15  # cmhmil 缓存标签相对原始 test 起点的前移量 (w-1)

DATASETS = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI"]


def cached_labels(ds):
    d = np.load(os.path.join(CACHE, f"cmhmil_{ds}_seed7.npz"))
    return d["labels"].astype(np.int8), d["scores"].shape[0]


def parse_smd_interpretation(path):
    """'15849-16368:1,9,10' -> [(s,e,[ch...]), ...] (原始坐标, 1-based 通道号)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rng, chs = line.split(":")
            s, e = (int(v) for v in rng.split("-"))
            out.append((s, e, [int(c) for c in chs.split(",") if c]))
    return out


def rebuild_test_labels(ds):
    """第2篇 loader 重建 (train+test 全序列标签, test 起点偏移). 返回 (Y_all, n_before_test)."""
    sys.path.insert(0, r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/src")
    from data.loaders import load_dataset
    from data.split import chronological_split
    _, Y = load_dataset(ds)
    sp = chronological_split(len(Y), 0.35, 0.15)
    return Y[sp.test], int(sp.test.start)


def build(ds, verify=True):
    y, T = cached_labels(ds)
    events = events_from_binary(y)
    rows = []

    interp = None
    if ds == "SMD":
        interp = parse_smd_interpretation(
            os.path.join(DS_ROOT, "SMD", "interpretation_label", "machine-1-1.txt"))

    # 验证坐标锚定
    note_verify = "verified"
    if verify:
        try:
            y_test, test_start = rebuild_test_labels(ds)
            if len(y_test) - W_TRIM != T or not np.array_equal(y_test[W_TRIM:], y):
                raise ValueError("缓存标签与 loader 重建不一致")
        except Exception as e:  # noqa: BLE001
            note_verify = f"verify-failed({e})"
    else:
        note_verify = "not-verified(loader慢, 跳过)"

    for seg_id, (s, e) in enumerate(events):
        s_orig, e_orig = s + W_TRIM, e + W_TRIM
        if interp is not None:
            chs = []
            for is_, ie, ic in interp:
                if min(e_orig, ie + 1) > max(s_orig, is_):  # 原始坐标重叠
                    chs.extend(c for c in ic if c not in chs)
            ch_str = ",".join(str(c) for c in sorted(chs)) if chs else "unannotated"
            n_ch, src = (len(chs), "SMD interpretation_label")
        else:
            ch_str, n_ch, src = ("aggregate", np.nan, "整体流标签(无逐通道标注)")
        rows.append(dict(
            dataset=ds, seg_id=seg_id, start=s, end=e, duration=e - s,
            start_orig=s_orig, end_orig=e_orig,
            n_channels_annotated=n_ch, channels=ch_str, label_source=src,
            note=note_verify))
    return pd.DataFrame(rows), T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    for ds in args.datasets:
        df, T = build(ds, verify=not args.no_verify)
        out = os.path.join(OUT_DIR, f"{ds}_事件段清单.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        big = df[df.duration >= 10]
        summary.append(dict(dataset=ds, T_test=T, n_segments=len(df),
                            n_segments_ge10=len(big),
                            median_dur=big.duration.median() if len(big) else np.nan,
                            max_dur=big.duration.max() if len(big) else np.nan,
                            anom_ratio=round(float(y_sum(df, T)), 4)))
        print(f"{ds}: T={T}, 段数={len(df)} (>=10点: {len(big)}), "
              f"段长中位数={summary[-1]['median_dur']}, 已写出 {out}")
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT_DIR, "_汇总.csv"), index=False, encoding="utf-8-sig")
    print(sm.to_string(index=False))


def y_sum(df, T):
    return df.duration.sum() / T


if __name__ == "__main__":
    main()

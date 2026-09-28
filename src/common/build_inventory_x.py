"""newdatasetzh:eventsegmentzh:zh_6725 (loader labelzh:zh_2480, R3 extended).

TE / MetroPT3 / BATADAL / NEweather / 5ECK2022 / TSB-AD-MV stratifiedsubset(~26rows):
zh:directzh_8215 loader   Y -> 35/15/50 zh:zh_311 -> test label -> zh:eventsegment (withzh:zh_33cachezh:zh_4371identical,
TE/MetroPT3 pairzh:zh_3974cachebit-exact zh:verify).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.events import events_from_binary  # noqa: E402
from scorers_extra import (load_any, load_tsb, tsb_select, _split, W)  # noqa: E402

OUT_DIR = r"D:/0科研/工作1/第15篇SCI/01_zh:zh_8028withzh:zh_9766/zh:eventsegmentzh:zh_7558"


def test_labels(Y):
    _, n2, _ = _split(len(Y))
    return Y[n2:][W - 1:]


def build(name, Y, src):
    y = test_labels(Y)
    ev = events_from_binary(y)
    rows = [dict(dataset=name, seg_id=k, start=s, end=e, duration=e - s,
                 start_orig=s + W - 1, end_orig=e + W - 1,
                 n_channels_annotated=np.nan, channels="aggregate", label_source=src,
                 note="R3extended") for k, (s, e) in enumerate(ev)]
    return pd.DataFrame(rows), len(y)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    jobs = [("TE", "Riethzh:zh_5666TE(8hinjectedzh:approxzh_6266,zh:zh_8847includezh_3965paircachebit-exact zh:verify)"),
            ("MetroPT3", "MetroPT3(zh:zh_2819eventzh_4076,ds=5,paircachebit-exact zh:verify)"),
            ("BATADAL", "BATADAL_training2"),
            ("NEweather", "NEweather data+class")]
    for name, src in jobs:
        _, Y = load_any(name)
        df, T = build(name, Y, src)
        df.reindex(columns=["dataset","seg_id","start","end","duration","start_orig","end_orig","n_channels_annotated","channels","label_source","note"]).to_csv(os.path.join(OUT_DIR, f"{name}_zh:eventsegmentzh:zh_7558.csv"), index=False, encoding="utf-8-sig")
        big = df[df["duration"] >= 10] if len(df) else df
        summary.append(dict(dataset=name, T_test=T, n_segments=len(df), n_ge10=len(big),
                            median_dur=big.duration.median() if len(big) else np.nan,
                            anom_ratio=round(df.duration.sum() / T, 4)))
        print(f"{name}: T={T} segmentzh:zh_4930={len(df)} (>=10: {len(big)})")
    # TSB subset
    tsb_rows = []
    for idx, fname in tsb_select():
        _, Y = load_tsb(idx)
        name = f"TSB{idx:03d}"
        df, T = build(name, Y, f"TSB-AD-MV/{fname}")
        df.reindex(columns=["dataset","seg_id","start","end","duration","start_orig","end_orig","n_channels_annotated","channels","label_source","note"]).to_csv(os.path.join(OUT_DIR, f"{name}_zh:eventsegmentzh:zh_7558.csv"), index=False, encoding="utf-8-sig")
        big = df[df["duration"] >= 10] if len(df) else df
        summary.append(dict(dataset=name, T_test=T, n_segments=len(df), n_ge10=len(big),
                            median_dur=big.duration.median() if len(big) else np.nan,
                            anom_ratio=round(df["duration"].sum() / max(T, 1), 4) if len(df) else 0.0))
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT_DIR, "_R3extendedzh:zh_5435.csv"), index=False, encoding="utf-8-sig")
    print(sm.to_string(index=False))
    print(f"TSB subset: {len(tsb_select())} rows; zh:zh_8811datasetzh:zh_8518 = 6 zh:core + 5 zh:zh_7334 + {len(tsb_select())} TSB")


if __name__ == "__main__":
    main()

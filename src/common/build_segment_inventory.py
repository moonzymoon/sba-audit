"""zh:eventsegmentzh:zh_6725 (zh:zh_5303segment0-4, allzh:zh_5348 zh:zh_5604).

each datasetwrites CSV: segmentID/zh:zh_1435(cachezh:axis+zh:zh_3710testzh:axis)/zh:zh_7211/zh:zh_1996andchannel/zh:zh_663source.

zh:axiszh_4371:
- cachezh:axis: cmhmil_{ds}_seed7.npz   labels index (zh:zh_136allzh:zh_2310use zh:series);
  zh:zh_8388 = 35/15/50 chronological split  test segment, zh:zh_9603 w-1=15 (zh:zh_5328, labelzh:zh_2874point=windowzh:zh_1538).
- zh:zh_3710axis: zh:zh_1257  test segmentzh:axis (with SMD interpretation_label / iTrust zh:zh_4879tablepairzh:zh_6385).
- zh:verify: zh:zh_8215paper 2 loaders+chronological_split zh:zh_335 test label, zh:zh_8488 cachelabels == testY[15:].
  (SWaT/WADI loader zh:parsezh_916 CSV zh:zh_250, zh:zh_8227 --no-verify skipzh:zh_4845 notes zh:zh_8044.)

channel zh:zh_3121:
- SMD: interpretation_label/machine-1-1.txt zh:zh_2843segmentzh:zh_671 (1-based), perzh:zh_3710axiszh_5188;
- MSL/SMAP: zh:zh_4251 NASA AllInOne zh:zh_7289is zh:zh_8759and info.json is zh:zh_6162 -> aggregate (zh:zh_4389intervalnotzh:zh_8356);
- PSM/SWaT/WADI: labelzh:i.e.zh_4348label -> aggregate.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.events import events_from_binary  # noqa: E402

CACHE = r"D:/0keyan/gongzuo1/paper 6SCI/src/_score_cache"
DS_ROOT = r"D:/0keyan/gongzuo1/paper 2SCI/Contrastive_TopK_MIL/datasets"
OUT_DIR = r"D:/0keyan/gongzuo1/paper 15SCI/01_zh:zh_8028withzh:zh_9766/zh:eventsegmentzh:zh_7558"
W_TRIM = 15  # cmhmil cachelabelzh:zh_9362pairzh:zh_3710 test zh:zh_8664point zh:zh_3576 (w-1)

DATASETS = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI"]


def cached_labels(ds):
    d = np.load(os.path.join(CACHE, f"cmhmil_{ds}_seed7.npz"))
    return d["labels"].astype(np.int8), d["scores"].shape[0]


def parse_smd_interpretation(path):
    """'15849-16368:1,9,10' -> [(s,e,[ch...]), ...] (zh:zh_3710axis, 1-based zh:zh_5220)."""
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
    """paper 2 loader zh:zh_335 (train+test zh:zh_9405serieslabel, test zh:zh_8664pointzh_853). returns (Y_all, n_before_test)."""
    sys.path.insert(0, r"D:/0keyan/gongzuo1/paper 2SCI/Contrastive_TopK_MIL/src")
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

    # zh:verifyaxiszh_970
    note_verify = "verified"
    if verify:
        try:
            y_test, test_start = rebuild_test_labels(ds)
            if len(y_test) - W_TRIM != T or not np.array_equal(y_test[W_TRIM:], y):
                raise ValueError("cachelabelwith loader zh:zh_335notidentical")
        except Exception as e:  # noqa: BLE001
            note_verify = f"verify-failed({e})"
    else:
        note_verify = "not-verified(loaderzh:zh_424, skip)"

    for seg_id, (s, e) in enumerate(events):
        s_orig, e_orig = s + W_TRIM, e + W_TRIM
        if interp is not None:
            chs = []
            for is_, ie, ic in interp:
                if min(e_orig, ie + 1) > max(s_orig, is_):  # zh:zh_3710axiszh_8297
                    chs.extend(c for c in ic if c not in chs)
            ch_str = ",".join(str(c) for c in sorted(chs)) if chs else "unannotated"
            n_ch, src = (len(chs), "SMD interpretation_label")
        else:
            ch_str, n_ch, src = ("aggregate", np.nan, "zh:zh_4348label(no zh:zh_2613)")
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
        out = os.path.join(OUT_DIR, f"{ds}_zh:eventsegmentzh:zh_7558.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        big = df[df.duration >= 10]
        summary.append(dict(dataset=ds, T_test=T, n_segments=len(df),
                            n_segments_ge10=len(big),
                            median_dur=big.duration.median() if len(big) else np.nan,
                            max_dur=big.duration.max() if len(big) else np.nan,
                            anom_ratio=round(float(y_sum(df, T)), 4)))
        print(f"{ds}: T={T}, segmentzh:zh_4930={len(df)} (>=10zh:point: {len(big)}), "
              f"segmentzh:zh_8199bitzh_4930={summary[-1]['median_dur']}, written {out}")
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT_DIR, "_zh:zh_5435.csv"), index=False, encoding="utf-8-sig")
    print(sm.to_string(index=False))


def y_sum(df, T):
    return df.duration.sum() / T


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""R24 stage 2+3: E3-clean Type I on 5 EMG cells (B=10,000) + direct floors
(10 shallow pairs, B=400) + segment inventory. Writes r24_emg_summary.json."""
import json, sys, os, itertools
import numpy as np

sys.path.insert(0, r"D:\0keyan\gongzuo1\paper 15SCI\src")
from bootstrap.e3_clean import run, load_goodness  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from common.blocks import build_blocks  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

RES = r"D:\0keyan\gongzuo1\paper 15SCI\02_shiyanjilu\results"
SC = ["iforest", "pca", "gmm", "ocsvm", "lof"]

# segment inventory on the cached label stream (test half, as the suite uses)
_, y = load_goodness("iforest", "EMG")
T = len(y)
blocks = build_blocks(events_from_binary(y.astype(np.int8)), T)
ev = [e - s for k, s, e in blocks if k == "event"]
no = [e - s for k, s, e in blocks if k == "normal"]
K = sum(1 for x in ev if x >= 10) + len(no)
inv = {"T_test": T, "pi": round(float(y.mean()), 4), "n_event": len(ev),
       "n_event_qual": sum(1 for x in ev if x >= 10), "n_normal": len(no), "K": K,
       "ev_len_med": int(np.median(ev)), "ev_len_max": max(ev)}
print("inventory:", inv)

out = {"inventory": inv, "e3": [], "floors": []}
for sc in SC:
    r = run("EMG", sc, 10000)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()
           if k.startswith("size_") or k in ("dataset", "scorer", "T", "n_blocks")},
          flush=True)
    out["e3"].append(r)

# direct floors: 10 pairs, B=400, paired rng (direct_scale_verdict conventions)
from common.blocks import resample_blocks  # noqa: E402
B = 400
rng = np.random.default_rng(20260910)
gs = {}
y = None
for sc in SC:
    g, y_ = load_goodness(sc, "EMG")
    y = y_.astype(float)
    gs[sc] = g
blocks = build_blocks(events_from_binary(y.astype(np.int8)), T)
ud = {sc: np.empty(B) for sc in SC}
for b in range(B):
    # save state ONCE per replicate so all scorers AND labels share the same block indices
    # (R40 shared-block-indices convention; matches direct_scale_verdict.py)
    state = rng.bit_generator.state
    for sc in SC:
        rng.bit_generator.state = state
        g_star, _, _ = resample_blocks(gs[sc], blocks, rng)
        rng.bit_generator.state = state
        ys, _, _ = resample_blocks(y, blocks, rng)
        ud[sc][b] = roc_auc_score(ys.astype(int), np.abs(g_star))
aucs = {sc: float(roc_auc_score(y.astype(int), np.abs(gs[sc]))) for sc in SC}
pi = float(y.mean())
from scipy import stats as st
for a, b2 in itertools.combinations(SC, 2):
    D = ud[a] - ud[b2]
    sd = D.std(ddof=1)
    mde = 2.8 * sd * 100
    gap = abs(aucs[a] - aucs[b2]) * 100
    out["floors"].append(dict(bm="EMG", pair=f"{a}-{b2}", B=B,
                              sd_direct_pts=round(float(sd * 100), 2),
                              mde_direct_pts=round(float(mde), 1),
                              gap_pts=round(gap, 1),
                              aucs=round(aucs[a], 3), aucs_b=round(aucs[b2], 3),
                              certifiable=bool(gap >= mde)))
    print(out["floors"][-1], flush=True)

json.dump(out, open(os.path.join(RES, "r24_emg_summary.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print("saved r24_emg_summary.json")

# -*- coding: utf-8 -*-
"""R30-2: end-to-end cross-seed null for a SECOND (shallow) family: isolation
forest re-fit with 10 random states per canonical suite (same frozen protocol
as scorers_extra iforest: fit calib windows, score test, right-end labels),
then all C(10,2)=45 pairs per suite -> window-level paired t and segment-level
t rejection rates, alongside the deep detector's 72%."""
import os, sys, json, itertools
import numpy as np
from scipy import stats as st
from sklearn.ensemble import IsolationForest

sys.path.insert(0, r"D:\0科研\工作1\第15篇SCI\src")
sys.path.insert(0, r"D:\0科研\工作1\第6篇SCI\src")
from cadms.scorers import get_scorer  # for labels alignment via iforest cache
from common.events import events_from_binary  # noqa: E402
from common.blocks import build_blocks, block_stats  # noqa: E402

W = 16
SEEDS = list(range(10))
BMS = ["SMD", "PSM", "MSL", "SMAP", "WADI"]
RES = r"D:\0科研\工作1\第15篇SCI\02_实验记录\results"

def win(A):
    return np.lib.stride_tricks.sliding_window_view(A, W, axis=0).reshape(-1, A.shape[1] * W)

out = []
for ds in BMS:
    # load raw series via paper-2 loader (same path as suite protocol)
    sys.path.insert(0, r"D:\0科研\工作1\第2篇SCI\Contrastive_TopK_MIL\src")
    from data.loaders import load_dataset
    X, Y = load_dataset(ds)
    T = len(X)
    n1, n2 = int(T * 0.35), int(T * 0.50)
    mu, sd = X[:n1].mean(0), X[:n1].std(0) + 1e-8
    Xs = ((X - mu) / sd).astype(np.float32)
    F_cal = win(Xs[n1:n2])
    Xt, Yt = Xs[n2:], Y[n2:]
    F_test = win(Xt)
    y_test = Yt[W - 1:]
    scores = {}
    for s in SEEDS:
        ifo = IsolationForest(n_estimators=100, contamination="auto",
                              random_state=s, n_jobs=-1).fit(F_cal)
        scores[s] = -ifo.decision_function(F_test)
    from sklearn.metrics import roc_auc_score
    aucs = {s: float(roc_auc_score(y_test, scores[s])) for s in SEEDS}
    blocks = build_blocks(events_from_binary(y_test.astype(np.int8)), len(y_test))
    rej_w = rej_b = tot = 0
    for a, b in itertools.combinations(SEEDS, 2):
        d = scores[a] - scores[b]
        if st.ttest_1samp(d, 0.0).pvalue < .05:
            rej_w += 1
        marr = np.array([bb["m"] for bb in block_stats(d, blocks, min_seg=10)])
        if len(marr) >= 3 and st.ttest_1samp(marr, 0.0).pvalue < .05:
            rej_b += 1
        tot += 1
    rec = dict(dataset=ds, family="iforest", n_runs=len(SEEDS), n_pairs=tot,
               win_rej=rej_w, win_rate=round(rej_w / tot, 3),
               blk_rej=rej_b, blk_rate=round(rej_b / tot, 3),
               auroc_spread_pts=round(100 * (max(aucs.values()) - min(aucs.values())), 2))
    out.append(rec)
    print(rec, flush=True)

json.dump(out, open(os.path.join(RES, "r30_iforest_e2e.json"), "w"), indent=1)
print("saved r30_iforest_e2e.json")

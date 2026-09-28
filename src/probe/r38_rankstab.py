# -*- coding: utf-8 -*-
"""R38: rank stability analysis (suite-level + seed-level bootstrap).
Outputs results/r38_rankstab.json."""
import os, sys, json
import numpy as np
from scipy import stats as st
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"D:/0科研/工作1/第15篇SCI/src/mixed")
from e1_full import build_table  # noqa: E402

CACHE = r"D:/0科研/工作1/第6篇SCI/src/_score_cache"
RES = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results"
CORE6 = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI"]
DET6 = ["cmhmil", "iforest", "pca", "gmm", "ocsvm", "lof"]
CMH_SEEDS = {"SMD": [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024],
             "PSM": [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024],
             "SWaT": [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024],
             "MSL": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "SMAP": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "WADI": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

_auc_cache = {}
def auc(ds, s):
    key = (ds, s)
    if key not in _auc_cache:
        d = np.load(os.path.join(CACHE, f"cmhmil_{ds}_seed{s}.npz"))
        _auc_cache[key] = float(roc_auc_score(d["labels"], d["scores"]))
    return _auc_cache[key]

df = build_table()
d2 = df.copy()
cmh = df[df.detector == "cmhmil"]
d2.loc[d2.detector == "cmhmil", "AUROC"] = cmh.groupby("dataset")["AUROC"].transform("mean")
d2 = d2.drop_duplicates(["dataset", "detector"])
sub = d2[d2.dataset.isin(CORE6) & d2.detector.isin(DET6)]
piv = sub.pivot_table(index="dataset", columns="detector", values="AUROC").loc[CORE6, DET6]
X = piv.values
assert not np.isnan(X).any()

base_rank = st.rankdata(-X.mean(axis=0))
rng = np.random.default_rng(20260925)
B = 1500

res_a = {}
for m in range(1, 7):
    taus = np.empty(B)
    top1 = {d: 0 for d in DET6}
    for b in range(B):
        pick = rng.integers(0, 6, m)
        means = X[np.ix_(pick, range(6))].mean(axis=0)
        taus[b] = st.kendalltau(-means, base_rank).statistic
        top1[DET6[int(np.argmax(means))]] += 1
    res_a[m] = dict(tau_med=round(float(np.median(taus)), 3),
                    tau_iqr=[round(float(np.percentile(taus, 25)), 3),
                             round(float(np.percentile(taus, 75)), 3)],
                    top1_share={k: round(v / B, 3) for k, v in top1.items()})
    print("m=%d tau_med=%.3f top1=%s" % (m, res_a[m]["tau_med"], res_a[m]["top1_share"]))

# seed axis: cmhmil rank among 6 detectors, seeds subsampled per suite
others = X[:, 1:].mean(axis=0)
res_b = {}
for s_n in (2, 4, 6, 8, 10):
    suite_seeds = {ds: CMH_SEEDS[ds][:s_n] for ds in CORE6}
    moves = []
    for b in range(B):
        rng2 = np.random.default_rng(20260925 + b)
        vals = []
        for j, ds in enumerate(CORE6):
            ss = suite_seeds[ds]
            pick = [ss[i] for i in rng2.integers(0, len(ss), s_n)]
            vals.append(np.mean([auc(ds, s) for s in pick]))
        cmh_val = vals[0]  # cmhmil is DET6[0]
        others_v = vals[1:]
        rank = int(sum(1 for v in others_v if v > cmh_val)) + 1
        moves.append(rank)
    res_b[s_n] = dict(median=round(float(np.median(moves)), 1),
                      ci=[int(np.percentile(moves, 2.5)), int(np.percentile(moves, 97.5))])
    print("seeds=%d cmhmil rank med=%s ci=%s" % (s_n, res_b[s_n]["median"], res_b[s_n]["ci"]))

json.dump({"suite_bootstrap": res_a, "seed_axis": res_b},
          open(os.path.join(RES, "r38_rankstab.json"), "w"), indent=1)
print("saved r38_rankstab.json")

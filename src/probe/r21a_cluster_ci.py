# -*- coding: utf-8 -*-
"""Plan A Card 3a: recompute all 300 end-to-end seed pairs, run-level cluster bootstrap CI.

Panel: cmhmil seeds {0,1,2,3,7,31,42,97,123,2024} on SMD/PSM/SWaT (10 each),
{0..10} on MSL/SMAP/WADI (11 each) -> 64 runs, 300 unordered pairs.
Statistic: window-level paired t p-value per pair (the de-facto protocol the
paper audits); rejection = p<.05. Validation target: 217/300 = 72%.
CI: dataset-stratified run-level cluster bootstrap, 2000 replicates.
"""
import os, sys, itertools, json
import numpy as np
from scipy import stats

sys.path.insert(0, r"D:\0科研\工作1\第15篇SCI\src")
from bootstrap.e3_clean import load_goodness  # noqa: E402

DATASETS = {
    "SMD":  [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024],
    "PSM":  [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024],
    "SWaT": [0, 1, 2, 3, 7, 31, 42, 97, 123, 2024],
    "MSL":  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "SMAP": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "WADI": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}
OUT = r"D:\0科研\工作1\第15篇SCI\02_实验记录\results\r21a_cluster_ci.csv"
JS = r"D:\0科研\工作1\第15篇SCI\02_实验记录\results\r21a_cluster_ci.json"

pairs = []  # (dataset, seedA, seedB, p_win_t)
for ds, seeds in DATASETS.items():
    gs = {}
    for s in seeds:
        g, _ = load_goodness("cmhmil_seed%d" % s, ds)
        gs[s] = g
    for a, b in itertools.combinations(seeds, 2):
        d = gs[a] - gs[b]
        t2 = stats.ttest_1samp(d, 0.0)
        pairs.append((ds, a, b, float(t2.pvalue)))
    print(ds, "pairs done", len(pairs))

n_rej = sum(1 for p in pairs if p[3] < .05)
print("rejections %d/%d = %.1f%%" % (n_rej, len(pairs), 100 * n_rej / len(pairs)))
per_ds = {}
for ds in DATASETS:
    pp = [p for p in pairs if p[0] == ds]
    r = sum(1 for p in pp if p[3] < .05)
    per_ds[ds] = (r, len(pp), round(100 * r / len(pp), 1))
    print(ds, r, len(pp), round(100 * r / len(pp), 1))

# run-level cluster bootstrap (dataset-stratified)
rej = {(p[0], p[1], p[2]): (p[3] < .05) for p in pairs}
rng = np.random.default_rng(20260910)
B = 2000
rates = []
for b in range(B):
    tot, rej_n = 0, 0
    for ds, seeds in DATASETS.items():
        draw = [seeds[i] for i in rng.integers(0, len(seeds), len(seeds))]
        for a, bb in itertools.combinations(range(len(draw)), 2):
            sa, sb = draw[a], draw[bb]
            if sa == sb:
                continue
            key = (ds, min(sa, sb), max(sa, sb))
            tot += 1
            rej_n += 1 if rej.get(key, False) else 0
    rates.append(rej_n / tot)
rates = np.sort(np.array(rates))
ci = (round(100 * rates[int(.025 * B)], 1), round(100 * rates[int(.975 * B)], 1))
print("run-level cluster bootstrap 95%% CI: [%s, %s] %% (point %s)" %
      (ci[0], ci[1], round(100 * n_rej / len(pairs), 1)))

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    f.write("dataset,seedA,seedB,p_win_t,reject\n")
    for ds, a, b_, p in pairs:
        f.write("%s,%d,%d,%.6g,%d\n" % (ds, a, b_, p, 1 if p < .05 else 0))
json.dump({"n_rej": n_rej, "n_pairs": len(pairs),
           "rate_pct": round(100 * n_rej / len(pairs), 1),
           "ci95_pct": list(ci), "per_dataset": per_ds, "B": B},
          open(JS, "w", encoding="utf-8"), indent=1)
print("saved", OUT)

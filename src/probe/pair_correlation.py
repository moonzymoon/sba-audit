# -*- coding: utf-8 -*-
"""R7 zh:zh_299: pairedzh:zh_3615 — realzh:zh_8800detectorblockzh:zh_4542difference vs zh:zh_9771σ_d (zh:externalzh_8980V5/M10zh:zh_1311)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "D:/0keyan/gongzuo1/paper 15SCI/src")
sys.path.insert(0, "D:/0keyan/gongzuo1/paper 15SCI/src/bootstrap")
from common.blocks import build_blocks, block_stats  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from bootstrap.e3_clean import load_goodness  # noqa: E402

R = "D:/0keyan/gongzuo1/paper 15SCI/02_shiyanjilu/results/"
BMS = ["SMD", "PSM", "MSL", "SMAP", "WADI"]
SC = ["cmhmil_seed7", "iforest", "pca", "gmm", "ocsvm", "lof"]

e4 = pd.read_csv(R + "e4_mde_table.csv").set_index("dataset")
g8 = pd.read_csv(R + "e8_gap_landscape.csv")

rows = []
for bm in BMS:
    means, y = {}, None
    for sc in SC:
        g, y_ = load_goodness(sc, bm)
        y = y_
        st = block_stats(g, build_blocks(events_from_binary(y), len(y)), min_seg=10)
        means[sc] = np.array([b["m"] for b in st])
    mde_new = e4.loc[bm, "MDE_AUROC_pts"]
    sd_d = e4.loc[bm, "sigma_d"]
    for i in range(len(SC)):
        for j in range(i + 1, len(SC)):
            d = means[SC[i]] - means[SC[j]]
            sd_pair = float(d.std(ddof=1))
            rho_ab = float(np.corrcoef(means[SC[i]], means[SC[j]])[0, 1])
            gap = g8[(g8.dataset == bm) & (g8["pair"] == f"deep-{SC[j].replace('_seed7','')}")]
            gp = float(gap.gap_pts.iloc[0]) if len(gap) else np.nan
            ratio = sd_pair / sd_d
            rows.append(dict(bm=bm, pair=f"{SC[i][:4]}-{SC[j][:4]}", sd_pair=sd_pair,
                             sd_twodraw=sd_d, ratio=ratio, rho_ab=rho_ab,
                             mde_new=mde_new, mde_paired=mde_new * ratio,
                             gap=gp, rho_paired=(gp / (mde_new * ratio) if np.isfinite(gp) else np.nan)))
df = pd.DataFrame(rows)
df.to_csv(R + "pair_scale_analysis.csv", index=False)
print("== per benchmark summary ==")
for bm in BMS:
    x = df[df.bm == bm]
    print("%-5s ratio SD_pair/sd_d: mean %.3f  max %.3f | rho_ab mean %.3f | "
          "MDE_new %.0f -> MDE_paired max %.0f | flips(rho>=1): %d/%d"
          % (bm, x.ratio.mean(), x.ratio.max(), x.rho_ab.mean(),
             x.mde_new.iloc[0], x.mde_paired.max(), int((x.rho_paired >= 1).sum()), len(x)))
print("\nmax rho_paired overall: %.3f" % df.rho_paired.max())

print("\n== V_seed zh:zh_2843datasetzh:zh_6589 ==")
mt = pd.read_csv(R + "e1_full_metric_table.csv")
cmh = mt[(mt.detector == "cmhmil") & (mt.dataset.isin(BMS))]
v = cmh.groupby("dataset")["AUROC"].agg(["var", "count", "std"])
print(v.round(5).to_string())

print("\n== WADI blockzh:zh_1935 CV ==")
g, y = load_goodness("cmhmil_seed7", "WADI")
st = block_stats(g, build_blocks(events_from_binary(y), len(y)), min_seg=10)
ev = np.array([b["n"] for b in st if b["kind"] == "event"])
no = np.array([b["n"] for b in st if b["kind"] == "normal"])
print("event blocks: n=%d len CV=%.2f | normal: n=%d len CV=%.2f"
      % (len(ev), ev.std(ddof=1) / ev.mean(), len(no), no.std(ddof=1) / no.mean()))

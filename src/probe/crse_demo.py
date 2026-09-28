# -*- coding: utf-8 -*-
"""R8 zh:zh_5793: (a) zh:zh_5509SE vs blockzh:zh_4542t vs zh:zh_533windowt zh:zh_4930valuepairzh:zh_9388 (75pair);
(b) %GRR pair V_eval zh:zh_3551 zh:zh_7253; (c) zh:benchmarkzh_796tabledata."""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, "D:/0科研/工作1/第15篇SCI/src")
sys.path.insert(0, "D:/0科研/工作1/第15篇SCI/src/bootstrap")
from common.blocks import build_blocks, block_ids  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from bootstrap.e3_clean import load_goodness  # noqa: E402

R = "D:/0科研/工作1/第15篇SCI/02_实验记录/results/"
BMS = ["SMD", "PSM", "MSL", "SMAP", "WADI"]
SC = {"cmhmil_seed7": "deep", "iforest": "iforest", "pca": "pca", "gmm": "gmm", "ocsvm": "ocsvm", "lof": "lof"}
MIN_SEG = 10

rows = []
for bm in BMS:
    means, gs, y = {}, {}, None
    for sc in SC:
        g, y_ = load_goodness(sc, bm)
        y = y_
        gs[sc] = g
    blocks = build_blocks(events_from_binary(y), len(y))
    bid = block_ids(blocks, len(y))
    # zh:zh_7974cellblockzh:zh_4161 (zh:eventsegment>=10) — with block_stats identical
    qual = np.zeros(len(blocks), dtype=bool)
    for k, (_, s, e) in enumerate(blocks):
        qual[k] = (_[0] if isinstance(_, str) else blocks[k][0]) == "normal" or (e - s) >= MIN_SEG
    ks = list(SC)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = gs[ks[i]] - gs[ks[j]]
            Tn = len(d)
            dbar = d.mean()
            # zh:zh_533window t (T-1 df)
            p_naive = st.ttest_1samp(d, 0.0).pvalue
            # blockzh:zh_4542 t: zh:zh_7974cellblock blockmeandifference, K-1 df
            mask = qual[bid]
            dm, cnt = {}, {}
            for t in range(Tn):
                if mask[t]:
                    dm[bid[t]] = dm.get(bid[t], 0.0) + d[t]
                    cnt[bid[t]] = cnt.get(bid[t], 0) + 1
            dmeans = np.array([dm[b] / cnt[b] for b in sorted(dm)])
            K = len(dmeans)
            p_block = st.ttest_1samp(dmeans, 0.0).pvalue
            # CR1 zh:zh_5509 (zh:zh_5665blockis zh:zh_8965, K-1 df)
            S = {b: 0.0 for b in dm}
            for t in range(Tn):
                if bid[t] in S:
                    S[bid[t]] += d[t] - dbar * cnt.get(bid[t], 1) / cnt.get(bid[t], 1)
            # CR1: sum_j (S_j - n_j*dbar)^2 / T^2 * K/(K-1), S_j=zh:zh_4499
            Sj = np.array([dm[b] for b in sorted(dm)]) * np.array([cnt[b] for b in sorted(dm)])
            nj = np.array([cnt[b] for b in sorted(dm)])
            Vcr = (K / (K - 1)) * np.sum((Sj - nj * dbar) ** 2) / Tn**2
            t_cr = dbar / np.sqrt(Vcr)
            p_cr = 2 * st.t.sf(abs(t_cr), K - 1)
            rows.append(dict(bm=bm, pair=f"{SC[ks[i]]}-{SC[ks[j]]}", K=K,
                             p_naive=p_naive, p_cr=p_cr, p_block=p_block,
                             d_cr_block=abs(p_cr - p_block)))
df = pd.DataFrame(rows)
df.to_csv(R + "crse_vs_block.csv", index=False)
naive_sig = (df.p_naive < .05).mean()
cr_sig = (df.p_cr < .05).mean()
blk_sig = (df.p_block < .05).mean()
print("75 pairs: naive p<.05: %.2f | CRSE p<.05: %.2f | block p<.05: %.2f" % (naive_sig, cr_sig, blk_sig))
print("|p_CR - p_block| median %.4f max %.4f" % (df.d_cr_block.median(), df.d_cr_block.max()))
print(df.nlargest(3, "d_cr_block")[["bm", "pair", "p_naive", "p_cr", "p_block", "d_cr_block"]].round(4).to_string())

# (b) %GRR pair V_eval zh:zh_3551
j = json.load(open(R + "e1_full_components.json", encoding="utf-8"))["AUROC"]
Ve, Vsd, Vdd, Vds, Vdet = j["V_eval(mean)"], j["V_seed"], j["V_detxds_resid"], j["V_dataset"], j["V_detector"]
for scale in [1, 2, 4, 8]:
    ve = Ve / scale
    tot = Vds + Vdet + Vdd + Vsd + ve
    grr = np.sqrt((Vsd + ve + Vdd) / tot) * 100
    ndc = 1.41 * np.sqrt(Vdet / (Vsd + ve + Vdd))
    print("Veval/%d: %%GRR=%.1f NDC=%.3f" % (scale, grr, ndc))

# (c) zh:benchmarkzh_796
e3 = pd.read_csv(R + "e3_clean_summary.csv")
e4 = pd.read_csv(R + "e4_mde_table.csv")
tau = pd.read_csv(R + "e2_tau_map.csv")
tab = e3.groupby("dataset").agg(T=("T", "median")).reset_index()
print(tab.merge(e4[["dataset", "anom_rate", "K_blocks"]], on="dataset", how="left").to_string())

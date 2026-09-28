# -*- coding: utf-8 -*-
"""R24: append EMG rows to inflation_factor.csv (r from e3clean npz), then
recompute the size-vs-prediction correlation over all cells (was 0.964/44)."""
import glob, os, sys
import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, r"D:\0科研\工作1\第15篇SCI\src")
from bootstrap.e3_clean import load_goodness  # noqa: E402

RCACHE = r"D:\0科研\工作1\第15篇SCI\_resample_cache"
RES = r"D:\0科研\工作1\第15篇SCI\02_实验记录\results"

rows = []
for f in sorted(glob.glob(os.path.join(RCACHE, "e3clean_EMG_*.npz"))):
    d = np.load(f)
    scorer = os.path.basename(f)[len("e3clean_EMG_"):-len(".npz")]
    T = int(d["T"]); V = float(d["dbar"].var(ddof=1))
    g, _ = load_goodness(scorer, "EMG")
    r = T * V / (2 * g.var(ddof=1))
    pred = 2 * st.norm.cdf(-1.96 / np.sqrt(r))
    swin = float((d["win_t"] < .05).mean())
    rows.append(dict(dataset="EMG", scorer=scorer, r=r, pred_size=pred, size_win_t=swin))
    print("EMG/%s r=%.1f pred=%.3f obs=%.3f" % (scorer, r, pred, swin))

inf_path = os.path.join(RES, "inflation_factor.csv")
inf = pd.read_csv(inf_path, encoding="utf-8-sig")
inf = inf[inf.dataset != "EMG"]
inf = pd.concat([inf, pd.DataFrame(rows)], ignore_index=True)
inf.to_csv(inf_path, index=False, encoding="utf-8-sig")

cc = inf.dropna(subset=["r", "pred_size", "size_win_t"])
rho = np.corrcoef(cc.pred_size, cc.size_win_t)[0, 1]
print("cells with computable r:", len(cc), "| correlation pred-vs-obs: %.3f" % rho)
# leave-one-out min
loos = []
vals = cc[["pred_size", "size_win_t"]].values
for i in range(len(vals)):
    m = np.delete(vals, i, axis=0)
    loos.append(np.corrcoef(m[:, 0], m[:, 1])[0, 1])
print("LOO min correlation: %.3f" % min(loos))
rmse = float(np.sqrt(((cc.pred_size - cc.size_win_t) ** 2).mean()))
print("RMSE: %.3f" % rmse)

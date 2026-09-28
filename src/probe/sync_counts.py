# -*- coding: utf-8 -*-
"""扫描收尾后的计数同步分析器 (armed; 聚合完成后运行).

输出论文需要联动更新的全部数字: 总格数/窗口可算/退化/块可算, 尺寸范围,
新增格明细(按 dataset×scorer 显式识别, 不依赖行序), Table 3 受影响行.
"""
import pandas as pd

R = "D:/0科研/工作1/第15篇SCI/02_实验记录/results/"
df = pd.read_csv(R + "e3_clean_summary.csv")
print("total cells:", len(df))

win, blk = df["size_win_t"], df["size_blk_t"]
n_win, n_blk = int(win.notna().sum()), int(blk.notna().sum())
print("window-computable:", n_win, "(old 83)")
print("degenerate:", len(df) - n_win, "(old 17)")
print("block-computable:", n_blk, "(old 69)")
print("window size range: %.3f - %.3f (old 0.089-0.972)" % (win.min(), win.max()))
print("block size range: %.3f - %.3f (old 0.000-0.055)" % (blk.min(), blk.max()))
print("block anti-conservative cells (>0.057):", int((blk > 0.057).sum()))

NEW = [("TE", "lof"), ("TE", "ocsvm"), ("MetroPT3", "pca"), ("NEweather", "ocsvm"),
       ("TSB017", "gmm"), ("TSB170", "gmm")]
m = df.apply(lambda r: (r["dataset"], r["scorer"]) in NEW, axis=1)
print("\n=== expansion cells (identified by dataset x scorer) ===")
print(df[m][["dataset", "scorer", "T", "n_blocks", "size_win_t",
             "size_blk_t", "size_n1_signperm"]].round(4).to_string())

print("\n=== Table 3 affected rows ===")
te = df[df.dataset.eq("TE")]
print("TE rows:\n", te[["scorer", "size_win_t", "size_blk_t"]].round(3).to_string())
tsb = df[df.dataset.str.startswith("TSB", na=False)]
ok = tsb[tsb["size_win_t"].notna()]
print("TSB adequate:", len(ok), "(old 11)",
      "win %.2f-%.2f (old 0.71-0.97) blk %.3f-%.3f (old 0.045-0.055)"
      % (ok["size_win_t"].min(), ok["size_win_t"].max(),
         ok["size_blk_t"].min(), ok["size_blk_t"].max()))
canon = df[df.dataset.isin(["SMD", "PSM", "MSL", "SMAP", "WADI"])]
sh = canon[canon.scorer.isin(["gmm", "ocsvm", "lof"])]
print("GMM/OCSVM/LOF canonical:", len(sh), "(old 15)",
      "win %.2f-%.2f blk %.3f-%.3f (old 0.85-0.97 / 0.042-0.052)"
      % (sh["size_win_t"].min(), sh["size_win_t"].max(),
         sh["size_blk_t"].min(), sh["size_blk_t"].max()))
ind = df[df.dataset.isin(["MetroPT3", "NEweather", "BATADAL", "TE"])]
print("industrial cells summary:\n",
      ind[["dataset", "scorer", "size_win_t", "size_blk_t"]].round(3).to_string())

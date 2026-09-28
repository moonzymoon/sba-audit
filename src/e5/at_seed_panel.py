"""E5b: 第二深度族 (Anomaly Transformer) 种子面板 — 消除 limits (iv) 的单族局限.

复用 paper2 GITHUB_UPLOAD 的 AT 实现与超参 (win 100, stride 50, d_model 64, 10 epochs,
lr 1e-4, lam 3.0, 35/10/10/45 四块分割), 逐 seed 训练并保存测试段分数缓存
at2_{ds}_seed{N}.npz (scores/labels 同 _score_cache 约定).

用法: python e5/at_seed_panel.py        # SMD,PSM × seeds {7,42,123,202,999}
"""
import os
import sys
import time

import numpy as np
import torch

AT_SRC = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/GITHUB_UPLOAD/src"
sys.path.insert(0, AT_SRC)
sys.path.insert(0, r"D:/0科研/工作1/第15篇SCI/src")
from baseline_anomaly_transformer import AnomalyTransformerModel, _make_windows  # noqa: E402
from scorers_extra import load_any  # noqa: E402

CACHE = r"D:/0科研/工作1/第6篇SCI/src/_score_cache"
SEEDS = [7, 42, 123, 202, 999]
DATASETS = ["SMD", "PSM"]
WIN, STRIDE, EPOCHS, BS, LR, LAM = 100, 50, 10, 32, 1e-4, 3.0


def train_score(X, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = X.shape[1]
    T = len(X)
    a, c = int(T * 0.35), int(T * 0.55)
    mu, sd = X[:a].mean(0), X[:a].std(0) + 1e-8
    Xs = (X - mu) / sd
    Xtr_w, _ = _make_windows(Xs[:a], WIN, STRIDE)
    model = AnomalyTransformerModel(D, WIN, d_model=64, n_heads=2, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt = torch.from_numpy(Xtr_w.astype(np.float32))
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xt))
        for i in range(0, len(perm), BS):
            x = xt[perm[i:i + BS]]
            recon, disc = model(x)
            loss = ((recon - x) ** 2).mean() + LAM * disc.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def score_windows(data_raw):
        wins, starts = _make_windows((data_raw - mu) / sd, WIN, STRIDE)
        wt = torch.from_numpy(wins.astype(np.float32))
        sc = np.zeros(len(data_raw))
        cnt = np.zeros(len(data_raw))
        with torch.no_grad():
            for i in range(0, len(wt), 64):
                recon, disc = model(wt[i:i + 64])
                re = ((recon - wt[i:i + 64]) ** 2).mean(-1).numpy()
                dc = disc.numpy()
                for j, s in enumerate(starts[i:i + 64]):
                    sc[s:s + WIN] += re[j] + dc[j]
                    cnt[s:s + WIN] += 1
        cnt[cnt == 0] = 1
        return sc / cnt

    return score_windows(X[c:]), c


def main():
    for ds in DATASETS:
        X, Y = load_any(ds)
        for seed in SEEDS:
            f = os.path.join(CACHE, f"at2_{ds}_seed{seed}.npz")
            if os.path.exists(f):
                print(f"[skip] {ds} seed{seed}")
                continue
            t0 = time.time()
            sc, c = train_score(X, seed)
            y = Y[c:]
            np.savez(f, scores=sc.astype(np.float64), labels=y.astype(np.int8))
            print(f"[done] {ds} seed{seed}: T={len(sc)} 用时{time.time()-t0:.0f}s", flush=True)
    print("at_seed_panel done")


if __name__ == "__main__":
    main()

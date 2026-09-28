"""扩充打分器与新数据集加载器 (预注册 R3 附加扩充, 2026-09-02).

目标: 4 族 8 检测器 (MIL族 cmhmil / 关联族 AT / 树族 iforest / 线性族 PCA /
密度族 GMM+OCSVM / 近邻族 LOF) × 数据集从 6 扩到 6+5+TSB子集.
纪律: 与既有缓存完全同协议 —— 35/15/50 时序切分, 窗口 w=16 flatten,
标签锚点=窗口右端 (yw = Yt[w-1:]), 缓存 {scorer}_{ds}.npz 进第6篇 _score_cache.
核方法 (OCSVM/LOF) 训练子采样 ≤8000, 超大测试集 (MetroPT3) 跳过并记录.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

P2 = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/src"
DS_ROOT = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/datasets"
CACHE = r"D:/0科研/工作1/第6篇SCI/src/_score_cache"
W = 16


def _zscore(Xtr, Xall):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xall - mu) / sd


def _split(T):
    """35/15/50 时序切分 (与第2篇 chronological_split 逐位同口径: 直接取整)."""
    a = int(T * 0.35)
    b = int(T * 0.50)
    return a, b, T  # weak_train [0,a), calib [a,b), test [b,T)


# ============ 新数据集加载器 (返回整段 (X, Y), 与第2篇 loader 同口径) ============
# TE 与 MetroPT3 采用第9篇的权威加载器 (与既有 cmhmil 缓存同源, 标签逐位一致):

P9_SRC = r"D:/0科研/工作1/第9篇SCI/code_release/src"


AIR_LEAK_EVENTS = [  # failure report 气漏区间 (闭区间), 第9篇 metropt3.py 同源
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 00:00", "2020-05-30 23:59"),
    ("2020-06-05 00:00", "2020-06-07 23:59"),
    ("2020-07-15 00:00", "2020-07-15 23:59"),
]
TE_FAULT_FILES = ["mode1_1_1", "mode1_1_2", "mode1_4_1",
                  "mode1_4_2", "mode1_11_1", "mode1_11_2"]  # Rieth IDV(1)/(4)/(11)


def load_metropt3():
    """MetroPT3 (UCI, 空压机). 标签=4 个气漏事件窗; ds=5; 语义与 cmhmil 缓存逐位一致(已验证)."""
    import pandas as pd
    csvs = [f for f in os.listdir(os.path.join(DS_ROOT, "MetroPT3")) if f.endswith(".csv")]
    df = pd.read_csv(os.path.join(DS_ROOT, "MetroPT3", csvs[0]))
    ts = pd.to_datetime(df["timestamp"])
    feat_cols = [c for c in df.columns if c not in ("timestamp", df.columns[0])]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    Y = np.zeros(len(df), dtype=np.int8)
    for s, e in AIR_LEAK_EVENTS:
        Y[((ts >= s) & (ts <= e)).values] = 1
    X, Y = X[::5], Y[::5]
    n_tr = int(len(X) * 0.35)
    mu, sd = X[:n_tr].mean(0, keepdims=True), X[:n_tr].std(0, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32), Y


def load_te():
    """TE (Rieth 扩展版). normal 500h + 6 个故障运行(IDV 1/4/11 ×2), 第 8h 注入;
    41 个 XMEAS 交集特征. 语义与 cmhmil_TE 缓存逐位一致(已验证)."""
    import pandas as pd
    te_dir = os.path.join(DS_ROOT, "TE")
    nd = pd.read_excel(os.path.join(te_dir, "mode1_normal_500.xlsx"))
    nfeat = [c for c in nd.columns if c != "Time"]
    xmeas = [f"XMEAS_{i}" for i in range(1, len(nfeat) + 1)]
    Xs = [nd.rename(columns={c: xmeas[i] for i, c in enumerate(nfeat)})[xmeas]
          .values.astype(np.float32)]
    Ys = [np.zeros(len(nd), dtype=np.int8)]
    for name in TE_FAULT_FILES:
        fd = pd.read_excel(os.path.join(te_dir, f"{name}.xlsx"))
        t = fd["Time (h)"].values
        Xs.append(fd[[c for c in fd.columns if c.startswith("XMEAS")]].values.astype(np.float32))
        Ys.append((t >= 8.0).astype(np.int8))
    X = np.vstack(Xs)
    Y = np.concatenate(Ys)
    n_tr = int(len(X) * 0.35)
    mu, sd = X[:n_tr].mean(0, keepdims=True), X[:n_tr].std(0, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32), Y


def load_batadal():
    import pandas as pd
    df = pd.read_csv(os.path.join(DS_ROOT, "BATADAL", "BATADAL_training2.csv"))
    df.columns = [c.strip() for c in df.columns]
    feats = [c for c in df.columns if c not in ("DATETIME", "ATT_FLAG")]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    Y = (df["ATT_FLAG"].values == 1).astype(np.int8)
    return _zscore(X[: len(X) // 2], X), Y


def load_neweather():
    import pandas as pd
    X = pd.read_csv(os.path.join(DS_ROOT, "NEweather", "NEweather_data.csv"),
                    header=None).values.astype(np.float32)
    Y = pd.read_csv(os.path.join(DS_ROOT, "NEweather", "NEweather_class.csv"),
                    header=None).values.ravel().astype(np.int8)
    Y = (Y > 0).astype(np.int8)
    return _zscore(X[: len(X) // 2], X), Y


# 5ECK2022: 数据集文件夹为空 — 移出扩充清单 (诚实登记)


TSB_DIR = os.path.join(DS_ROOT, "TSB-AD-MV", "TSB-AD-M")


def load_tsb(idx):
    """TSB-AD-MV 第 idx 个序列 (文件名排序). 首行为表头, 末列 Label."""
    import pandas as pd
    fs = sorted(glob.glob(os.path.join(TSB_DIR, "*.csv")))
    df = pd.read_csv(fs[idx], header=0)
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    Y = (df.iloc[:, -1].astype(float).values > 0).astype(np.int8)
    return _zscore(X[: len(X) // 2], X), Y


def tsb_list():
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(TSB_DIR, "*.csv")))


def tsb_select(per_source=2, extra=("OPPORTUNITY", "CATSv2", "LTDB", "SWaT", "Genesis", "GECCO")):
    """分层选子集: 大来源各 per_source 条 + 小来源各 1 条. 返回 (索引, 文件名) 列表."""
    files = tsb_list()
    by_src = {}
    for i, f in enumerate(files):
        src = f.split("_")[1]
        by_src.setdefault(src, []).append(i)
    picked = []
    for src, idxs in sorted(by_src.items()):
        k = 1 if src in extra else per_source
        picked.extend(idxs[:k])
    return [(i, files[i]) for i in sorted(picked)]


# ============ 新打分器 (与 iforest_SMD 协议一致) ============

def _windows(Xt, Yt, w=W):
    N = len(Xt) - w + 1
    feats = np.lib.stride_tricks.sliding_window_view(Xt, w, axis=0).reshape(N, -1)
    return feats, Yt[w - 1:]


def pca_scores(X, Y, w=W, use_cache=True, tag=""):
    from sklearn.decomposition import PCA
    name = f"pca{tag}"
    f = os.path.join(CACHE, f"{name}.npz")
    if use_cache and os.path.exists(f):
        d = np.load(f)
        return d["scores"], d["labels"]
    n1, n2, _ = _split(len(X))
    Xt, Yt = X[n2:], Y[n2:]
    Xw = np.lib.stride_tricks.sliding_window_view(X[n1:n2], w, axis=0).reshape(-1, X.shape[1] * w)
    F, yw = _windows(Xt, Yt, w)
    p = PCA(n_components=0.95, random_state=0).fit(Xw)
    sc = ((F - p.inverse_transform(p.transform(F))) ** 2).mean(axis=1)
    np.savez(f, scores=sc.astype(np.float64), labels=yw)
    return sc, yw


def gmm_scores(X, Y, w=W, use_cache=True, tag="", n_sub=8000):
    from sklearn.mixture import GaussianMixture
    name = f"gmm{tag}"
    f = os.path.join(CACHE, f"{name}.npz")
    if use_cache and os.path.exists(f):
        d = np.load(f)
        return d["scores"], d["labels"]
    n1, n2, _ = _split(len(X))
    Xt, Yt = X[n2:], Y[n2:]
    Ftr = np.lib.stride_tricks.sliding_window_view(X[n1:n2], w, axis=0).reshape(-1, X.shape[1] * w)
    rng = np.random.default_rng(0)
    if len(Ftr) > n_sub:
        Ftr = Ftr[rng.choice(len(Ftr), n_sub, replace=False)]
    F, yw = _windows(Xt, Yt, w)
    g = GaussianMixture(8, covariance_type="diag", random_state=0, reg_covar=1e-3).fit(Ftr)
    sc = -g.score_samples(F)
    np.savez(f, scores=sc.astype(np.float64), labels=yw)
    return sc, yw


def ocsvm_scores(X, Y, w=W, use_cache=True, tag="", n_sub=8000):
    from sklearn.svm import OneClassSVM
    name = f"ocsvm{tag}"
    f = os.path.join(CACHE, f"{name}.npz")
    if use_cache and os.path.exists(f):
        d = np.load(f)
        return d["scores"], d["labels"]
    n1, n2, _ = _split(len(X))
    Xt, Yt = X[n2:], Y[n2:]
    if len(Xt) > 400000:
        raise SkipDataset(f"ocsvm: 测试集过大 ({len(Xt)})")
    Ftr = np.lib.stride_tricks.sliding_window_view(X[n1:n2], w, axis=0).reshape(-1, X.shape[1] * w)
    rng = np.random.default_rng(0)
    if len(Ftr) > n_sub:
        Ftr = Ftr[rng.choice(len(Ftr), n_sub, replace=False)]
    F, yw = _windows(Xt, Yt, w)
    m = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(Ftr)
    sc = -m.decision_function(F)
    np.savez(f, scores=sc.astype(np.float64), labels=yw)
    return sc, yw


def lof_scores(X, Y, w=W, use_cache=True, tag="", n_sub=8000):
    from sklearn.neighbors import LocalOutlierFactor
    name = f"lof{tag}"
    f = os.path.join(CACHE, f"{name}.npz")
    if use_cache and os.path.exists(f):
        d = np.load(f)
        return d["scores"], d["labels"]
    n1, n2, _ = _split(len(X))
    Xt, Yt = X[n2:], Y[n2:]
    if len(Xt) > 400000:
        raise SkipDataset(f"lof: 测试集过大 ({len(Xt)})")
    Ftr = np.lib.stride_tricks.sliding_window_view(X[n1:n2], w, axis=0).reshape(-1, X.shape[1] * w)
    rng = np.random.default_rng(0)
    if len(Ftr) > n_sub:
        Ftr = Ftr[rng.choice(len(Ftr), n_sub, replace=False)]
    F, yw = _windows(Xt, Yt, w)
    m = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=2).fit(Ftr)
    sc = -m.score_samples(F)
    np.savez(f, scores=sc.astype(np.float64), labels=yw)
    return sc, yw


class SkipDataset(Exception):
    pass


LOADERS_X = {
    "TE": load_te, "MetroPT3": load_metropt3, "BATADAL": load_batadal,
    "NEweather": load_neweather,
}

CORE_LOADERS = {  # 6 核心数据集走第2篇 loader
    "SMD": None, "PSM": None, "MSL": None, "SMAP": None, "SWaT": None, "WADI": None,
}


def load_any(ds):
    if ds in LOADERS_X:
        return LOADERS_X[ds]()
    sys.path.insert(0, P2)
    from data.loaders import load_dataset
    return load_dataset(ds)


def score_all(datasets=None, scorers=None):
    datasets = datasets or list(LOADERS_X) + list(CORE_LOADERS)
    scorers = scorers or [pca_scores, gmm_scores, ocsvm_scores, lof_scores]
    for ds in datasets:
        try:
            X, Y = load_any(ds)
        except Exception as e:  # noqa: BLE001
            print(f"[LOAD FAIL] {ds}: {e}")
            continue
        for sc_fn in scorers:
            tag = "" if ds in CORE_LOADERS or ds in LOADERS_X else ""
            try:
                s, y = sc_fn(X, Y, use_cache=True, tag=f"_{ds}")
                print(f"[ok] {sc_fn.__name__} {ds}: T={len(s)}")
            except SkipDataset as e:
                print(f"[skip] {sc_fn.__name__} {ds}: {e}")
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {sc_fn.__name__} {ds}: {e}")
    print("score_all 完成")


if __name__ == "__main__":
    import sys as _s
    ds = _s.argv[1:] or None
    score_all(datasets=ds)

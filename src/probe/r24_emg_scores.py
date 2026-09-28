# -*- coding: utf-8 -*-
"""R24: EMG (NRdetector KDD'25 fifth benchmark) into the audit pipeline.

Stage 1 (this script): score EMG with the five shallow detectors under the
suite's frozen protocol and write caches {scorer}_EMG.npz into paper-6's
_score_cache, so bootstrap.e3_clean / load_goodness work unchanged.

Protocol mirrors the suite exactly:
- z-score via first-35% (train) stats (as load_metropt3/load_te do)
- chronological 35/15/50 split; window 16 flatten; label = window right end
- iforest fits on weak-train [0,35%) (paper-6 iforest_scores convention);
  pca/gmm/ocsvm/lof fit on the calibration slice [35%,50%) (scorers_extra
  convention, n_sub=8000 subsample for kernel methods)
- score sign: larger = more anomalous
Data: D:/0keyan/zh:data/TSAD_dataset/EMG/test/{data,label}.npy (sparse-cloned from
github.com/UCSC-REAL/NRdetector; T=130,900 x 8ch, pi=.057).
"""
import os
import numpy as np

EMG = r"D:\0keyan\zh:data\TSAD_dataset\EMG\test"
CACHE = r"D:\0keyan\gongzuo1\paper 6SCI\src\_score_cache"
W = 16

X = np.load(os.path.join(EMG, "data.npy")).astype(np.float32)
Y = np.load(os.path.join(EMG, "label.npy")).reshape(-1).astype(np.int8)
T = len(X)
n1, n2 = int(T * 0.35), int(T * 0.50)
mu, sd = X[:n1].mean(0), X[:n1].std(0) + 1e-8
X = ((X - mu) / sd).astype(np.float32)

def win(A):
    return np.lib.stride_tricks.sliding_window_view(A, W, axis=0).reshape(-1, A.shape[1] * W)

Xt, Yt = X[n2:], Y[n2:]
F_test = win(Xt)
y_test = Yt[W - 1:]
print("T=%d  test windows=%d  pi_test=%.4f" % (T, len(F_test), float(y_test.mean())))

def save(name, sc):
    sc = sc.astype(np.float64)
    np.savez(os.path.join(CACHE, name + "_EMG.npz"), scores=sc, labels=y_test)
    from sklearn.metrics import roc_auc_score
    print("%-8s AUROC=%.4f  saved" % (name, roc_auc_score(y_test, sc)))

# --- iforest: fit on weak-train [0,n1) ---
from sklearn.ensemble import IsolationForest
F_wt = win(X[:n1])
ifo = IsolationForest(n_estimators=100, contamination="auto", random_state=0, n_jobs=-1).fit(F_wt)
save("iforest", -ifo.decision_function(F_test))
del F_wt, ifo

# --- pca: fit on calib [n1,n2) ---
from sklearn.decomposition import PCA
F_cal = win(X[n1:n2])
p = PCA(n_components=0.95, random_state=0).fit(F_cal)
save("pca", ((F_test - p.inverse_transform(p.transform(F_test))) ** 2).mean(axis=1))
del F_cal, p

def sub(F, n=8000, seed=0):
    if len(F) > n:
        r = np.random.default_rng(seed)
        return F[r.choice(len(F), n, replace=False)]
    return F

# --- gmm ---
from sklearn.mixture import GaussianMixture
F_cal = win(X[n1:n2])
g = GaussianMixture(8, covariance_type="diag", random_state=0, reg_covar=1e-3).fit(sub(F_cal))
save("gmm", -g.score_samples(F_test))
del g

# --- ocsvm ---
from sklearn.svm import OneClassSVM
m = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(sub(F_cal))
save("ocsvm", -m.decision_function(F_test))
del m

# --- lof ---
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=2).fit(sub(F_cal))
save("lof", -lof.score_samples(F_test))
print("stage 1 done")

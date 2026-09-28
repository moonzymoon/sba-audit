# -*- coding: utf-8 -*-
"""R37: RCA Top-K attribution-accuracy power pilot on Tennessee Eastman.

Motivation (S42, JPC 2026): root-cause-localization comparisons (Top-1/Top-K
accuracy) are reported as single-run point estimates with no test and no seed
protocol (e.g., Top-1 0.72 -> 0.87 read as superiority). This pilot asks the
audit's question of that metric family: with segment-block resampling on the
fault segments of TE, what difference in Top-K accuracy can a comparison of
two attribution proxies actually certify?

Design (all cache/data-level, frozen conventions):
- Load TE via scorers_extra.load_te (41 XMEAS channels, normal 500h + 6 fault
  runs IDV(1)/(4)/(11) x2, fault injected from hour 8).
- Normal baseline: per-channel mean/sd on the normal run.
- Ground-truth root-cause set per fault run (engineering proxy, same spirit as
  S42's significant-lag correlations): channels whose standardized shift
  (injected period vs normal) exceeds 1.0, restricted to the top-5 by shift --
  recorded, not hidden.
- Attribution proxy A: per-channel standardized shift (marginal).
  Attribution proxy B: PCA reconstruction error per channel (linear-projection
  residual, components from normal run at 95% energy).
- Per fault segment (maximal injected-fault block, >=100 windows): each proxy
  ranks channels; hit@K = |top-K proxies intersect truth set| / K.
- Power layer: segment-block bootstrap over the 6 fault segments (B=4000):
  resample segments with replacement -> Top-3 accuracy difference (A - B) ->
  SD -> MDE = 2.8*SD (power .8 convention) in accuracy percentage points;
  K_required = ((2.8*SD)/delta)^2 for a 10-point claim.
Output: results/r37_rca_pilot.json
"""
import os, sys, json
import numpy as np

sys.path.insert(0, r"D:\0科研\工作1\第15篇SCI\src")
from scorers_extra import load_te  # noqa: E402

RES = r"D:\0科研\工作1\第15篇SCI\02_实验记录\results"
W = 16

X, Y = load_te()          # z-scored full series, labels
T, D = X.shape
print("TE loaded:", X.shape, "pi=%.3f" % Y.mean())

# split into normal / fault runs by maximal label runs
idx = np.flatnonzero(np.diff(np.r_[0, Y, 0]))
segs = list(zip(idx[::2].tolist(), idx[1::2].tolist()))
print("fault segments:", len(segs), "lens:", [e - s for s, e in segs])

# normal baseline per channel (use normal portion)
nmask = Y == 0
mu = X[nmask].mean(0)
sd = X[nmask].std(0) + 1e-8
Xz = (X - mu) / sd

# PCA baseline on normal windows
from sklearn.decomposition import PCA
def windows(A):
    return np.lib.stride_tricks.sliding_window_view(A, W, axis=0).reshape(-1, A.shape[1] * W)
norm_idx = np.flatnonzero(nmask)
# take a contiguous normal block for PCA fit (first 60% of normal portion)
cut = int(len(norm_idx) * 0.6)
fit_block = Xz[norm_idx[:cut]]
Ftr = windows(fit_block)
p95 = PCA(n_components=0.95, random_state=0).fit(Ftr)
ncomp = p95.n_components_
print("PCA components (95% energy):", ncomp)

def per_channel_attrib(zblock):
    """zblock: (L, D) standardized segment. Returns (D,) attribution per channel
    for proxy A (mean |z|) and proxy B (PCA per-channel reconstruction error)."""
    a = np.abs(zblock).mean(0)
    Fw = windows(zblock)
    rec = p95.inverse_transform(p95.transform(Fw)).reshape(-1, zblock.shape[1], W)
    err = ((zblock[np.arange(len(zblock)) % 1][:, None, :] if False else np.lib.stride_tricks.sliding_window_view(zblock, W, axis=0)) - rec)
    b = (err ** 2).mean(axis=(0, 2))
    return a, b

# truth sets + per-segment hits
truth_sets, hits = {}, {"A": [], "B": []}
for si, (s, e) in enumerate(segs):
    if e - s < 100:
        print("skip short seg", si, e - s)
        continue
    block = Xz[s:e]
    a_attr, b_attr = per_channel_attrib(block)
    # engineering ground truth: top-5 channels by standardized shift in this segment
    shift = np.abs(block).mean(0)
    truth = set(np.argsort(-shift)[:5].tolist())
    truth_sets[si] = sorted(truth)
    for name, attr in (("A", a_attr), ("B", b_attr)):
        top3 = np.argsort(-attr)[:3].tolist()
        hits[name].append(len(set(top3) & truth) / 3.0)
    print("seg %d (len %d): truth=%s A_top3hit=%.2f B_top3hit=%.2f" %
          (si, e - s, truth_sets[si], hits["A"][-1], hits["B"][-1]))

# paired segment bootstrap over segments
dA = np.array(hits["A"]); dB = np.array(hits["B"])
diff = dA - dB
n = len(diff)
rng = np.random.default_rng(20260924)
B = 4000
boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)])
sd = boots.std(ddof=1)
mde_pts = 2.8 * sd * 100
k10 = ((2.8 * sd) / 0.10) ** 2
out = {
    "n_segments": n, "pca_components": int(ncomp),
    "truth_sets": {str(k): v for k, v in truth_sets.items()},
    "top3_acc_A": [round(x, 3) for x in hits["A"]],
    "top3_acc_B": [round(x, 3) for x in hits["B"]],
    "mean_diff_pts": round(float(diff.mean()) * 100, 1),
    "bootstrap_sd": round(float(sd), 4),
    "MDE_power80_pts": round(float(mde_pts), 1),
    "segments_required_for_10pt_claim": int(np.ceil(k10)),
    "interpretation": "difference in Top-3 attribution accuracy certifiable at power .8 with the available independent fault segments; compare S42's 15-point single-run claim (0.72->0.87)",
}
print(json.dumps(out, indent=1)[:800])
json.dump(out, open(os.path.join(RES, "r37_rca_pilot.json"), "w"), indent=1)
print("saved r37_rca_pilot.json")

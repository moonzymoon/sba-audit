"""τ zh:zh_3183 (zh:zh_5303segment0zh:zh_6111, zh:zh_2856submissiondesign 2026-09-02, zh:zh_4980final draft).

 : zh:test"windowzh:zh_8297+zh:eventsegmentzh:zh_3635paireddifferencezh:zh_6254differencezh:inflation"zh:assumptionzh_3757**zh:eventsegmentzh:zh_4542**zh:zh_6721 bite.

zh:design:
- zh:data: SMD (= machine-1-1 cache); detectorpair: (cmhmil_seed7, iforest) zh:zh_8529pair, (cmhmil_seed7, AT) zh:zh_991pair
- paireddifferencezh:series: D_t = z(s1)_t - z(s2)_t, z is zh:zh_9405 test zh:serieszh_2570 (zh:zh_6891, notzh:zh_8576correlationzh_7202)
- zh:zh_7194 zh:zh_1031:
  (a) tau_win_full[stride]: zh:zh_9405 test zh:series(zh:normal+zh:anomaly)zh:zh_6301correlation — zh:zh_7623reference(zh:zh_216 O(w) zh:zh_4542, zh:i.e.zh_9837)
  (b) tau_seg[stride]:      zh:eventsegmentzh:zh_3990windowpaireddifference zh:zh_8705correlation — **zh:zh_1065**
      (segmenttreated aszh:zh_583, zh:zh_49differencezh:zh_816segmentzh:zh_3990pairzh:estimate, zh:globalmeanzh:zh_946; Geyer zh:initialzh_804serieszh_24)
  (c) segmentmeanzh:series d_k   lag-1 zh:zh_8847correlation (K=8 segment, onlyzh:zh_384)
- stride ∈ {1, 4, 16} (w=16: zh:zh_6003 / 75% zh:zh_8297 / notzh:zh_8297)
- zh:sensitivezh_4316: zh:zh_5265 <10  zh:zh_5458segmentzh:zh_882 tau_seg; segmentzh:zh_4542 bootstrap(B=200) zh:zh_2447 tau_seg zh:zh_5382 95%CI
- zh:zh_598 (pre-registered): tau_seg[stride=1] >= 1.212 → zh:zh_4384, zh:zh_1224;
  1.212 = zh:zh_785 α=0.05 two-zh:zh_2149 z zh:testempirical size zh:inflationzh_1166 1.5×  zh:zh_6254differencezh:inflationfactor:
  size = 2Φ(-1.96/√τ) = 0.075 ⇒ √τ = 1.96/Φ^{-1}(0.9625) ≈ 1.101 ⇒ τ ≈ 1.212

zh:zh_5018record: τ zh:zh_7448. zh:zh_1130pair tau_seg[stride=1] zh:zh_3634; zh:zh_1702is zh:reference.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.events import events_from_binary  # noqa: E402

CACHE = r"D:/0科研/工作1/第6篇SCI/src/_score_cache"
OUT = r"D:/0科研/工作1/第15篇SCI/02_实验记录/results/tau_probe.csv"
THRESH = 1.212
STRIDES = (1, 4, 16)
W = 16  # cmhmil/iforest zh:zh_33windowzh:zh_1935


def load_pair(n1: str, n2: str, max_shift=200):
    """zh:loadzh_1666 cachezh:scorezh_58perlabelzh:zh_8102bitzh_3601pairzh:zh_4248.

    notsamezh:zh_8071cachezh:zh_5104windowzh:zh_8501notsame (cmhmil: labelzh:zh_1261 w-1=15; AT: labelnotzh:zh_1371,
    zh:zh_3107tailzh_580 79 zh:point) —— zh:zh_8215labelzh:serieszh_8516 shift zh:zh_125segmentfullyidentical, zh:zh_6790.
    """
    d1 = np.load(os.path.join(CACHE, f"{n1}.npz"))
    d2 = np.load(os.path.join(CACHE, f"{n2}.npz"))
    s1, y1 = d1["scores"].astype(np.float64), d1["labels"].astype(np.int8)
    s2, y2 = d2["scores"].astype(np.float64), d2["labels"].astype(np.int8)
    best = None
    for shift in range(-max_shift, max_shift + 1):
        # y1[i] with y2[i+shift] pairzh:zh_2483
        a0, a1 = max(0, -shift), min(len(y1), len(y2) - shift)
        if a1 - a0 < 100:
            continue
        if np.array_equal(y1[a0:a1], y2[a0 + shift:a1 + shift]):
            best = shift
            break
    if best is None:
        raise ValueError(f"labelzh:zh_8102bitzh_3601pairzh:zh_4248fail: {n1} vs {n2}")
    shift = best
    a0, a1 = max(0, -shift), min(len(y1), len(y2) - shift)
    return s1[a0:a1], s2[a0 + shift:a1 + shift], y1[a0:a1]


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def _autocov_fft(x):
    n = len(x)
    m = 1
    while m < 2 * n:
        m <<= 1
    f = np.fft.rfft(x, m)
    return np.fft.irfft(f * np.conj(f), m)[:n].real / n


def ips_tau(x, max_lag=None):
    """Geyer zh:initialzh_804(andzh:zh_3163)zh:serieszh_24 zh:zh_6301correlation τ = 1 + 2Σρ_k. zh:zh_542is  NaN (zh:zh_332)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 8:
        return float("nan")
    acov = _autocov_fft(x - x.mean())
    if acov[0] <= 0:
        return float("nan")
    rho = acov / acov[0]
    lim = min(n, max_lag) if max_lag else n
    s, prev, mm = 0.0, np.inf, 1
    while 2 * mm < lim:
        g = rho[2 * mm - 1] + rho[2 * mm]
        if g <= 0:
            break
        g = min(g, prev)  # zh:initialzh_3163series
        s += g
        prev = g
        mm += 1
    return 1.0 + 2.0 * s


def cluster_ips_tau(D, events, T, stride, min_len=0):
    """zh:eventsegmentzh:zh_3990windowpaireddifference zh:zh_8705correlation (zh:zh_67).

    zh:zh_49differencezh:zh_816samesegmentzh:zh_3990 pairzh:estimate: ρ_k = Σ_seg Σ_i (x_i-x̄)(x_{i+k}-x̄) / (samezh:zh_624 k=0),
    x̄ is allzh:zh_6303window zh:globalmean (zh:zh_6303segmentzh:zh_415meanzh:zh_7202——H0 zh:zh_4954differencezh:inflationzh_1325sourcezh:zh_4637).
    returns (tau, N_retained_windows).
    """
    xc_pairs = []  # each segment  (anchors, values)
    N = 0
    total = 0.0
    cnt = 0
    vals_all = []
    for s, e in events:
        if e - s < min_len:
            continue
        idx = np.arange(s, e, stride)
        v = D[idx]
        if len(v) >= 2:
            xc_pairs.append(v)
            vals_all.append(v)
            N += len(v)
    if N < 8 or not xc_pairs:
        return float("nan"), N
    xbar = np.concatenate(vals_all).mean()
    # segmentzh:zh_4253differencezh:zh_3819/zh:zh_4096perpairzh:zh_4230
    num = {}
    den = 0.0
    for v in xc_pairs:
        vc = v - xbar
        den += float(vc @ vc)
        L = len(vc)
        for k in range(1, L):
            num[k] = num.get(k, 0.0) + float(vc[:-k] @ vc[k:])
    if den <= 0:
        return float("nan"), N
    # ρ_k zh:series (zh:zh_8505 lag), IPS zh:zh_24
    maxk = max(num) if num else 0
    rho = np.zeros(maxk + 1)
    rho[0] = 1.0
    for k, val in num.items():
        rho[k] = val / den
    s_acc, prev, mm = 0.0, np.inf, 1
    while 2 * mm < maxk + 1:
        g = rho[2 * mm - 1] + (rho[2 * mm] if 2 * mm <= maxk else 0.0)
        if g <= 0:
            break
        g = min(g, prev)
        s_acc += g
        prev = g
        mm += 1
    return 1.0 + 2.0 * s_acc, N


def seg_mean_lag1(D, events):
    """segmentmeanzh:series (perzh:zh_3615)   lag-1 zh:zh_8847correlation, zh:zh_384."""
    dm = np.array([D[s:e].mean() for s, e in events if e - s >= 10])
    if len(dm) < 4 or dm.std() == 0:
        return float("nan")
    dm = dm - dm.mean()
    return float(dm[:-1] @ dm[1:] / (dm @ dm))


def main():
    rng = np.random.default_rng(20260902)
    rows = []
    pairs = [("cmhmil_SMD_seed7", "iforest_SMD"), ("cmhmil_SMD_seed7", "AT_SMD")]
    for n1, n2 in pairs:
        s1, s2, y = load_pair(n1, n2)
        T = len(y)
        D = zscore(s1) - zscore(s2)
        events = events_from_binary(y)
        print(f"\n== pair: {n1} vs {n2}  (T={T}, segmentzh:zh_4930={len(events)}, "
              f"segmentzh:zh_1935={[e - s for s, e in events]}) ==")
        # (c) segmentzh:zh_6056
        print(f"  segmentmeanzh:series lag-1 ρ (zh:zh_384, K={sum(1 for s, e in events if e - s >= 10)}): "
              f"{seg_mean_lag1(D, events):.3f}")
        for stride in STRIDES:
            # (a) zh:zh_9405series IPS (zh:zh_7623)
            tau_full = ips_tau(D[::stride])
            # (b) segmentzh:zh_9236 IPS (zh:zh_2444)
            tau_seg, Nseg = cluster_ips_tau(D, events, T, stride)
            tau_seg_f, Nseg_f = cluster_ips_tau(D, events, T, stride, min_len=10)
            # segmentzh:resamplingzh_9818 CI (B=200, segmentis zh:zh_8741bit)
            boots = []
            ev_big = [ev for ev in events if ev[1] - ev[0] >= 10]
            for _ in range(200):
                ev_b = [ev_big[j] for j in rng.integers(0, len(ev_big), len(ev_big))]
                t_b, _ = cluster_ips_tau(D, ev_b, T, stride)
                if np.isfinite(t_b):
                    boots.append(t_b)
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2)
            rows.append(dict(pair=f"{n1}|{n2}", stride=stride, tau_win_full=tau_full,
                             tau_seg=tau_seg, N_seg=Nseg, tau_seg_minlen10=tau_seg_f,
                             N_seg_minlen10=Nseg_f, tau_seg_ci_lo=lo, tau_seg_ci_hi=hi))
            print(f"  stride={stride:2d}: tau_win_full={tau_full:6.2f} | "
                  f"tau_seg={tau_seg:6.3f} (N={Nseg:5d}) | "
                  f"tau_seg[zh:zh_8431segment]={tau_seg_f:6.3f} (N={Nseg_f:5d}) | "
                  f"95%CI=[{lo:.2f},{hi:.2f}]")
    # zh:zh_598
    primary = [r for r in rows if r["stride"] == 1]
    verdict = []
    for r in primary:
        ok = np.isfinite(r["tau_seg"]) and r["tau_seg"] >= THRESH
        verdict.append((r["pair"], ok, r["tau_seg"]))
    print("\n== zh:zh_598 (zh:zh_67 tau_seg[stride=1], zh:zh_5340 %.3f) ==" % THRESH)
    for pair, ok, tv in verdict:
        print(f"  {pair}: tau_seg={tv:.3f} -> {'zh:zh_4384' if ok else 'zh:zh_3468'}")
    passed = all(ok for _, ok, _ in verdict)
    print(f"\nzh:zh_3838: {'GO — zh:eventsegmentzh:zh_4542inflationassumptionzh_2725, zh:zh_8650segment1' if passed else 'NO-GO — zh:zh_1799, zh:etc.zh_9749bit'}")
    # zh:thresholdzh_7014
    from scipy.stats import norm
    implied = 2 * norm.sf(1.96 / np.sqrt(THRESH))
    print(f"zh:thresholdzh_7014: tau={THRESH} pairzh:zh_2483empirical size={implied:.4f} (zh:zh_3829 1.5x0.05=0.075)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    import csv
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()

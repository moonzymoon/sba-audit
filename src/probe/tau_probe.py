"""τ 量级探针 (阶段0门禁, 预提交设计 2026-09-02, 跑数前定稿).

目的: 检验"窗口重叠+事件段聚集使配对差方差膨胀"假设在**事件段级**是否有 bite.

设计:
- 数据: SMD (= machine-1-1 缓存); 检测器对: (cmhmil_seed7, iforest) 主对, (cmhmil_seed7, AT) 副对
- 配对差序列: D_t = z(s1)_t - z(s2)_t, z 为全 test 序列标准化 (标准化只统一量纲, 不改自相关结构)
- 三个量:
  (a) tau_win_full[stride]: 全 test 序列(正常+异常)积分自相关 — 上下文参考(预期 O(w) 级, 即稻草人数字)
  (b) tau_seg[stride]:      事件段内窗口配对差的簇加权积分自相关 — **探针主判定量**
      (段视为独立簇, 自协方差只用段内对估计, 全局均值中心化; Geyer 初始正序列截断)
  (c) 段均值序列 d_k 的 lag-1 自相关 (K=8 段, 仅描述性)
- stride ∈ {1, 4, 16} (w=16: 全重叠 / 75% 重叠 / 不重叠)
- 敏感性: 剔除原始时长 <10 的微段重算 tau_seg; 段级 bootstrap(B=200) 给 tau_seg 粗略 95%CI
- 判定 (预注册): tau_seg[stride=1] >= 1.212 → 通过, 继续;
  1.212 = 使标称 α=0.05 双侧 z 检验经验 size 膨胀到 1.5× 的方差膨胀因子:
  size = 2Φ(-1.96/√τ) = 0.075 ⇒ √τ = 1.96/Φ^{-1}(0.9625) ≈ 1.101 ⇒ τ ≈ 1.212

诚实记录: τ 小就写小. 判定只对 tau_seg[stride=1] 主判定量生效; 其余列为参考.
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
W = 16  # cmhmil/iforest 打分窗口长


def load_pair(n1: str, n2: str, max_shift=200):
    """加载两个缓存分数并按标签恒定位移对齐.

    不同打分器缓存时窗口修剪口径不同 (cmhmil: 标签前移 w-1=15; AT: 标签不修剪,
    但尾部短 79 点) —— 用标签序列搜索恒定 shift 使重叠段完全一致, 再截齐.
    """
    d1 = np.load(os.path.join(CACHE, f"{n1}.npz"))
    d2 = np.load(os.path.join(CACHE, f"{n2}.npz"))
    s1, y1 = d1["scores"].astype(np.float64), d1["labels"].astype(np.int8)
    s2, y2 = d2["scores"].astype(np.float64), d2["labels"].astype(np.int8)
    best = None
    for shift in range(-max_shift, max_shift + 1):
        # y1[i] 与 y2[i+shift] 对应
        a0, a1 = max(0, -shift), min(len(y1), len(y2) - shift)
        if a1 - a0 < 100:
            continue
        if np.array_equal(y1[a0:a1], y2[a0 + shift:a1 + shift]):
            best = shift
            break
    if best is None:
        raise ValueError(f"标签恒定位移对齐失败: {n1} vs {n2}")
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
    """Geyer 初始正(且单调)序列截断的积分自相关 τ = 1 + 2Σρ_k. 可为 NaN (样本过短)."""
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
        g = min(g, prev)  # 初始单调序列
        s += g
        prev = g
        mm += 1
    return 1.0 + 2.0 * s


def cluster_ips_tau(D, events, T, stride, min_len=0):
    """事件段内窗口配对差的簇加权积分自相关 (主判定量).

    自协方差只用同段内的对估计: ρ_k = Σ_seg Σ_i (x_i-x̄)(x_{i+k}-x̄) / (同式 k=0),
    x̄ 为全部保留窗口的全局均值 (保留段间均值结构——H0 下方差膨胀来源之一).
    返回 (tau, N_retained_windows).
    """
    xc_pairs = []  # 每段的 (anchors, values)
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
    # 段内自协方差分子/分母按对数加权累加
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
    # ρ_k 序列 (截到最大可用 lag), IPS 截断
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
    """段均值序列 (按时间序) 的 lag-1 自相关, 描述性."""
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
        print(f"\n== 对: {n1} vs {n2}  (T={T}, 段数={len(events)}, "
              f"段长={[e - s for s, e in events]}) ==")
        # (c) 段级描述
        print(f"  段均值序列 lag-1 ρ (描述性, K={sum(1 for s, e in events if e - s >= 10)}): "
              f"{seg_mean_lag1(D, events):.3f}")
        for stride in STRIDES:
            # (a) 全序列 IPS (上下文)
            tau_full = ips_tau(D[::stride])
            # (b) 段级簇 IPS (主判定)
            tau_seg, Nseg = cluster_ips_tau(D, events, T, stride)
            tau_seg_f, Nseg_f = cluster_ips_tau(D, events, T, stride, min_len=10)
            # 段重采样粗 CI (B=200, 段为抽样单位)
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
                  f"tau_seg[去微段]={tau_seg_f:6.3f} (N={Nseg_f:5d}) | "
                  f"95%CI=[{lo:.2f},{hi:.2f}]")
    # 判定
    primary = [r for r in rows if r["stride"] == 1]
    verdict = []
    for r in primary:
        ok = np.isfinite(r["tau_seg"]) and r["tau_seg"] >= THRESH
        verdict.append((r["pair"], ok, r["tau_seg"]))
    print("\n== 判定 (主判定量 tau_seg[stride=1], 门槛 %.3f) ==" % THRESH)
    for pair, ok, tv in verdict:
        print(f"  {pair}: tau_seg={tv:.3f} -> {'通过' if ok else '未过'}")
    passed = all(ok for _, ok, _ in verdict)
    print(f"\n探针结论: {'GO — 事件段级膨胀假设成立, 继续阶段1' if passed else 'NO-GO — 停下, 等待指令调整主推断单位'}")
    # 阈值自检
    from scipy.stats import norm
    implied = 2 * norm.sf(1.96 / np.sqrt(THRESH))
    print(f"阈值自检: tau={THRESH} 对应经验 size={implied:.4f} (目标 1.5x0.05=0.075)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    import csv
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"已写出: {OUT}")


if __name__ == "__main__":
    main()

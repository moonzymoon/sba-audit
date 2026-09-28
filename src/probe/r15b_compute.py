# -*- coding: utf-8 -*-
"""R15b compute: direct-scale cascade numbers (feedback-95 C1).

Outputs:
  results/e1_direct_components.json   (AUROC + AP decompositions with V_eval_direct)
  results/direct_mde_table.csv        (per-benchmark direct floors, K-needed)
  results/r15b_summary.txt            (every number the text cascade needs)
"""
import csv
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats as st

sys.path.insert(0, "D:/0科研/工作1/第15篇SCI/src")
sys.path.insert(0, "D:/0科研/工作1/第15篇SCI/src/bootstrap")
from common.blocks import build_blocks, resample_blocks  # noqa: E402
from common.events import events_from_binary  # noqa: E402
from bootstrap.e3_clean import load_goodness  # noqa: E402

R = "D:/0科研/工作1/第15篇SCI/02_实验记录/results/"
rng = np.random.default_rng(20260907)
OUT = []

def note(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)

# ---------- 1) per-dataset direct evaluation variance (AUROC + AP) ----------
DATASETS = ["SMD", "PSM", "MSL", "SMAP", "SWaT", "WADI", "TE", "MetroPT3"]
SCORERS = ["cmhmil_seed7", "pca", "iforest"]
ve_auc, ve_ap = {}, {}
for ds in DATASETS:
    got = []
    for sc in SCORERS:
        try:
            g, y = load_goodness(sc, ds)
        except Exception:
            continue
        blocks = build_blocks(events_from_binary(y), len(y))
        B = 400 if ds in ("SMD", "PSM", "MSL", "SMAP", "WADI") else 300
        ua = np.empty(B); up = np.empty(B)
        for b in range(B):
            state = rng.bit_generator.state
            gs, _, _ = resample_blocks(g, blocks, rng)
            rng.bit_generator.state = state
            ys, _, _ = resample_blocks(y.astype(float), blocks, rng)
            ys = ys.astype(int)
            ua[b] = roc_auc_score(ys, np.abs(gs))
            up[b] = average_precision_score(ys, np.abs(gs))
        got.append((sc, ua.var(ddof=1), up.var(ddof=1)))
        note(f"[veval] {ds} {sc}: B={B} Var(AUROC)={ua.var(ddof=1):.6f} Var(AP)={up.var(ddof=1):.6f}")
    if got:
        ve_auc[ds] = float(np.mean([v for _, v, _ in got]))
        ve_ap[ds] = float(np.mean([v for _, _, v in got]))
    else:
        note(f"[veval] {ds}: NO CACHE")

v_eval_auc = float(np.mean(list(ve_auc.values())))
v_eval_ap = float(np.mean(list(ve_ap.values())))

# ---------- 2) e1 decomposition with V_eval_direct ----------
j = json.load(open(R + "e1_full_components.json"))
comp = {}
for metric, ve in [("AUROC", v_eval_auc), ("AP", v_eval_ap)]:
    Vds, Vdet, Vres, Vsd = (j[metric]["V_dataset"], j[metric]["V_detector"],
                            j[metric]["V_detxds_resid"], j[metric]["V_seed"])
    tot = Vds + Vdet + Vres + Vsd + ve
    grr2 = Vsd + ve + Vres
    rec = dict(V_dataset=Vds, V_detector=Vdet, V_detxds_resid=Vres, V_seed=Vsd,
               V_eval_direct=ve, V_total=tot,
               V_dataset_pct=round(100 * Vds / tot, 1),
               V_detector_pct=round(100 * Vdet / tot, 1),
               V_detxds_resid_pct=round(100 * Vres / tot, 1),
               V_seed_pct=round(100 * Vsd / tot, 1),
               V_eval_direct_pct=round(100 * ve / tot, 1),
               NDC=round(1.41 * float(np.sqrt(Vdet / grr2)), 2),
               GRR_pctSV=round(100 * float(np.sqrt(grr2 / tot)), 1),
               V_eval_direct_by_ds={k: round(v, 6) for k, v in
                                    (ve_auc if metric == "AUROC" else ve_ap).items()})
    shrink = []
    for f in (2, 4, 8):
        g = Vsd + ve / f + Vres
        t = Vds + Vdet + Vres + Vsd + ve / f
        shrink.append((f, round(100 * np.sqrt(g / t), 1)))
    rec["shrink_series"] = shrink
    comp[metric] = rec
    note(f"[e1-direct {metric}] %GRR={rec['GRR_pctSV']} NDC={rec['NDC']} "
         f"eval%={rec['V_eval_direct_pct']} dataset%={rec['V_dataset_pct']} "
         f"shrink={shrink}")
json.dump(comp, open(R + "e1_direct_components.json", "w"), indent=1)

# ---------- 3) per-pair direct stats ----------
rows = list(csv.DictReader(open(R + "direct_scale_verdict.csv")))
rho = np.array([float(r["gap_pts"]) / float(r["mde_direct_pts"]) for r in rows])
cert = [r for r in rows if r["certifiable_direct"] in ("True", "TRUE", "1")]
note(f"[75] median rho={np.median(rho):.2f} max rho={rho.max():.2f} "
     f"certifiable={len(cert)}/75")
for r in cert:
    note("   CERT:", r["bm"], r["pair"], "gap", r["gap_pts"], "mde", r["mde_direct_pts"],
         "rho", round(float(r["gap_pts"]) / float(r["mde_direct_pts"]), 2))
bonn_mult = (st.norm.ppf(1 - .05 / 30) + st.norm.ppf(.8)) / (st.norm.ppf(.975) + st.norm.ppf(.8))
note(f"[bonf] multiplier={bonn_mult:.3f} max rho adj={rho.max() / bonn_mult:.2f} "
     f"cert-after-bonf={int((rho / bonn_mult >= 1).sum())}/75")

# per-benchmark medians + K-needed
bm_med = {}
for bm in ["SMD", "PSM", "MSL", "SMAP", "WADI"]:
    rr = [r for r in rows if r["bm"] == bm]
    m = float(np.median([float(x["mde_direct_pts"]) for x in rr]))
    sdd = float(np.median([float(x["sd_direct_pts"]) for x in rr]))
    K = {"SMD": 14, "PSM": 107, "MSL": 61, "SMAP": 97, "WADI": 16}[bm]
    bm_med[bm] = m
    note(f"[bm] {bm}: median SD={sdd:.2f} median MDE={m:.1f} K={K} "
         f"K_needed(5pts)={K * (m / 5) ** 2:.0f}")

# pooled FE direct
V = np.array([(float(r["mde_direct_pts"]) / 100 / 2.8016) ** 2 for r in rows])
# use per-benchmark median-based V (one per suite) for the pooled statement
Vb = np.array([(bm_med[b] / 100 / 2.8016) ** 2 for b in bm_med])
mde_pool = 2.8016 * 100 * np.sqrt(1 / np.sum(1 / Vb))
note(f"[pool] FE pooled direct MDE = {mde_pool:.1f} pts (per-suite median basis)")
si = np.sqrt(j["AUROC"]["V_detxds_resid"]); tau = np.sqrt(2) * si
note(f"[pool] RE floor 5 suites = {2.8 * tau / np.sqrt(5) * 100:.1f} pts; suites for 5 = {(2.8 * tau / .05) ** 2:.0f}")

# power .9
mult9 = (st.t.ppf(.975, 13) + st.t.ppf(.90, 13)) / (st.t.ppf(.975, 13) + st.t.ppf(.80, 13))
note(f"[power.9] mult={mult9:.2f} SMD floor {bm_med['SMD']:.1f}->{bm_med['SMD'] * mult9:.1f}")

# walkthrough pair + replay
wt = [r for r in rows if r["bm"] == "SMD" and r["pair"] == "cmhmil_seed7-pca"][0]
wt_mde = float(wt["mde_direct_pts"])
note(f"[walkthrough] SMD deep-pca: gap 19.0 mde_direct={wt_mde:.1f} "
     f"rho={19.0 / wt_mde:.2f} rho_bonf={19.0 / (wt_mde * bonn_mult):.2f}")
for bm, gap in [("MSL", 1.6), ("PSM", 0.5), ("WADI", 1.6)]:
    note(f"[replay] {bm} gap {gap}: rho={gap / bm_med[bm]:.3f}")
note(f"[replay] PSM gap 3.9: rho={3.9 / bm_med['PSM']:.3f}")

with open(R + "r15b_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\nsaved e1_direct_components.json / r15b_summary.txt")

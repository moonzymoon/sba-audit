# -*- coding: utf-8 -*-
"""R15b tex part 2b: abstract, contribution bullet, e1 table+paragraph, fig9 caption.
Values read from results/e1_direct_components.json + direct_scale_verdict.csv."""
import json
import re
import numpy as np

R = "D:/0科研/工作1/第15篇SCI/02_实验记录/results/"
p = "D:/0科研/工作1/第15篇SCI/03_论文/main.tex"
s = open(p, encoding="utf-8").read()
BS = chr(92)
J = json.load(open(R + "e1_direct_components.json"))
A = J["AUROC"]; AP = J["AP"]
rows = list(open(R + "direct_scale_verdict.csv", encoding="utf-8-sig").readlines())
import csv
dv = list(csv.DictReader(open(R + "direct_scale_verdict.csv", encoding="utf-8-sig")))
rho = np.array([float(r["gap_pts"]) / float(r["mde_direct_pts"]) for r in dv])
med_rho, max_rho = np.median(rho), rho.max()
ncert = sum(r["certifiable_direct"].strip().lower() in ("true", "1") for r in dv)
floors = {}
for bm in ["SMD", "PSM", "MSL", "SMAP", "WADI"]:
    floors[bm] = float(np.median([float(r["mde_direct_pts"]) for r in dv if r["bm"] == bm]))
fmin, fmax = min(floors.values()), max(floors.values())
print("VALUES: med_rho=%.2f max_rho=%.2f ncert=%d floors=%s" % (med_rho, max_rho, ncert, floors))

def rep(a, b, tag):
    global s
    assert s.count(a) == 1, "anchor: " + tag
    s = s.replace(a, b)
    print("ok:", tag)

# ---------- abstract ----------
rep("Segment-level permutation tests are calibrated (0.042--0.053) but underpowered: minimum detectable effects of 26--610 AUROC points exceed all 75 observed single-run detector gaps, and gauge R\\&R criteria classify the suite as incapable (\\%GRR 93.8\\%, NDC 0.11).",
    "Segment-level permutation tests are calibrated (0.042--0.053) but underpowered: directly resampled minimum detectable effects of %.0f--%.0f AUROC points exceed %d of the 75 observed single-run detector gaps, and gauge R\\&R criteria still classify the suite as incapable (\\%%GRR %.0f\\%%, NDC %.2f)."
    % (fmin, fmax, ncert, A["GRR_pctSV"], A["NDC"]), "abstract")

# ---------- contribution bullet ----------
rep("Evaluation noise accounts for 83.0\\% of total AUROC-scale variance in the audited suite.",
    "On the directly resampled scale, benchmark-to-benchmark differences dominate score variance (dataset %.0f\\%%) and evaluation noise contributes %.0f\\%%---yet the suite still fails every gauge criterion (\\%%GRR %.0f\\%%, NDC %.2f)."
    % (A["V_dataset_pct"], A["V_eval_direct_pct"], A["GRR_pctSV"], A["NDC"]), "contribution bullet")

# ---------- tab:e1 body rows ----------
rep("""Dataset & 0.0282 & 11.4 \\\\
Detector & 0.0013 & 0.5 \\\\
Detector$\\times$dataset (residual) & 0.0065 & 2.7 \\\\
Seed & 0.0059 & 2.4 \\\\
Evaluation (resampling) & 0.2051 & 83.0 \\\\""".replace("\\\\", BS + BS),
    "Dataset & %.4f & %.1f %s%s Detector & %.4f & %.1f %s%s Detector$%stimes$dataset (residual) & %.4f & %.1f %s%s Seed & %.4f & %.1f %s%s Evaluation (direct resampling) & %.4f & %.1f %s" % (
        A["V_dataset"], A["V_dataset_pct"], BS + BS, BS,
        A["V_detector"], A["V_detector_pct"], BS + BS, BS,
        BS, A["V_detxds_resid"], A["V_detxds_resid_pct"], BS + BS, BS,
        A["V_seed"], A["V_seed_pct"], BS + BS, BS,
        A["V_eval_direct"], A["V_eval_direct_pct"], BS), "e1 table rows")

rep("\\%GRR (study var) & \\multicolumn{2}{c}{93.8\\% (AIAG: $>$30\\% unacceptable)} \\\\".replace("\\%", BS + "%").replace("\\multicolumn", BS + "multicolumn").replace("$>$", "$>$"),
    "\\%%GRR (study var) & %s{2}{c}{%.0f%s (AIAG: $>$30%s unacceptable)} %s" % (
        BS, A["GRR_pctSV"], BS + "%", BS + "%", BS + BS).replace("\\%%", BS + "%"),
    "e1 GRR row")
rep("NDC & \\multicolumn{2}{c}{0.11 (AIAG: $\\geq$5 required)} \\\\".replace("\\multicolumn", BS + "multicolumn").replace("\\geq", BS + "geq"),
    "NDC & %s{2}{c}{%.2f (AIAG: $%sgeq$5 required)} %s" % (BS, A["NDC"], BS, BS + BS), "e1 NDC row")

# ---------- e1 discussion paragraph (full rewrite between anchors) ----------
a0 = "Table~" + BS + "ref{tab:e1}: evaluation noise dominates the total"
i0 = s.find(a0)
i1 = s.find("Two readings of ``repeatability''")
assert 0 < i0 < i1, "e1 paragraph anchors"
newpar = (
 "Table~" + BS + "ref{tab:e1}: on the directly resampled scale, benchmark-to-benchmark differences dominate (%.1f%%) and evaluation noise contributes %.1f%%; detector-to-detector signal remains %.1f%%. "
 "The component estimates carry ANOVA-model uncertainty; we read the thresholds qualitatively, and the verdict no longer hinges on the evaluation-noise estimate at all: shrinking $V_" + BS + "text{eval}$ twofold leaves %%GRR at %.1f%%, fourfold at %.1f%%, eightfold at %.1f%% (the series is nearly flat because evaluation noise is no longer the dominant term). "
 "Under the AIAG criteria the audited suite remains beyond the ``unacceptable'' boundary: with detectors as parts, %%GRR is %.1f%% and the benchmark can distinguish %.2f detector levels---in practice, none. "
 "Nor is the verdict an artifact of the AUROC scale: the average-precision decomposition gives %%GRR %.1f%% and NDC %.2f---crossing the same boundaries. "
 % (A["V_dataset_pct"], A["V_eval_direct_pct"], A["V_detector_pct"],
    A["shrink_series"][0][1], A["shrink_series"][1][1], A["shrink_series"][2][1],
    A["GRR_pctSV"], A["NDC"], AP["GRR_pctSV"], AP["NDC"]))
s = s[:i0] + newpar + s[i1:]
print("ok: e1 paragraph")

# ---------- fig9 body sentence + caption ----------
rep("Figure~\\ref{fig:vardecomp} visualizes the decomposition and its per-dataset texture: evaluation noise dominates on every series with measurable segment structure, and its absolute level varies by two orders of magnitude across benchmarks.",
    "Figure~" + BS + "ref{fig:vardecomp} visualizes the decomposition and its per-dataset texture: directly resampled evaluation noise varies by two orders of magnitude across benchmarks, with WADI the largest and SWaT structurally zero.", "fig9 body")
rep("Left: AUROC-scale variance decomposition of Table~\\ref{tab:e1}---evaluation (resampling) noise accounts for 83.0\\% of total variance; dataset differences 11.4\\%; detector signal 0.5\\%.",
    "Left: AUROC-scale variance decomposition of Table~" + BS + "ref{tab:e1} (direct resampling)---dataset differences %.1f%%, evaluation noise %.1f%%, detector signal %.1f%%."
    % (A["V_dataset_pct"], A["V_eval_direct_pct"], A["V_detector_pct"]), "fig9 caption")

open(p, "w", encoding="utf-8").write(s)
print("part 2b applied")

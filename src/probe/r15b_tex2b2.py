# -*- coding: utf-8 -*-
"""R15b tex 2b continuation: GRR/NDC rows, e1 paragraph, fig9 caption+body."""
import json

R = "D:/0keyan/gongzuo1/paper 15SCI/02_shiyanjilu/results/"
p = "D:/0keyan/gongzuo1/paper 15SCI/03_lunwen/main.tex"
s = open(p, encoding="utf-8").read()
BS = chr(92)
A = json.load(open(R + "e1_direct_components.json"))["AUROC"]
AP = json.load(open(R + "e1_direct_components.json"))["AP"]

def rep(a, b, tag):
    global s
    assert s.count(a) == 1, "anchor: " + tag
    s = s.replace(a, b)
    print("ok:", tag)

new_grr = (BS + "%GRR (study var) & " + BS + "multicolumn{2}{c}{%.1f" % A["GRR_pctSV"]
           + BS + "% (AIAG: $>$30" + BS + "% unacceptable)} " + BS + BS)
new_ndc = "NDC & " + BS + "multicolumn{2}{c}{%.2f (AIAG: $" % A["NDC"] + BS + "geq$5 required)} " + BS + BS
if new_grr in s:
    print("skip: GRR/NDC rows already applied")
else:
    rep(BS + "%GRR (study var) & " + BS + "multicolumn{2}{c}{93.8" + BS + "% (AIAG: $>$30" + BS + "% unacceptable)} " + BS + BS, new_grr, "e1 GRR row")
    rep("NDC & " + BS + "multicolumn{2}{c}{0.11 (AIAG: $" + BS + "geq$5 required)} " + BS + BS, new_ndc, "e1 NDC row")

# e1 discussion paragraph rewrite
i0 = s.find("Table~" + BS + "ref{tab:e1}: evaluation noise dominates the total")
i1 = s.find("Two readings of ``repeatability''")
assert 0 < i0 < i1, "e1 paragraph anchors"
pc = BS + "%"
newpar = (
 "Table~" + BS + "ref{tab:e1}: on the directly resampled scale, benchmark-to-benchmark differences "
 "dominate (" + str(A["V_dataset_pct"]) + pc + ") and evaluation noise contributes "
 + str(A["V_eval_direct_pct"]) + pc + "; detector-to-detector signal remains "
 + str(A["V_detector_pct"]) + pc + ". "
 "The component estimates carry ANOVA-model uncertainty; we read the thresholds qualitatively, and the "
 "verdict no longer hinges on the evaluation-noise estimate at all: shrinking $V_" + BS + "text{eval}$ "
 "twofold leaves " + BS + "%GRR at " + str(A["shrink_series"][0][1]) + pc + ", fourfold at "
 + str(A["shrink_series"][1][1]) + pc + ", eightfold at " + str(A["shrink_series"][2][1]) + pc
 + " (the series is nearly flat because evaluation noise is no longer the dominant term). "
 "Under the AIAG criteria the audited suite remains beyond the \`\`unacceptable'' boundary: with detectors "
 "as parts, " + BS + "%GRR is " + str(A["GRR_pctSV"]) + pc + " and the benchmark can distinguish "
 + str(A["NDC"]) + " detector levels---in practice, none. Nor is the verdict an artifact of the AUROC "
 "scale: the average-precision decomposition gives " + BS + "%GRR " + str(AP["GRR_pctSV"]) + pc
 + " and NDC " + str(AP["NDC"]) + "---crossing the same boundaries. ")
s = s[:i0] + newpar + s[i1:]
print("ok: e1 paragraph")

rep("Figure~\\ref{fig:vardecomp} visualizes the decomposition and its per-dataset texture: evaluation noise dominates on every series with measurable segment structure, and its absolute level varies by two orders of magnitude across benchmarks.",
    "Figure~" + BS + "ref{fig:vardecomp} visualizes the decomposition and its per-dataset texture: "
    "directly resampled evaluation noise varies by two orders of magnitude across benchmarks, with WADI "
    "the largest and SWaT structurally zero.", "fig9 body")

old_cap = ("Left: AUROC-scale variance decomposition of Table~" + BS + "ref{tab:e1}---evaluation "
           "(resampling) noise accounts for 83.0" + BS + "% of total variance; dataset differences "
           "11.4" + BS + "%; detector signal 0.5" + BS + "%.")
new_cap = ("Left: AUROC-scale variance decomposition of Table~" + BS + "ref{tab:e1} (direct "
           "resampling)---dataset differences %.1f%s, evaluation noise %.1f%s, detector signal "
           "%.1f%s." % (A["V_dataset_pct"], BS + "%", A["V_eval_direct_pct"], BS + "%",
                        A["V_detector_pct"], BS + "%"))
rep(old_cap, new_cap, "fig9 caption")

open(p, "w", encoding="utf-8").write(s)
print("2b continuation applied")

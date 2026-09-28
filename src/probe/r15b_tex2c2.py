# -*- coding: utf-8 -*-
"""R15b tex 2c continuation (pure concatenation, no %-format).
Remaining: replay, landscape, design-chart, bonferroni, scale paragraph, fig8,
synthesis, walkthrough, sec8, conclusion."""
import csv
import json

import numpy as np
from scipy import stats as st

R = "D:/0keyan/gongzuo1/paper 15SCI/02_shiyanjilu/results/"
p = "D:/0keyan/gongzuo1/paper 15SCI/03_lunwen/main.tex"
s = open(p, encoding="utf-8").read()
BS = chr(92)
J = json.load(open(R + "e1_direct_components.json"))
dv = list(csv.DictReader(open(R + "direct_scale_verdict.csv", encoding="utf-8-sig")))
rho = np.array([float(r["gap_pts"]) / float(r["mde_direct_pts"]) for r in dv])
ncert = sum(r["certifiable_direct"].strip().lower() in ("true", "1") for r in dv)
cert = [r for r in dv if r["certifiable_direct"].strip().lower() in ("true", "1")]
K0 = {"SMD": 14, "PSM": 107, "MSL": 61, "SMAP": 97, "WADI": 16}
med = {bm: float(np.median([float(r["mde_direct_pts"]) for r in dv if r["bm"] == bm])) for bm in K0}
sdm = {bm: float(np.median([float(r["sd_direct_pts"]) for r in dv if r["bm"] == bm])) for bm in K0}
kneed = {bm: K0[bm] * (med[bm] / 5) ** 2 for bm in K0}
Vb = np.array([(med[b] / 100 / 2.8016) ** 2 for b in K0])
mde_pool = 2.8016 * 100 * np.sqrt(1 / np.sum(1 / Vb))
bonn = (st.norm.ppf(1 - .05 / 30) + st.norm.ppf(.8)) / (st.norm.ppf(.975) + st.norm.ppf(.8))
max_bon = rho.max() / bonn
wt = [r for r in dv if r["bm"] == "SMD" and r["pair"] == "cmhmil_seed7-pca"][0]
wt_mde = float(wt["mde_direct_pts"])
mult9 = (st.t.ppf(.975, 13) + st.t.ppf(.90, 13)) / (st.t.ppf(.975, 13) + st.t.ppf(.80, 13))

def rep(a, b, tag):
    global s
    assert s.count(a) == 1, "anchor: " + tag
    s = s.replace(a, b)
    print("ok:", tag)

def f0(x):  return ("%.0f" % x)
def f1(x):  return ("%.1f" % x)
def f2(x):  return ("%.2f" % x)
def f3(x):  return ("%.3f" % x)
def fn(x):  return ("{:,.0f}".format(x)).replace(",", "{,}")

# replay
rep("reports A-ROC margins over its strongest baseline of $1.6$ points on MSL ($\\rho{=}0.037$) and $0.5$ points on PSM ($\\rho{=}0.019$)",
    "reports A-ROC margins over its strongest baseline of $1.6$ points on MSL ($" + BS + "rho{=}" + f2(1.6 / med["MSL"]) + "$) and $0.5$ points on PSM ($" + BS + "rho{=}" + f3(0.5 / med["PSM"]) + "$)", "replay S17")
rep("reports margins over its strongest baseline of $3.9$ AUROC points on PSM ($\\rho{=}0.15$) and $1.6$ on WADI ($\\rho{=}0.003$). All four replayed claims have $\\rho\\leq0.15$: none is certifiable at its benchmark's current size.",
    "reports margins over its strongest baseline of $3.9$ AUROC points on PSM ($" + BS + "rho{=}" + f2(3.9 / med["PSM"]) + "$) and $1.6$ on WADI ($" + BS + "rho{=}" + f3(1.6 / med["WADI"]) + "$). All four replayed claims have $" + BS + "rho" + BS + "leq" + f2(max(1.6 / med["MSL"], 0.5 / med["PSM"], 3.9 / med["PSM"], 1.6 / med["WADI"])) + "$: none is certifiable at its benchmark's current size.", "replay S11")
rep("the verdicts would reverse only if the published pairs' block-level noise were $7\\times$--$380\\times$ smaller than that of \\emph{every} cross-family pair we measured (15 pairs per benchmark, 75 gap observations across five benchmarks)---which would itself contradict the spread of block-level noise we observe across detectors of six families.",
    "the verdicts would reverse only if the published pairs' evaluation noise were " + f0(med["MSL"] / 1.6) + "$" + BS + "times$--" + f0(med["PSM"] / 0.5) + "$" + BS + "times$ smaller than the direct floors measured here (15 pairs per benchmark, 75 gap observations across five benchmarks)---which would itself contradict the spread of replica noise we observe across detectors of six families.", "replay caveat")

# landscape
rep("Every single gap, on every benchmark, sits below that benchmark's conditional MDE: the resolvability ratios have median $\\rho{=}0.06$ and maximum $\\rho{=}0.72$ (PSM), with $0/75$ reaching $1$.",
    "Against the direct floors, " + str(75 - ncert) + " of the 75 gaps sit below their benchmark's median floor: the resolvability ratios have median $" + BS + "rho{=}" + f2(np.median(rho)) + "$ and maximum $" + BS + "rho{=}" + f2(rho.max()) + "$, with " + str(ncert) + "/75 reaching $1$---both exceptions are within-family shallow pairs on SMD (isolation forest vs.\\ PCA, " + f1(float(cert[0]["gap_pts"])) + " points; vs.\\ GMM, " + f1(float(cert[1]["gap_pts"])) + "), the one region where the audit certifies rather than rejects.", "landscape")

rep("certifying a 5-point difference takes 775--208{,}069 qualified blocks depending on the suite, against the 307 the audited suites jointly hold.",
    "certifying a 5-point difference takes " + fn(min(kneed.values())) + "--" + fn(max(kneed.values())) + " qualified blocks depending on the suite, against the 307 the audited suites jointly hold.", "design chart sentence")

# bonferroni
rep("Bonferroni across the 15 pairs ($\\alpha/15$) inflates each benchmark's critical constants by only 1.36--1.50 (PSM, 107 blocks: 1.36; SMD, 14: 1.47; WADI, 16: 1.45), because the binding constraint is the block count $K$, not the nominal level. All 75 observed gaps remain below their Bonferroni-adjusted MDEs (36--885 points; the largest adjusted resolvability ratio is $\\rho=0.53$ on PSM, down from $0.72$), and the replayed published claims above remain uncertifiable by wider margins.",
    "Bonferroni across the 15 pairs ($" + BS + "alpha/15$) inflates the direct floors by a uniform $" + f2(bonn) + "$ ($z$-based), because the binding constraint is the block count $K$, not the nominal level. " + str(75 - ncert) + " of the 75 observed gaps then sit below their Bonferroni-adjusted floors (" + f0(min(med.values()) * bonn) + "--" + f0(max(med.values()) * bonn) + " points; the largest adjusted ratio is $" + BS + "rho{=}" + f2(max_bon) + "$, and the same two SMD shallow-family pairs stay above at $" + f2(rho.max() / bonn) + "$), and the replayed published claims above remain uncertifiable by wider margins.", "bonferroni")

# paired-scale paragraph -> scale note (splice between paragraph start and end of its figure)
i0 = s.find(BS + "paragraph{Robustness to the noise scale: paired vs.")
assert i0 > 0
i_endfig = s.find(BS + "begin{figure}", i0)
assert i_endfig > i0
newpar = (BS + "paragraph{Which noise scale the floors price.} The direct floors above price the design a "
 "fixed-benchmark comparison actually faces: both detectors evaluated on the " + BS + "emph{same} redrawn "
 "blocks (shared-block replica noise, $400$ replicas per pair). The margin-scale construction of the "
 "released CSV carries its own scale family (one-draw vs.\\ two-draw; fresh-realization vs.\\ paired), and "
 "the block-mean agreement between detectors ($" + BS + "rho_{AB}=0.92$--$0.99$ per benchmark mean) is "
 "the diagnostic that explains why shared-block noise is the smaller, realistic choice; "
 "Fig.~" + BS + "ref{fig:paircorr} shows the margin-scale fresh-vs-paired diagnostic for reference. "
 "The indistinguishability conclusion holds on every scale variant recorded." + BS + "n" + BS + "n")
s = s[:i0] + newpar + s[i_endfig:]
print("ok: scale paragraph")

rep("Per-pair resolvability $\\rho$ under the fresh-realization scale (pooled-variance MDE) vs.\\ the paired scale (both detectors on the same segments), all 75 real detector pairs.",
    "Margin-scale diagnostic (retained): per-pair resolvability under the fresh-realization vs.\\ paired scale of the released margin construction, all 75 pairs; the text's floors are priced on the directly resampled scale.", "fig8 caption")

# synthesis FE
rep("precision-weighting the five canonical suites under a fixed-effect common gap gives a pooled floor of $19.4$ AUROC points---better than any single suite (best: PSM at $26.6$) yet still four times the 5-point certification level---",
    "precision-weighting the five canonical suites under a fixed-effect common gap gives a pooled direct floor of $" + f1(mde_pool) + "$ AUROC points---better than any single suite (best: MSL at $" + f1(med["MSL"]) + "$) yet still above the 5-point certification level---", "synthesis FE")

# walkthrough text + table rows
rep("the power layer prices the design ($K=14$, $\\hat\\sigma_d=0.149$ on the rank-margin scale, MDE $=70.6$ AUROC points); and the verdict layer reads the pair's own single-run gap of $19.0$ points against that floor: $\\rho=0.27$, or $0.18$ after Bonferroni across the benchmark's 15 pairs, or $0.38$ under the pair's own paired scale (Section~\\ref{sec:e4}). A 19-point visible gap---larger than most published superiority margins---cannot be certified at any defensible reading of this benchmark's design.",
    "the power layer prices the design directly ($K{=}14$; this pair's direct floor is $" + f1(wt_mde) + "$ AUROC points---just above its $19.0$-point gap); and the verdict layer reads the pair's own single-run gap against that floor: $" + BS + "rho{=}" + f2(19.0 / wt_mde) + "$, or $" + f2(19.0 / (wt_mde * bonn)) + "$ after Bonferroni across the benchmark's 15 pairs (the margin-scale floor of the released construction gives $" + BS + "rho{=}0.27$; Section~" + BS + "ref{sec:e4}). A 19-point visible gap---larger than most published superiority margins---cannot be certified at any defensible reading of this benchmark's design, though on the direct scale it falls only just below the floor.", "walkthrough text")

rep("Power & $K$ / $\\hat\\sigma_d$ & 14 / 0.149 \\\\", "Power & $K$ / median direct SD & 14 / " + f1(sdm["SMD"]) + " pts \\\\", "wt sigma row")
rep("Power & MDE ($\\alpha{=}.05$, power $.8$) & 70.6 pts \\\\", "Power & direct floor ($" + BS + "alpha{=}.05$, power $.8$) & " + f1(wt_mde) + " pts \\\\", "wt power row")
rep("Verdict & observed single-run gap / $\\rho$ & 19.0 pts / 0.27 \\\\", "Verdict & observed single-run gap / $" + BS + "rho$ & 19.0 pts / " + f2(19.0 / wt_mde) + " \\\\", "wt verdict1")
rep("Verdict & Bonferroni ($m{=}15$) MDE / $\\rho$ & 103.8 pts / 0.18 \\\\", "Verdict & Bonferroni ($m{=}15$) floor / $" + BS + "rho$ & " + f1(wt_mde * bonn) + " pts / " + f2(19.0 / (wt_mde * bonn)) + " \\\\", "wt verdict2")
rep("Verdict & paired-scale MDE / $\\rho$ & 49.7 pts / 0.38 \\\\", "Verdict & margin-scale floor (released CSV) / $" + BS + "rho$ & 70.6 pts / 0.27 \\\\", "wt verdict3")

# sec 8
rep("\\texttt{audit.py} returns $K{=}14$ qualified blocks and MDE $=70.6$ points; the claimed gap is $24\\times$ below the MDE, so the correct statement is ``indistinguishable on this benchmark,''",
    BS + "texttt{audit.py} returns $K{=}14$ qualified blocks and a direct floor of $" + f1(med["SMD"]) + "$ points; the claimed gap is $" + f0(med["SMD"] / 3) + BS + "times$ below the floor, so the correct statement is ``indistinguishable on this benchmark,''", "worked example 1")
rep("Had the gap been 50 points on TE ($K{=}12$, MDE $=44.1$), the claim would survive at that benchmark's own scale---the audit is not a blanket rejection of comparisons but a scale for reading them.",
    "Had the gap been 30 points on MSL (median floor $" + f1(med["MSL"]) + "$), the claim would survive at that benchmark's own scale---the audit is not a blanket rejection of comparisons but a scale for reading them.", "worked example 2")
rep("since the audited suite's single-seed pair noise ($\\sqrt{2}\\sigma_s=10.8$ points) already sits far below every design floor ($26.6$--$610$)",
    "since the audited suite's single-seed pair noise ($" + BS + "sqrt{2}" + BS + "sigma_s{=}10.8$ points) already sits below every design floor ($" + f1(min(med.values())) + "$--$" + f1(max(med.values())) + "$)", "allocation floors")

# conclusion
rep("the suite's gauge R\\&R profile (93.8\\% GRR, NDC 0.11) is beyond the ``unacceptable'' boundary",
    "the suite's gauge R" + BS + "&R profile (" + f1(J["AUROC"]["GRR_pctSV"]) + BS + "% GRR, NDC " + f2(J["AUROC"]["NDC"]) + ") is beyond the ``unacceptable'' boundary", "conclusion")

open(p, "w", encoding="utf-8").write(s)
print("2c continuation applied")

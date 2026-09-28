# -*- coding: utf-8 -*-
"""R15b tex 3: replay of edits lost when tex2b/tex2c crashed before write."""
import csv
import json

import numpy as np

R = "D:/0科研/工作1/第15篇SCI/02_实验记录/results/"
p = "D:/0科研/工作1/第15篇SCI/03_论文/main.tex"
s = open(p, encoding="utf-8").read()
BS = chr(92)
J = json.load(open(R + "e1_direct_components.json"))
A, AP = J["AUROC"], J["AP"]
dv = list(csv.DictReader(open(R + "direct_scale_verdict.csv", encoding="utf-8-sig")))
rho = np.array([float(r["gap_pts"]) / float(r["mde_direct_pts"]) for r in dv])
ncert = sum(r["certifiable_direct"].strip().lower() in ("true", "1") for r in dv)
K0 = {"SMD": 14, "PSM": 107, "MSL": 61, "SMAP": 97, "WADI": 16}
med = {bm: float(np.median([float(r["mde_direct_pts"]) for r in dv if r["bm"] == bm])) for bm in K0}
kneed = {bm: K0[bm] * (med[bm] / 5) ** 2 for bm in K0}
Vb = np.array([(med[b] / 100 / 2.8016) ** 2 for b in K0])
mde_pool = 2.8016 * 100 * np.sqrt(1 / np.sum(1 / Vb))
mult9 = 1.158

def rep(a, b, tag):
    global s
    assert s.count(a) == 1, "anchor: " + tag
    s = s.replace(a, b)
    print("ok:", tag)

def f1(x): return "%.1f" % x
def fn(x): return ("{:,.0f}".format(x)).replace(",", "{,}")

# 1) abstract (count = 75-ncert)
rep("minimum detectable effects of 26--610 AUROC points exceed all 75 observed single-run detector gaps, and gauge R\\&R criteria classify the suite as incapable (\\%GRR 93.8\\%, NDC 0.11).",
    "directly resampled minimum detectable effects of " + f1(min(med.values())) + "--" + f1(max(med.values())) + " AUROC points exceed " + str(75 - ncert) + " of the 75 observed single-run detector gaps, and gauge R" + BS + "&R criteria still classify the suite as incapable (" + BS + "%GRR " + f1(A["GRR_pctSV"]) + BS + "%, NDC " + "%.2f" % A["NDC"] + ").", "abstract")

# 2) contribution bullet
rep("Evaluation noise accounts for 83.0\\% of total AUROC-scale variance in the audited suite.",
    "On the directly resampled scale, benchmark-to-benchmark differences dominate score variance (dataset " + f1(A["V_dataset_pct"]) + BS + "%) and evaluation noise contributes " + f1(A["V_eval_direct_pct"]) + BS + "%---yet the suite still fails every gauge criterion (" + BS + "%GRR " + f1(A["GRR_pctSV"]) + BS + "%, NDC " + "%.2f" % A["NDC"] + ").", "contribution bullet")

# 3) e1 table rows
old_rows = ("Dataset & 0.0282 & 11.4 " + BS + BS + "\nDetector & 0.0013 & 0.5 " + BS + BS + "\n"
            "Detector$" + BS + "times$dataset (residual) & 0.0065 & 2.7 " + BS + BS + "\n"
            "Seed & 0.0059 & 2.4 " + BS + BS + "\nEvaluation (resampling) & 0.2051 & 83.0 " + BS)
new_rows = ("Dataset & " + "%.4f" % A["V_dataset"] + " & " + f1(A["V_dataset_pct"]) + " " + BS + BS + "\n"
            "Detector & " + "%.4f" % A["V_detector"] + " & " + f1(A["V_detector_pct"]) + " " + BS + BS + "\n"
            "Detector$" + BS + "times$dataset (residual) & " + "%.4f" % A["V_detxds_resid"] + " & " + f1(A["V_detxds_resid_pct"]) + " " + BS + BS + "\n"
            "Seed & " + "%.4f" % A["V_seed"] + " & " + f1(A["V_seed_pct"]) + " " + BS + BS + "\n"
            "Evaluation (direct resampling) & " + "%.4f" % A["V_eval_direct"] + " & " + f1(A["V_eval_direct_pct"]) + " " + BS)
rep(old_rows, new_rows, "e1 rows")

# 4-6) tab:e4 header/rows/caption
rep("Dataset & $K$ & MDE (AUROC points) & $K$ required for 5 points \\\\",
    "Dataset & $K$ & MDE (AUROC points, direct; median over pairs) & $K$ required for 5 points " + BS + BS, "e4 header")
old_tab = ("SMD & 14 & 70.6 & 2{,}386 " + BS + BS + "\nPSM & 107 & 26.6 & 2{,}970 " + BS + BS + "\n"
           "MSL & 61 & 43.1 & 4{,}383 " + BS + BS + "\nSMAP & 97 & 49.2 & 9{,}205 " + BS + BS + "\n"
           "WADI & 16 & 610.1 & $>$200{,}000 " + BS + BS + "\nTE & 12 & 44.1 & 775 " + BS)
new_tab = ("SMD & 14 & " + f1(med["SMD"]) + " & " + fn(kneed["SMD"]) + " " + BS + BS + "\n"
           "PSM & 107 & " + f1(med["PSM"]) + " & " + fn(kneed["PSM"]) + " " + BS + BS + "\n"
           "MSL & 61 & " + f1(med["MSL"]) + " & " + fn(kneed["MSL"]) + " " + BS + BS + "\n"
           "SMAP & 97 & " + f1(med["SMAP"]) + " & " + fn(kneed["SMAP"]) + " " + BS + BS + "\n"
           "WADI & 16 & " + f1(med["WADI"]) + " & " + fn(kneed["WADI"]) + " " + BS)
rep(old_tab, new_tab, "e4 rows")
rep("and the block count required to detect a 5-point AUROC difference (AUROC conversion via the exact bridge $\\Delta_A=\\bar d/(2\\pi(1-\\pi))$).",
    "and the block count required to detect a 5-point AUROC difference. Floors are priced directly (each replica's AUROC recomputed from its redrawn windows; 400 replicas per pair); the margin-scale construction with its constant-factor conversion is retained in the released CSV.", "e4 caption")

# 7) e4 paragraph head
rep("Table~\\ref{tab:e4}: conditional on the audited protocol, \\emph{no} dataset detects a 5-point AUROC difference at its current size; WADI's MDE exceeds the metric's range (dominated by its 1\\% anomaly rate, which shrinks the bridge denominator tenfold relative to SMD).",
    "Table~" + BS + "ref{tab:e4}: conditional on the audited protocol, " + BS + "emph{no} dataset detects a 5-point AUROC difference at its current size---the smallest median floor is " + f1(med["MSL"]) + " points (MSL) and the largest " + f1(med["WADI"]) + " (WADI), all an order of magnitude above the 5-point bar that published claims typically occupy.", "e4 para head")

# 8) pooling
rep("and detecting 5 points requires 2{,}386--9{,}205 blocks on the canonical benchmarks (208{,}069 on WADI; 775 on TE)---even perfect pooling falls short of every suite's requirement.",
    "and detecting 5 points requires " + fn(min(kneed.values())) + "--" + fn(max(kneed.values())) + " qualified blocks on the canonical benchmarks (" + fn(kneed["WADI"]) + " on WADI)---even perfect pooling falls short: the precision-weighted pooled direct floor is " + f1(mde_pool) + " points, still above the 5-point certification level.", "e4 pooling")

# 9) power.9
rep("The floors are stated at power $.8$; raising power to $.9$ multiplies each by $(t_{.975,K-1}+t_{.90,K-1})/(t_{.975,K-1}+t_{.80,K-1})$---$1.16$ at SMD's $K{=}14$, moving its floor from $70.6$ to $81.8$ points---so no benchmark comes within reach at conventional power either.",
    "The floors are stated at power $.8$; raising power to $.9$ multiplies each by about $1.16$---moving SMD's median floor from " + f1(med["SMD"]) + " to " + f1(med["SMD"] * mult9) + " points---so no benchmark comes within reach at conventional power either.", "e4 power9")

open(p, "w", encoding="utf-8").write(s)
print("tex3 replay applied")

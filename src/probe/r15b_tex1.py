# -*- coding: utf-8 -*-
"""R15b tex part 1: bridge disclosure (sec 4), AP-link, anatomy scale note."""
BS = chr(92)
p = r"D:\0科研\工作1\第15篇SCI\03_论文\main.tex"
s = open(p, encoding="utf-8").read()

def rep(a, b, tag):
    global s
    assert s.count(a) == 1, "anchor: " + tag
    s = s.replace(a, b)
    print("ok:", tag)

# --- sec 4 bridge paragraph: replica-level convention disclosure ---
rep("This makes the audit's power statements interpretable in the field's native metric.",
    "This makes the audit's power statements interpretable in the field's native metric. "
    "One scope rule matters and was sharpened in revision: the identity holds for "
    "\\emph{full-run} comparisons, where each side's ranks are computed within its own complete sample "
    "(including the paired single-run gaps of Section~\\ref{sec:e4}, where direct AUROC differences and "
    "bridged differences agree to machine precision). Segment-block replicas reuse the original run's "
    "rank values, and at replica level the constant-factor conversion is a convention, not an identity: "
    "across 15 cells ${\\times}$ 400 replicas, reused-rank conversions of replica AUROCs show median "
    "errors of $9$--$153$ points and variance inflated by factors of $4$--$1600$ relative to directly "
    "recomputed replica AUROCs (released CSV), and adversarial block compositions can break the "
    "conversion entirely. All noise-scale quantities of Sections~\\ref{sec:e1} and~\\ref{sec:e4} are "
    "therefore priced \\emph{directly}---each replica's AUROC (or AP) is recomputed from its redrawn "
    "windows---while the bridge is retained only where it is exact (full-run point differences) and as "
    "the scale convention of the released margin-scale construction.", "bridge disclosure")

# --- AP paragraph link sentence (same phenomenon) ---
rep("AP-scale floors therefore cannot inherit the bridge and must be priced by direct per-cell resampling, which the released engine performs.",
    "AP-scale floors therefore cannot inherit the bridge and must be priced by direct per-cell resampling, "
    "which the released engine performs. The same replica-level failure extends to \\emph{any} "
    "fixed-margin conversion, including the constant AUROC factor itself (Section~4); this is why every "
    "floor in this paper is now priced directly from recomputed replica metrics.", "AP link")

# --- anatomy 0.70 sentence: scale reference now CSV ---
rep("---$0.70\\times$ the audited fresh-realization scale of Table~\\ref{tab:e4}, exactly the paired/fresh ratio of Section~\\ref{sec:e4}'s robustness analysis---",
    "---$0.70\\times$ the margin-scale fresh floor of the released construction ($\\hat\\sigma_d{=}0.149$), "
    "matching the paired/fresh ratio of the margin-scale robustness analysis---", "anatomy scale note")

open(p, "w", encoding="utf-8").write(s)
print("part 1 applied")

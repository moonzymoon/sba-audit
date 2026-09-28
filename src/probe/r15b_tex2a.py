# -*- coding: utf-8 -*-
"""R15b tex part 2a: e1 method sentence -> direct recomputation (framework text)."""
p = r"D:\0科研\工作1\第15篇SCI\03_论文\main.tex"
s = open(p, encoding="utf-8").read()
BS = chr(92)

a = ("and evaluation noise $V_{\\text{eval}}$ from the segment-block resampling of "
     "Section~\\ref{sec:e3method} mapped to the AUROC scale by "
     "$V_{\\text{eval}} = V_{\\text{clean}}/(2\\pi(1-\\pi))^2$.")
b = ("and evaluation noise $V_{\\text{eval}}$ estimated \\emph{directly}: each segment-block replica's "
     "AUROC (and AP) is recomputed from its redrawn windows, and $V_{\\text{eval}}$ is the per-dataset "
     "replica variance averaged over the eight seed-rich datasets and the scorers with released caches "
     "(300--400 replicas per cell; the earlier constant-factor conversion of the margin-scale "
     "construction is retained in the released CSV as a cross-check, Section~4).")
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("e1 method sentence -> direct")

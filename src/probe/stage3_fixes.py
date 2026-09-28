# -*- coding: utf-8 -*-
"""STAGE 3 deterministic fixes S1,S2,S4,S6,S7,S9,S10,S11 to main.tex."""
BS = chr(92)
p = r"D:\0keyan\gongzuo1\paper 15SCI\03_lunwen\main.tex"
s = open(p, encoding="utf-8").read()
n0 = len(s)

def rep(a, b, tag):
    global s
    assert s.count(a) == 1, "anchor not unique/found: " + tag
    s = s.replace(a, b)
    print("ok:", tag)

# S1 broken ref (already absent in current tex; conditional)
if "Section~ef{sec:e2}" in s:
    rep("Section~ef{sec:e2}", "Section~" + BS + "ref{sec:e2}", "S1 broken ref")
else:
    print("skip: S1 (broken ref already absent in current tex; reviewer saw older build)")

# S2 seed-variance caption
rep("$V_{\\text{seed}}$ is the mean of per-dataset seed variances (range $0.0004$--$0.0057$, dominated by PSM).",
    "$V_{\\text{seed}}$ is the mean of per-dataset seed variances over the six seed-rich datasets "
    "(per-dataset range $0.0004$--$0.0224$, the maximum being SWaT's).".replace("$V_{\\text{seed}}$",
    "$V_{\\text{seed}}$"), "S2 caption")

# S4a Prop 3 remark: MC vs enumeration + cost
rep("Second, the reference has $2^K$ sign assignments, so on few-block series its null distribution is coarse and the test can be conservative---consistent with the observed $0.000$--$0.053$ size range of this reference across all computable cells.",
    "Second, the reference has $2^K$ sign assignments, so on few-block series its null distribution is "
    "coarse and the test can be conservative---consistent with the observed $0.000$--$0.053$ size range of "
    "this reference across all computable cells. Third, on implementation: for $K{\\le}20$ the released "
    "engine enumerates all $2^K$ assignments exactly; beyond that it uses Monte-Carlo sign-flipping with "
    "$m{=}499$ random assignments, whose $p$-value is the exchangeability-valid empirical tail fraction "
    "$(1+\\#{\\rm more\\ extreme})/(m{+}1)$---exactness is claimed only for the enumerated version, and the "
    "two agree wherever both are computable.", "S4a Prop3 remark")

# S4b Algorithm 1 line
rep("\\State apply battery: window $t$/Wilcoxon on $d_{1:T}$; block $t$/Wilcoxon on segment means of $d$; sign-permutation reference (Proposition~3)",
    "\\State apply battery: window $t$/Wilcoxon on $d_{1:T}$; block $t$/Wilcoxon on segment means of $d$; sign-permutation reference (Proposition~3; exact enumeration for $K{\\le}20$, Monte-Carlo sign-flipping $m{=}499$ beyond)", "S4b Algorithm 1")

# S4c cost sentence in recommendations
rep("plus a one-off $O(2^K K)$ enumeration per cell for the exact sign-permutation reference---the per-comparison audit a reviewer would run is of order minutes.",
    "plus $O(2^K K)$ exact enumeration of the sign-permutation reference for $K{\\le}20$ (Monte-Carlo sign-flipping, $O(mK)$ with $m{=}499$, beyond)---the per-comparison audit a reviewer would run is of order minutes.", "S4c cost")

# S6 ARE fix: sign-flipping permutation on means is asymptotically t-equivalent (ARE 1);
# 2/pi belongs to the sign-count test (not used); Wilcoxon leg keeps hodges1956.
rep("The reference trades power for exactness (asymptotic relative efficiency $2/\\pi\\approx0.64$ against the $t$ test under normality---near the bottom of the realistic range for block-mean differences, since heavier tails raise it toward $2$, and the battery's Wilcoxon leg stays above $0.864$ of the $t$ test for every error distribution~\\citep{hodges1956}); the calibrated block-level $t$ and Wilcoxon regain it where their assumptions hold, which is why the battery reports both.",
    "On efficiency: the reference permutes signs of the \\emph{mean} statistic, and a sign-flipping "
    "permutation test of a mean is asymptotically equivalent to the $t$ test under normality (efficiency "
    "tending to $1$; \\citealp{hoeffding1952}); its finite-$K$ price is the grid coarseness already noted, "
    "not an asymptotic loss. The battery's Wilcoxon leg stays above $0.864$ of the $t$ test for every "
    "error distribution~\\citep{hodges1956}---constants such as $2/\\pi$ belong to sign-\\emph{count} "
    "tests, which the battery does not use. The calibrated block-level $t$ and Wilcoxon therefore cost "
    "nothing asymptotically and regain finite-sample power where their assumptions hold, which is why the "
    "battery reports all three.", "S6 ARE")

# S7a walkthrough wording
rep("A 19-point visible gap---larger than most published superiority margins---is noise at any defensible reading of this benchmark.",
    "A 19-point visible gap---larger than most published superiority margins---cannot be certified at any defensible reading of this benchmark's design.", "S7a walkthrough")

# S7b recommended-phrasing quote
rep("``on this set of data, detector differences below the MDE cannot be distinguished from noise; any ranking among them is noise.''",
    "``on this set of data, the design cannot resolve detector differences of this size at the planned error rates; any ranking among them is uncertifiable at this design.''", "S7b phrasing")

# S7c no-equivalence disclaimer after the MDE paragraph intro (append to conditional-phrasing sentence)
rep("The phrasing we adopt---and recommend---is conditional:",
    "Sub-MDE gaps are a statement about this design's resolving power, not evidence that the population difference is zero; equivalence would require a pre-specified margin and an interval-based test, which this audit does not attempt (its TOST band concerns variance-component calibration, a different object). The phrasing we adopt---and recommend---is conditional:", "S7c equivalence")

# S9a Lemma 1 scope
rep("The de-facto protocol's apparent advantage is an illusion of degrees of freedom ($T{-}1$ instead of $K{-}1$); with an honest variance, that advantage vanishes identically.",
    "The de-facto protocol's apparent advantage is an illusion of degrees of freedom ($T{-}1$ instead of $K{-}1$); with an honest variance, that advantage vanishes identically. The identity is algebraic and holds \\emph{up to the choice of variance estimator}---different consistent estimators (plug-in, HAC, CR1, bootstrap) yield the same numerator with different variances and hence different degrees of freedom and $p$-values, which is precisely the CR1-vs-segment difference of Section~\\ref{sec:e3}.", "S9a Lemma scope")

# S9b Corollary 1 scope
rep("every design-calibrated studentization carries at most $K{-}C{-}p$ degrees of freedom---the ceiling is a property of clustered replication, not of anomaly detection.",
    "every studentization in the design-based class---those assembled from within-type pool variances of cluster summaries---carries at most $K{-}C{-}p$ degrees of freedom; beating the ceiling requires within-block model assumptions (Proposition~2(iv)). The ceiling is a property of clustered replication, not of anomaly detection.", "S9b Corollary scope")

# S10 e1 double-count disclosure (append to tab:e1 discussion paragraph, after shrinkage sentence)
rep("Nor is the verdict an artifact of the AUROC scale:",
    "One structural overlap should be disclosed: the balanced table's cells are single realizations for five of the six detectors, so its interaction residual partly contains their single-run evaluation noise, which is then also added as $V_{\\text{eval}}$---a bounded double-count that we keep visible rather than re-orthogonalize (with $V_{\\text{eval}}$ dominating, the overlap does not change any verdict reading). Nor is the verdict an artifact of the AUROC scale:", "S10 disclosure")

# S11 anonymity defense
rep("we maintain for this paper (29 entries, 10 fully verified; Section~\\ref{sec:e6})",
    "we maintain for this paper (29 entries, 10 fully verified; Section~\\ref{sec:e6}; the entry-to-paper mapping is withheld by design to keep the audit non-adversarial, and the aggregate counts are reproducible from the released protocol)", "S11 anonymity")

open(p, "w", encoding="utf-8").write(s)
print("all edits applied; size %d -> %d" % (n0, len(s)))

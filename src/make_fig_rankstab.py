# -*- coding: utf-8 -*-
"""R38: fig10 rank stability figure (dual panel)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = r"D:\0keyan\gongzuo1\paper15\03_lunwen\figs"
d = json.load(open(os.path.join(r"D:\0keyan\gongzuo1\paper15\02_shiyanjilu\results", "r38_rankstab.json")))
BLUE, COBALT, O = "#2c7fb8", "#1c4f81", "#e08214"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
ms = sorted(int(m) for m in d["suite_bootstrap"])
taus = [d["suite_bootstrap"][str(m)]["tau_med"] for m in ms]
iqr_lo = [d["suite_bootstrap"][str(m)]["tau_iqr"][0] for m in ms]
iqr_hi = [d["suite_bootstrap"][str(m)]["tau_iqr"][1] for m in ms]
ax1.plot(ms, taus, "o-", color=COBALT, lw=1.6)
ax1.fill_between(ms, iqr_lo, iqr_hi, color=COBALT, alpha=0.15)
ax1.axhline(0.9, color="gray", ls="--", lw=0.8)
ax1.set_xlabel("suites sampled ($m$)")
ax1.set_ylabel("median Kendall $\\tau$")
ax1.set_title("Rank stability vs suites sampled", fontsize=9)

# seed axis: cmhmil median rank
seed_ms = sorted(int(s) for s in d["seed_axis"])
meds = [d["seed_axis"][str(s)]["median"] for s in seed_ms]
cils = [d["seed_axis"][str(s)]["ci"][0] for s in seed_ms]
cihs = [d["seed_axis"][str(s)]["ci"][1] for s in seed_ms]
ax2.plot(seed_ms, meds, "s-", color=O, lw=1.6)
ax2.fill_between(seed_ms, cils, cihs, color=O, alpha=0.15)
ax2.set_xlabel("retraining seeds used ($s$)")
ax2.set_ylabel("deep detector rank")
ax2.set_title("Deep-detector rank vs seeds", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig10_rankstab.png"), dpi=600)
print("fig10 written")

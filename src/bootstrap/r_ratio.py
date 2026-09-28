"""R zh:zh_8961value (pre-registered §7.3): R = V_seed-pair / V_clean.

V_clean:  N2 (s, s*)   10000 zh:zh_6249 d̄* empiricalzh:zh_6254difference (E3-clean zh:zh_6595cache);
V_seed-pair: samedatasetzh:zh_3990realzh:seedpair d̄  empiricalzh:zh_6254difference (3 pairzh:zh_5104is zh:zh_449value, pre-registeredzh:zh_3017).
zh:zh_6254point = blockzh:zh_4542paireddifferencemean zh:zh_6254difference (rankzh:zh_405, zh:zh_6924 R1).
"""
from __future__ import annotations

import os

import numpy as np

RCACHE = r"D:/0keyan/gongzuo1/paper 15SCI/_resample_cache"


def compute_R(dbar_clean, dbar_seeds):
    """R = Var(d̄_seed-pair) / Var(d̄*_clean). zh:zh_6507is zh:zh_4920meanzh:statisticszh_1031array."""
    dbar_clean = np.asarray(dbar_clean, float)
    dbar_seeds = np.asarray(dbar_seeds, float)
    if len(dbar_seeds) < 2 or len(dbar_clean) < 2:
        return float("nan"), float("nan"), float("nan")
    v_seed = float(np.var(dbar_seeds, ddof=1))
    v_clean = float(np.var(dbar_clean, ddof=1))
    r = v_seed / v_clean if v_clean > 0 else float("inf")
    return r, v_seed, v_clean


def load_clean_dbar(dataset, scorer="cmhmil_seed7"):
    f = os.path.join(RCACHE, f"e3clean_{dataset}_{scorer}.npz")
    if not os.path.exists(f):
        return None
    return np.load(f)["dbar"]


def r_report(pairs_dbar, dataset, scorer="cmhmil_seed7"):
    dbar_clean = load_clean_dbar(dataset, scorer)
    if dbar_clean is None:
        return dict(dataset=dataset, R=np.nan, V_seed=np.nan, V_clean=np.nan,
                    note="E3-clean cachezh:missing")
    r, vs, vc = compute_R(dbar_clean, pairs_dbar)
    return dict(dataset=dataset, R=r, V_seed=vs, V_clean=vc, n_pairs=len(pairs_dbar),
                note="3 pairis zh:zh_449value" if len(pairs_dbar) == 3 else "")

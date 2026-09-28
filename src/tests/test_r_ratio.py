"""R zh:zh_8961valuecomputezh_3208test (zh:zh_5303segment1-2 zh:zh_7103)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bootstrap.r_ratio import compute_R  # noqa: E402


def test_r_recovers_known_ratio():
    rng = np.random.default_rng(0)
    # zh:construction V_seed = 4, V_clean = 1 (zh:zh_5949 R≈4; zh:zh_6792difference)
    dbar_seeds = rng.normal(0, 2.0, size=20000)   # var 4
    dbar_clean = rng.normal(0, 1.0, size=20000)   # var 1
    r, vs, vc = compute_R(dbar_clean, dbar_seeds)
    assert abs(vs - 4.0) < 0.1 and abs(vc - 1.0) < 0.05
    assert abs(r - 4.0) < 0.15


def test_r_degenerate_inputs():
    r, vs, vc = compute_R(np.array([0.1, 0.2]), np.array([0.5]))
    assert np.isnan(r), "zh:zh_8166seedpairzh:zh_2483returns NaN zh:zh_4489"


def test_r_zero_clean_variance():
    r, _, _ = compute_R(np.zeros(100), np.array([1.0, 2.0, 3.0]))
    assert np.isinf(r), "V_clean=0 zh:zh_5104 R=inf (zh:zh_2724)"

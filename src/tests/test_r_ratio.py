"""R 比值计算单元测试 (阶段1-2 交付要求)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bootstrap.r_ratio import compute_R  # noqa: E402


def test_r_recovers_known_ratio():
    rng = np.random.default_rng(0)
    # 构造 V_seed = 4, V_clean = 1 (期望 R≈4; 大样本容差)
    dbar_seeds = rng.normal(0, 2.0, size=20000)   # var 4
    dbar_clean = rng.normal(0, 1.0, size=20000)   # var 1
    r, vs, vc = compute_R(dbar_clean, dbar_seeds)
    assert abs(vs - 4.0) < 0.1 and abs(vc - 1.0) < 0.05
    assert abs(r - 4.0) < 0.15


def test_r_degenerate_inputs():
    r, vs, vc = compute_R(np.array([0.1, 0.2]), np.array([0.5]))
    assert np.isnan(r), "单一种子对应返回 NaN 而非崩溃"


def test_r_zero_clean_variance():
    r, _, _ = compute_R(np.zeros(100), np.array([1.0, 2.0, 3.0]))
    assert np.isinf(r), "V_clean=0 时 R=inf (除零防护)"

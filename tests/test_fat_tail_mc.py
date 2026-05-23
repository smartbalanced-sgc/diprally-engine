"""Tests for W9 PR #48 — fat-tail Monte Carlo innovations.

Verifies:
  - _draw_innovations produces unit-variance draws for both normal
    and Student-t paths (so σ-input interpretation is preserved).
  - Student-t draws have fatter tails than normal at matched σ
    (99.5th percentile of |z| is meaningfully higher).
  - run_mc_joint_conditional accepts distribution / df params and
    produces paths whose terminal-return distribution is fatter-
    tailed than the normal counterpart.
  - Config validation rejects df ≤ 2 (variance undefined).
  - Backward compat: default distribution="normal" matches prior
    behavior (when explicitly passed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import MC_DISTRIBUTION
from src.math_utils import _draw_innovations, run_mc_joint_conditional


def test_normal_draws_have_unit_variance():
    rng = np.random.default_rng(seed=1)
    z = _draw_innovations(rng, 100_000, 60, distribution="normal")
    # Sample variance ≈ 1.0 for N(0,1) at this sample count.
    assert abs(z.std() - 1.0) < 0.01


def test_student_t_draws_rescaled_to_unit_variance():
    """Raw Student-t(df) has variance df/(df-2); _draw_innovations
    rescales so the σ-scaling in GBM keeps its interpretation."""
    rng = np.random.default_rng(seed=2)
    for df in (3.5, 5.0, 7.0, 10.0):
        z = _draw_innovations(rng, 200_000, 60, distribution="student_t", df=df)
        # Should be within 3% of unit variance at this sample count.
        # df=3.5 is heavy-tailed → wider sampling error so loosen.
        tol = 0.05 if df <= 4 else 0.03
        assert abs(z.std() - 1.0) < tol, (
            f"df={df}: std={z.std()} not unit"
        )


def test_student_t_has_fatter_tails_than_normal():
    """At matched unit variance, Student-t(5) should have a higher
    99.5th percentile of |z| than the normal — that's the whole point
    of using it for fat-tail MC."""
    rng = np.random.default_rng(seed=3)
    z_n = _draw_innovations(rng, 500_000, 1, distribution="normal")
    z_t = _draw_innovations(rng, 500_000, 1, distribution="student_t", df=5.0)
    p995_n = np.percentile(np.abs(z_n), 99.5)
    p995_t = np.percentile(np.abs(z_t), 99.5)
    # Normal's 99.5th |z| ≈ 2.81; t(5)'s ≈ 4.03 → ratio ~1.4
    assert p995_t > p995_n * 1.20, (
        f"t(5) 99.5th pct {p995_t:.3f} not meaningfully fatter than "
        f"normal's {p995_n:.3f}"
    )


def test_lower_df_has_fatter_tails():
    """t(3) has heavier tails than t(7) at same scaling."""
    rng = np.random.default_rng(seed=4)
    z_3 = _draw_innovations(rng, 500_000, 1, distribution="student_t", df=3.5)
    z_7 = _draw_innovations(rng, 500_000, 1, distribution="student_t", df=7.0)
    p99_3 = np.percentile(np.abs(z_3), 99.0)
    p99_7 = np.percentile(np.abs(z_7), 99.0)
    assert p99_3 > p99_7, (
        f"df=3.5 99th pct {p99_3:.3f} should exceed df=7 {p99_7:.3f}"
    )


def test_invalid_df_raises():
    rng = np.random.default_rng(seed=5)
    with pytest.raises(ValueError):
        _draw_innovations(rng, 100, 10, distribution="student_t", df=2.0)
    with pytest.raises(ValueError):
        _draw_innovations(rng, 100, 10, distribution="student_t", df=1.5)


def test_unknown_distribution_raises():
    rng = np.random.default_rng(seed=6)
    with pytest.raises(ValueError):
        _draw_innovations(rng, 100, 10, distribution="cauchy")


def test_mc_paths_fatter_terminal_under_student_t():
    """Compare terminal return distributions between normal and
    Student-t with everything else identical. The t-distribution path
    should produce a heavier-tailed terminal return."""
    paths_n = run_mc_joint_conditional(
        S0=100.0, sigma=0.50, mu=0.05, horizon_days=60, n_paths=50_000,
        seed=42, distribution="normal",
    )
    paths_t = run_mc_joint_conditional(
        S0=100.0, sigma=0.50, mu=0.05, horizon_days=60, n_paths=50_000,
        seed=42, distribution="student_t", df=5.0,
    )
    # Terminal returns.
    r_n = (paths_n[:, -1] / 100.0) - 1.0
    r_t = (paths_t[:, -1] / 100.0) - 1.0
    # Both should have roughly similar central tendency (σ-input
    # preserved), but t's 99th-percentile drawdown is deeper.
    p1_n = np.percentile(r_n, 1)
    p1_t = np.percentile(r_t, 1)
    assert p1_t < p1_n, (
        f"t(5) 1st-pctile terminal return {p1_t:.3f} should be "
        f"more negative than normal's {p1_n:.3f}"
    )


def test_normal_distribution_path_reproducible():
    """Deterministic given seed."""
    paths_a = run_mc_joint_conditional(
        S0=100.0, sigma=0.30, mu=0.0, horizon_days=20, n_paths=1000,
        seed=99, distribution="normal",
    )
    paths_b = run_mc_joint_conditional(
        S0=100.0, sigma=0.30, mu=0.0, horizon_days=20, n_paths=1000,
        seed=99, distribution="normal",
    )
    np.testing.assert_array_equal(paths_a, paths_b)


def test_student_t_path_reproducible():
    paths_a = run_mc_joint_conditional(
        S0=100.0, sigma=0.30, mu=0.0, horizon_days=20, n_paths=1000,
        seed=99, distribution="student_t", df=5.0,
    )
    paths_b = run_mc_joint_conditional(
        S0=100.0, sigma=0.30, mu=0.0, horizon_days=20, n_paths=1000,
        seed=99, distribution="student_t", df=5.0,
    )
    np.testing.assert_array_equal(paths_a, paths_b)


def test_yaml_config_loads_with_per_class_df():
    """YAML-loaded config: EXTREME < HIGH < MID for df (heavier tails
    on more-volatile classes)."""
    cfg = MC_DISTRIBUTION
    assert cfg.default in ("normal", "student_t")
    if cfg.default == "student_t":
        assert cfg.default_df > 2.0
    extreme = cfg.per_class.get("EXTREME", cfg.default_df)
    high = cfg.per_class.get("HIGH", cfg.default_df)
    mid = cfg.per_class.get("MID", cfg.default_df)
    # All must be > 2 (variance condition).
    assert extreme > 2.0 and high > 2.0 and mid > 2.0
    # Heavier tails on EXTREME — lower df.
    assert extreme <= high <= mid


def test_config_rejects_df_le_2():
    """Pydantic + post-init must reject Student-t df ≤ 2."""
    from src.config import MCDistributionConfig
    with pytest.raises(ValueError):
        MCDistributionConfig(default="student_t", default_df=2.0,
                              per_class={})
    with pytest.raises(ValueError):
        MCDistributionConfig(default="student_t", default_df=5.0,
                              per_class={"EXTREME": 1.5})


def test_config_rejects_unknown_default():
    from src.config import MCDistributionConfig
    with pytest.raises(ValueError):
        MCDistributionConfig(default="laplace", default_df=5.0,
                              per_class={})

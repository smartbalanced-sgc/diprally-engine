"""Tests for src/ambiguity.py — W4 PR #28.

The ambiguity score is the broker's sort key (PR #29). It must:
  - Peak when math is genuinely "on the edge" (AI could flip decision)
  - Drop low when math gives a clear qualify or clear refusal
  - Stay in [0, 1] under all input combinations
  - Handle no-pair runs gracefully (None inputs)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ambiguity import WEIGHTS, AmbiguityScore, compute_ambiguity


def _baseline(**overrides):
    """Baseline 'on-the-edge' inputs — every component returns ~0.5 to
    1.0 so we can perturb one at a time and observe the response."""
    defaults = dict(
        best_p_dip=0.65,                  # right at MID conviction threshold
        conviction_dip=0.65,
        best_ev_pct_of_dip=0.005,         # right at 50bps EV hurdle
        sigma_divergence_pp=20.0,         # at full-scale → saturates to 1.0
        method_max_delta_pp=4.0,          # right at refuse threshold (peaks)
        method_refuse_threshold_pp=4.0,
        mom_30d=-0.25,                    # right at sacred #14 trend threshold
    )
    defaults.update(overrides)
    return defaults


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_returns_score_object():
    score = compute_ambiguity(**_baseline())
    assert isinstance(score, AmbiguityScore)
    assert 0.0 <= score.overall <= 1.0
    assert set(score.components) == set(WEIGHTS)


def test_on_the_edge_yields_high_ambiguity():
    """All five components at peak → overall should be ~1.0."""
    score = compute_ambiguity(**_baseline())
    assert score.overall >= 0.95, f"on-the-edge score {score.overall} too low"


def test_clearly_qualifies_yields_low_ambiguity():
    """p_dip far above threshold, EV way above hurdle, σ tight, math
    agrees, momentum healthy — nothing for AI to add."""
    score = compute_ambiguity(
        best_p_dip=0.95,                  # +30pp above 0.65 threshold
        conviction_dip=0.65,
        best_ev_pct_of_dip=0.10,          # 1000 bps — way above 50bps hurdle
        sigma_divergence_pp=0.5,          # tight
        method_max_delta_pp=0.1,          # methods agree
        method_refuse_threshold_pp=4.0,
        mom_30d=0.10,                     # +10% — far from -25% trend filter
    )
    # All five components should be near 0 → overall near 0.
    assert score.overall < 0.10, f"clearly-qualifies score {score.overall} too high"


def test_clearly_refused_yields_low_ambiguity():
    """p_dip far below threshold, EV way below hurdle, mom_30d
    catastrophic — the call is clean (no), AI input can't flip it."""
    score = compute_ambiguity(
        best_p_dip=0.30,                  # -35pp below 0.65 threshold
        conviction_dip=0.65,
        best_ev_pct_of_dip=-0.05,         # -500 bps — way below 50bps hurdle
        sigma_divergence_pp=0.5,
        method_max_delta_pp=0.1,
        method_refuse_threshold_pp=4.0,
        mom_30d=-0.50,                    # -50% — well past -25%, in panic territory
    )
    assert score.overall < 0.10, f"clearly-refused score {score.overall} too high"


def test_no_pair_handled():
    """When math finds no candidate pair, None inputs default the
    conviction/EV components to 0.5 — moderate ambiguity (we don't
    know if AI would flip it)."""
    score = compute_ambiguity(
        best_p_dip=None,
        conviction_dip=0.65,
        best_ev_pct_of_dip=None,
        sigma_divergence_pp=2.0,
        method_max_delta_pp=1.0,
        method_refuse_threshold_pp=4.0,
        mom_30d=0.0,
    )
    assert 0.0 <= score.overall <= 1.0
    assert score.components["conviction_proximity"] == 0.5
    assert score.components["ev_hurdle_proximity"] == 0.5


def test_overall_stays_in_unit_interval_under_extreme_inputs():
    """Pathological inputs shouldn't push the score out of [0, 1]."""
    for case in (
        dict(best_p_dip=10.0, conviction_dip=0.65),
        dict(best_p_dip=-5.0, conviction_dip=0.65),
        dict(sigma_divergence_pp=10000.0),
        dict(method_max_delta_pp=10000.0, method_refuse_threshold_pp=0.001),
        dict(mom_30d=-10.0),
    ):
        inputs = _baseline(**case)
        score = compute_ambiguity(**inputs)
        assert 0.0 <= score.overall <= 1.0, (
            f"out-of-range overall {score.overall} for case {case}"
        )


def test_conviction_proximity_has_highest_weight():
    """Conviction is the most decision-critical signal — weight must
    exceed every other component."""
    others = {k: v for k, v in WEIGHTS.items() if k != "conviction_proximity"}
    assert WEIGHTS["conviction_proximity"] > max(others.values())


def test_zero_refuse_threshold_doesnt_crash():
    """Defensive: if method_refuse_threshold_pp is 0 (degenerate case
    when σ → 0), the proximity component should fall back to 0 rather
    than ZeroDivisionError."""
    score = compute_ambiguity(
        best_p_dip=0.65,
        conviction_dip=0.65,
        best_ev_pct_of_dip=0.005,
        sigma_divergence_pp=0.0,
        method_max_delta_pp=0.0,
        method_refuse_threshold_pp=0.0,
        mom_30d=0.0,
    )
    assert score.components["method_proximity"] == 0.0
    assert 0.0 <= score.overall <= 1.0

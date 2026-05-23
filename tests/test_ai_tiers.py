"""Tests for src/ai_tiers.py — W4 PR #27 tier resolution.

Sacred CLAUDE.md tier ladder:
  T0: math only ($0.00)
  T1: Haiku Pass 1 only (~$0.02)
  T2: Sonnet Pass 1 + Sonnet Pass 2 (~$0.10)
  T3: Opus Pass 1 + Sonnet Pass 2 + Haiku stress (~$0.30)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ai_tiers import resolve_tier, t0
from src.config import AI_TIERS, MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET


def test_all_four_tiers_resolve():
    """T0 / T1 / T2 / T3 must all be present."""
    for name in ("T0", "T1", "T2", "T3"):
        spec = resolve_tier(name)
        assert spec.name == name


def test_t0_is_math_only():
    spec = resolve_tier("T0")
    assert spec.pass1_model is None
    assert spec.pass2_model is None
    assert spec.stress_model is None
    assert spec.catalyst_verification_model is None
    assert spec.estimated_cost_usd == 0.0
    assert spec.runs_ai is False


def test_t1_is_haiku_pass1_only():
    spec = resolve_tier("T1")
    assert spec.pass1_model == MODEL_HAIKU
    assert spec.pass2_model is None
    assert spec.stress_model is None
    # W6 PR #33: T1's single Pass-1 web_search call rarely surfaces
    # catalysts specific enough to warrant verification.
    assert spec.catalyst_verification_model is None
    assert spec.pass1_web_search_max == 1
    assert spec.runs_ai is True


def test_t2_is_sonnet_pass1_plus_pass2_with_verification():
    spec = resolve_tier("T2")
    assert spec.pass1_model == MODEL_SONNET
    assert spec.pass2_model == MODEL_SONNET
    assert spec.stress_model is None
    # W6 PR #33: catalyst verification runs at T2+.
    assert spec.catalyst_verification_model == MODEL_HAIKU
    assert spec.runs_ai is True


def test_t3_is_opus_sonnet_haiku_full_stack_with_verification():
    """T3 must have the deepest stack: Opus Pass 1 + Sonnet Pass 2 +
    Haiku stress + Haiku verification. PR #52: web_search reduced
    from 5 to 3 (rarely-hit upper bound; Pass 1 already gathers
    10+ sources from 3 searches)."""
    spec = resolve_tier("T3")
    assert spec.pass1_model == MODEL_OPUS
    assert spec.pass2_model == MODEL_SONNET
    assert spec.stress_model == MODEL_HAIKU
    assert spec.catalyst_verification_model == MODEL_HAIKU
    # PR #52 — web_search reduced 5→3. Sanity-check it stays > 1
    # (Pass 1 needs at least one search to gather catalysts).
    assert spec.pass1_web_search_max >= 2
    assert spec.runs_ai is True


def test_tier_costs_monotonic():
    """T0 < T1 < T2 < T3 (budget allocator depends on monotonicity)."""
    costs = [resolve_tier(t).estimated_cost_usd for t in ("T0", "T1", "T2", "T3")]
    assert costs == sorted(costs)
    assert costs[0] == 0.0
    assert costs[3] >= 0.10  # T3 nontrivial


def test_unknown_tier_raises():
    with pytest.raises(KeyError):
        resolve_tier("T4")


def test_t0_shorthand_matches_resolve():
    assert t0() == resolve_tier("T0")


def test_runs_ai_property_consistent():
    """runs_ai is True iff at least Pass 1 model is set."""
    for name in ("T0", "T1", "T2", "T3"):
        spec = resolve_tier(name)
        assert spec.runs_ai == (spec.pass1_model is not None)


def test_daily_budget_cap_loads():
    """Sacred $2/day cap is configurable but defaults to 2.00."""
    from src.config import AI_DAILY_BUDGET_CAP_USD
    assert AI_DAILY_BUDGET_CAP_USD > 0.0
    assert AI_DAILY_BUDGET_CAP_USD == 2.00

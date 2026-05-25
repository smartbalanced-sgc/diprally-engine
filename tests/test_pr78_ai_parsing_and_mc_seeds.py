"""Tests for PR #78 — audit findings #7, #9, #10.

#7  Pass 2 `missing_catalysts_added` contaminates key_risks.
    Pre-fix: parse_ai_pass2 emitted key_risks =
        [primary_critique_str, catalyst_dict_1, catalyst_dict_2, ...]
    Downstream `validate_pass2_critique(critique_text=pass2.key_risks[0])`
    silently treated catalyst dicts beyond [0] as critique text in the
    reporter's loop (printed `{...}` as a key-risk line). Plus
    `missing_catalysts_added` was already merged into `revised_catalysts`
    above — double-counted.
    Fix: key_risks now holds ONLY the primary_critique string.

#9  CSV `horizon_days` empty string → silent row skip.
    Pre-fix: `int(row.get("horizon_days", DEFAULT))` returns the default
    only when the KEY is missing; an empty-string CELL raises
    ValueError, caught by the surrounding broad except, dropping the
    row from backtest aggregation invisibly.
    Fix: `int(row.get("horizon_days") or DEFAULT)` coerces empty / None
    cleanly.

#10 Shared MC seeds across sensitivity scenarios.
    Pre-fix: analyze_joint_conditional defaulted to seeds 42/43 for
    its internal bridge-touch RNG. Sensitivity-table scenarios share
    the same path-generation seed via the OUTER MC call
    (run_mc_joint_conditional(seed=42+i)), but the bridge correction
    INSIDE analyze_joint_conditional used the constant 42/43 — all
    scenarios saw identical bridge randomness, defeating scenario
    independence.
    Fix: `seed` is now a parameter of analyze_joint_conditional and
    threaded from the caller in compute_sensitivity_table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# Finding #7 — Pass 2 key_risks no longer holds catalyst dicts
# =============================================================================

def test_parse_ai_pass2_key_risks_excludes_missing_catalysts_added():
    """key_risks must be a list of STRINGS (just primary_critique).
    `missing_catalysts_added` belongs in catalysts, not key_risks."""
    from src.ai_layer import parse_ai_pass2
    from src.engine import AIPassOutput

    pass1 = AIPassOutput(
        pass_number=1, drift_estimate=0.10, drift_range=(-0.10, 0.30),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.05, raw_sources_cited=3,
    )
    raw = {
        "revised_drift_estimate": 0.05,
        "revised_confidence": "LOW",
        "revised_vol_regime": "HIGH",
        "primary_critique": "Pass 1 underweighted earnings risk.",
        "missing_catalysts_added": [
            {"name": "Q2 earnings", "magnitude": "high",
             "direction_risk": "bearish",
             "date_or_window": "2026-07-25"},
        ],
        "agreement_with_pass1": "partial_disagree",
        "revision_reasoning": "x",
    }
    pass2 = parse_ai_pass2(raw, pass1, cost=0.04)

    # key_risks holds ONLY the critique string.
    assert all(isinstance(k, str) for k in pass2.key_risks)
    assert len(pass2.key_risks) == 1
    assert "earnings risk" in pass2.key_risks[0]

    # missing_catalysts_added is reflected in revised_catalysts (the
    # already-existing merge from lines 663-667), NOT in key_risks.
    assert any(
        isinstance(c, dict) and c.get("name") == "Q2 earnings"
        for c in pass2.catalysts
    )


def test_parse_ai_pass2_empty_primary_critique_yields_empty_key_risks():
    """If Pass 2 doesn't emit a critique string, key_risks is just []
    (was [""] previously — empty-string sentinel that the reporter
    would render as an empty bullet line)."""
    from src.ai_layer import parse_ai_pass2
    from src.engine import AIPassOutput

    pass1 = AIPassOutput(
        pass_number=1, drift_estimate=0.10, drift_range=(-0.10, 0.30),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.05, raw_sources_cited=3,
    )
    raw = {
        "revised_drift_estimate": 0.05,
        "primary_critique": "",  # explicit empty
    }
    pass2 = parse_ai_pass2(raw, pass1, cost=0.04)
    assert pass2.key_risks == []


def test_validate_pass2_critique_gets_clean_string():
    """End-to-end: the engine's `pass2.key_risks[0]` lookup must yield
    a plain string ready for `validate_pass2_critique`. Pre-fix it
    could be a dict if primary_critique was missing and
    missing_catalysts_added wasn't (the [0] slot would be a dict)."""
    from src.ai_layer import parse_ai_pass2
    from src.engine import AIPassOutput

    pass1 = AIPassOutput(
        pass_number=1, drift_estimate=0.10, drift_range=(-0.10, 0.30),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.05, raw_sources_cited=3,
    )
    # No primary_critique, but missing_catalysts_added present →
    # pre-fix would have made key_risks[0] a catalyst dict.
    raw = {
        "revised_drift_estimate": 0.05,
        "missing_catalysts_added": [
            {"name": "rumor", "direction_risk": "bullish"},
        ],
    }
    pass2 = parse_ai_pass2(raw, pass1, cost=0.04)
    critique_text = pass2.key_risks[0] if pass2.key_risks else ""
    assert isinstance(critique_text, str)


# =============================================================================
# Finding #9 — horizon_days empty-string coercion
# =============================================================================

def test_build_per_day_status_handles_empty_horizon_days_cell():
    """Row with horizon_days="" must NOT be silently skipped — it
    should fall back to DEFAULT_HORIZON_DAYS like a missing key.

    Exercises `_build_per_day_status` (one of the two int() sites);
    same `or DEFAULT` fix applies to `run_backtest_layer`'s loop.
    """
    from datetime import date, timedelta
    from src.engine import _build_per_day_status, DEFAULT_HORIZON_DAYS

    pred_old = (date.today() - timedelta(days=200)).strftime("%Y-%m-%d")
    rows = [
        # Row 1: empty-string horizon_days cell (pre-fix → silently dropped)
        {"date": pred_old, "spot": "100", "recommended_dip": "95",
         "recommended_rally": "110", "p_round_trip": "0.5",
         "horizon_days": ""},
        # Row 2: explicit value
        {"date": pred_old, "spot": "100", "recommended_dip": "95",
         "recommended_rally": "110", "p_round_trip": "0.5",
         "horizon_days": str(DEFAULT_HORIZON_DAYS)},
        # Row 3: key missing entirely (always worked — default kicks in)
        {"date": pred_old, "spot": "100", "recommended_dip": "95",
         "recommended_rally": "110", "p_round_trip": "0.5"},
    ]
    out = _build_per_day_status(rows, current_price=100.0)
    # All three rows survive the int() coercion. Pre-fix row 1 raised
    # ValueError on `int("")` and was caught by the broad except.
    assert len(out) == 3


# =============================================================================
# Finding #10 — sensitivity scenarios get independent bridge RNG
# =============================================================================

def test_analyze_joint_conditional_seed_parameter_changes_bridge_output():
    """Calling analyze_joint_conditional with different seeds on the
    SAME paths must produce different bridge-touch days (and thus
    slightly different scenario partitions). Pre-fix the bridge always
    used seed=42/43 regardless of caller intent."""
    from src.math_utils import (
        analyze_joint_conditional,
        run_mc_joint_conditional,
    )
    S0 = 100.0
    sigma = 0.30
    mu = 0.0
    horizon = 60
    paths = run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu,
        horizon_days=horizon, n_paths=5000, seed=100,
    )
    r_a = analyze_joint_conditional(
        paths, S0, dip_price=92.0, rally_price=108.0,
        horizon_days=horizon, sigma=sigma, seed=42,
    )
    r_b = analyze_joint_conditional(
        paths, S0, dip_price=92.0, rally_price=108.0,
        horizon_days=horizon, sigma=sigma, seed=99,
    )
    # Same paths, different bridge seed → bridge-corrected touch counts
    # differ (small but non-zero). At least ONE of the four partition
    # probabilities must differ to confirm the seed actually threaded
    # through.
    keys = ("p_round_trip", "p_bag_hold", "p_no_trade_rally_first", "p_neither")
    assert any(r_a[k] != r_b[k] for k in keys), (
        f"bridge seed had no effect — A={r_a}, B={r_b}"
    )


def test_analyze_joint_conditional_seed_default_unchanged():
    """Default seed=42 preserves prior behavior for callers that don't
    pass the new arg (backward compat)."""
    from src.math_utils import (
        analyze_joint_conditional,
        run_mc_joint_conditional,
    )
    paths = run_mc_joint_conditional(
        S0=100.0, sigma=0.30, mu=0.0,
        horizon_days=60, n_paths=2000, seed=7,
    )
    r_default = analyze_joint_conditional(
        paths, 100.0, dip_price=92.0, rally_price=108.0,
        horizon_days=60, sigma=0.30,
    )
    r_explicit = analyze_joint_conditional(
        paths, 100.0, dip_price=92.0, rally_price=108.0,
        horizon_days=60, sigma=0.30, seed=42,
    )
    for k in ("p_round_trip", "p_bag_hold", "p_no_trade_rally_first", "p_neither"):
        assert r_default[k] == r_explicit[k]

"""Tests for signal_from_fundamentals — W6 PR #34.

Pure-function tests against the YAML-loaded thresholds. The data-fetch
side (fetch_fundamentals from FMP) is exercised in integration smoke
runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import SIGNAL_FUNDAMENTALS
from src.signals import signal_from_fundamentals


def _f(fcf=None, leverage=None, margin_trend=None):
    """Build a fundamentals dict matching fetch_fundamentals's shape."""
    n = sum(1 for x in (fcf, leverage, margin_trend) if x is not None)
    return {
        "fcf_yield": fcf,
        "net_debt_to_ebitda": leverage,
        "op_margin_trend": margin_trend,
        "n_components_available": n,
    }


def test_none_fundamentals_returns_none_signal():
    result = signal_from_fundamentals(None)
    assert result["drift"] is None
    assert result["confidence"] == "LOW"


def test_empty_fundamentals_returns_none_signal():
    result = signal_from_fundamentals(_f())
    assert result["drift"] is None
    assert result["confidence"] == "LOW"


def test_high_fcf_low_leverage_improving_margins_is_strong_bullish():
    """Triple positive: cash-generative, deleveraged, margins expanding.
    Combined drift should be solidly positive (capped at drift_cap_abs)."""
    result = signal_from_fundamentals(_f(
        fcf=0.08, leverage=0.5, margin_trend=0.08,
    ))
    assert result["drift"] > 0
    assert result["confidence"] == "HIGH"
    assert "FCF strong+" in result["notes"]
    assert "leverage low+" in result["notes"]
    assert "improving" in result["notes"]


def test_negative_fcf_high_leverage_deteriorating_is_bearish():
    """Triple negative: cash-burning, over-levered, margins compressing."""
    result = signal_from_fundamentals(_f(
        fcf=-0.08, leverage=6.0, margin_trend=-0.08,
    ))
    assert result["drift"] < 0
    assert result["confidence"] == "HIGH"
    assert "strong-" in result["notes"]


def test_pre_revenue_name_with_only_fcf_returns_low_confidence():
    """EXTREME / pre-revenue: only FCF yield computable (EBITDA ≤ 0,
    no 8Q of revenue). One sub-component → LOW confidence."""
    result = signal_from_fundamentals(_f(fcf=-0.10))
    assert result["drift"] is not None
    assert result["drift"] < 0  # negative FCF yield → bearish
    assert result["confidence"] == "LOW"


def test_drift_capped_at_configured_max():
    """Even with extreme inputs, combined drift can't exceed drift_cap_abs."""
    result = signal_from_fundamentals(_f(
        fcf=0.50, leverage=0.1, margin_trend=0.30,
    ))
    cap = SIGNAL_FUNDAMENTALS.drift_cap_abs
    assert result["drift"] <= cap
    assert result["drift"] >= -cap


def test_confidence_ladder_matches_n_components():
    one = signal_from_fundamentals(_f(fcf=0.03))
    two = signal_from_fundamentals(_f(fcf=0.03, leverage=2.0))
    three = signal_from_fundamentals(_f(fcf=0.03, leverage=2.0, margin_trend=0.0))
    assert one["confidence"] == "LOW"
    assert two["confidence"] == "MEDIUM"
    assert three["confidence"] == "HIGH"


def test_neutral_band_produces_zero_drift():
    """All three sub-components in their neutral zone → 0 drift."""
    result = signal_from_fundamentals(_f(
        fcf=0.01,        # in neutral band (between 0 and mild_bull 0.02)
        leverage=2.0,    # within neutral leverage band
        margin_trend=0.0,
    ))
    assert result["drift"] == 0.0
    assert result["confidence"] == "HIGH"  # all three known, just neutral


def test_fcf_yield_boundary_inclusive_strong_bull():
    """Exactly at fcf_yield_strong_bull → bullish_drift_pp contribution."""
    bull = SIGNAL_FUNDAMENTALS.fcf_yield_strong_bull
    result = signal_from_fundamentals(_f(fcf=bull))
    assert result["drift"] is not None
    assert result["drift"] > 0


def test_leverage_threshold_high_yields_strong_bearish():
    """Net debt / EBITDA above leverage_mild_bear → bullish_drift_pp
    bearish (full magnitude, not mild)."""
    high_lev = SIGNAL_FUNDAMENTALS.leverage_mild_bear + 1.0
    result = signal_from_fundamentals(_f(leverage=high_lev))
    assert result["drift"] is not None
    # Magnitude matches bullish_drift_pp (full bearish), not mild.
    assert abs(result["drift"]) == pytest.approx(
        SIGNAL_FUNDAMENTALS.bullish_drift_pp, abs=1e-9
    )


def test_signal_does_not_mutate_input():
    """Pure function — fundamentals dict unchanged after call."""
    f = _f(fcf=0.03, leverage=2.0)
    before = dict(f)
    signal_from_fundamentals(f)
    assert f == before

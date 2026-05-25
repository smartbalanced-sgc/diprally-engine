"""Tests for W8 PR #49 — portfolio correlation gate.

Pure-function tests with synthetic return histories. The gate's job
is to refuse SUBSTITUTE recommendations — when two tickers in the
accepted set have ρ ≥ threshold, the lower-EV one gets dropped.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.portfolio import (
    GateResult,
    PortfolioRecommendation,
    _pearson,
    format_gate_result,
    gate_by_correlation,
)


def _history(returns, start_price=100.0):
    """Build a synthetic price-history DataFrame from a list of daily
    log returns. Length(returns) bars → length+1 closes."""
    closes = [start_price]
    for r in returns:
        closes.append(closes[-1] * np.exp(r))
    dates = pd.bdate_range(end="2026-05-22", periods=len(closes))
    return pd.DataFrame({"Date": dates, "Close": closes})


def _rec(ticker, ev_bps, returns):
    """Build a PortfolioRecommendation with synthetic returns."""
    return PortfolioRecommendation(
        ticker=ticker, ev_bps=ev_bps,
        history_df=_history(returns),
    )


def test_empty_input_returns_empty_result():
    result = gate_by_correlation([])
    assert result.accepted == []
    assert result.dropped == {}


def test_single_ticker_always_accepted():
    rng = np.random.default_rng(seed=1)
    rec = _rec("INTC", 75.0, rng.normal(0, 0.02, 80).tolist())
    result = gate_by_correlation([rec])
    assert result.accepted == ["INTC"]
    assert result.dropped == {}


def test_uncorrelated_tickers_both_accepted():
    """Two independent return series at ρ ≈ 0 → both pass."""
    rng = np.random.default_rng(seed=2)
    recs = [
        _rec("INTC", 75.0, rng.normal(0, 0.02, 80).tolist()),
        _rec("ASTS", 60.0, rng.normal(0, 0.03, 80).tolist()),
    ]
    result = gate_by_correlation(recs)
    assert set(result.accepted) == {"INTC", "ASTS"}
    assert result.dropped == {}


def test_perfectly_correlated_pair_drops_lower_ev():
    """Two identical return histories at ρ=1.0 → only the higher-EV
    ticker survives."""
    rng = np.random.default_rng(seed=3)
    rets = rng.normal(0, 0.02, 80).tolist()
    recs = [
        _rec("HI_EV", 100.0, rets),
        _rec("LO_EV", 50.0, rets),  # IDENTICAL returns
    ]
    result = gate_by_correlation(recs, threshold=0.85, window_days=60)
    assert "HI_EV" in result.accepted
    assert "LO_EV" in result.dropped
    assert "HI_EV" in result.dropped["LO_EV"]
    assert "ρ=1.00" in result.dropped["LO_EV"]


def test_strongly_correlated_dropped():
    """Construct a pair at ρ ≈ 0.95 — should be dropped at default
    threshold 0.85."""
    rng = np.random.default_rng(seed=4)
    common = rng.normal(0, 0.02, 80)
    # ticker B = 0.95 * common + 0.05 * noise → ρ ≈ 0.95
    noise = rng.normal(0, 0.02, 80)
    a_rets = common.tolist()
    b_rets = (0.95 * common + 0.05 * noise).tolist()
    recs = [
        _rec("A", 100.0, a_rets),
        _rec("B",  50.0, b_rets),
    ]
    result = gate_by_correlation(recs, threshold=0.85, window_days=60)
    assert "A" in result.accepted
    assert "B" in result.dropped


def test_weakly_correlated_accepted():
    """Pair at ρ ≈ 0.30 → both accepted at default threshold 0.85."""
    rng = np.random.default_rng(seed=5)
    common = rng.normal(0, 0.02, 80)
    noise = rng.normal(0, 0.02, 80)
    b_rets = (0.30 * common + 0.95 * noise).tolist()  # ρ ≈ 0.30
    recs = [
        _rec("A", 100.0, common.tolist()),
        _rec("B",  50.0, b_rets),
    ]
    result = gate_by_correlation(recs, threshold=0.85, window_days=60)
    assert set(result.accepted) == {"A", "B"}
    assert result.dropped == {}


def test_threshold_boundary_inclusive():
    """ρ exactly at threshold → drops (inclusive >=)."""
    rng = np.random.default_rng(seed=6)
    common = rng.normal(0, 0.02, 80).tolist()
    recs = [
        _rec("A", 100.0, common),
        _rec("B",  50.0, common),  # ρ = 1.0
    ]
    # Set threshold = 1.0 exactly.
    result = gate_by_correlation(recs, threshold=1.0, window_days=60)
    assert "A" in result.accepted
    assert "B" in result.dropped


def test_ev_ordering_determines_priority():
    """In a correlated trio (A,B,C), highest EV survives."""
    rng = np.random.default_rng(seed=7)
    common = rng.normal(0, 0.02, 80).tolist()
    recs = [
        _rec("MID", 60.0, common),
        _rec("HI",  90.0, common),
        _rec("LO",  20.0, common),
    ]
    result = gate_by_correlation(recs, threshold=0.85, window_days=60)
    assert result.accepted == ["HI"]  # only the top-EV one
    assert "MID" in result.dropped
    assert "LO" in result.dropped


def test_alphabetical_tiebreak_at_equal_ev():
    """Equal EV → alphabetical wins (deterministic)."""
    rng = np.random.default_rng(seed=8)
    common = rng.normal(0, 0.02, 80).tolist()
    recs = [
        _rec("ZEBRA",  75.0, common),
        _rec("ALPHA",  75.0, common),  # ALPHA first alphabetically
    ]
    result = gate_by_correlation(recs, threshold=0.85, window_days=60)
    assert result.accepted == ["ALPHA"]
    assert "ZEBRA" in result.dropped


def test_insufficient_history_accepted_defensively():
    """Ticker with <window_days bars → accepted (gate can't disprove
    correlation, give benefit of the doubt to math layer)."""
    rng = np.random.default_rng(seed=9)
    # Only 5 bars — less than 60d window.
    short_rets = rng.normal(0, 0.02, 5).tolist()
    long_rets = rng.normal(0, 0.02, 80).tolist()
    recs = [
        _rec("LONG",  100.0, long_rets),
        _rec("SHORT", 50.0,  short_rets),
    ]
    result = gate_by_correlation(recs, threshold=0.85, window_days=60)
    assert "LONG" in result.accepted
    assert "SHORT" in result.accepted  # defensive accept


def test_enabled_false_passes_through():
    """Master switch off → all input accepted in input order, no gating."""
    rng = np.random.default_rng(seed=10)
    common = rng.normal(0, 0.02, 80).tolist()
    recs = [
        _rec("A", 100.0, common),
        _rec("B",  50.0, common),  # would drop if enabled
    ]
    result = gate_by_correlation(recs, enabled=False)
    assert result.accepted == ["A", "B"]
    assert result.dropped == {}


def test_three_independent_clusters():
    """Three pairs of correlated tickers across 3 sectors — should
    keep one representative from each cluster."""
    rng = np.random.default_rng(seed=11)
    semi_common = rng.normal(0, 0.02, 80)
    space_common = rng.normal(0, 0.025, 80)
    storage_common = rng.normal(0, 0.018, 80)
    recs = [
        # Semi cluster
        _rec("INTC", 80.0, semi_common.tolist()),
        _rec("AMD",  60.0, (semi_common + rng.normal(0, 0.005, 80)).tolist()),
        # Space cluster
        _rec("RKLB", 70.0, space_common.tolist()),
        _rec("ASTS", 50.0, (space_common + rng.normal(0, 0.005, 80)).tolist()),
        # Storage cluster
        _rec("STX",  65.0, storage_common.tolist()),
        _rec("WDC",  40.0, (storage_common + rng.normal(0, 0.005, 80)).tolist()),
    ]
    result = gate_by_correlation(recs, threshold=0.85, window_days=60)
    # Higher-EV in each cluster should survive.
    assert "INTC" in result.accepted
    assert "RKLB" in result.accepted
    assert "STX" in result.accepted
    assert "AMD" in result.dropped
    assert "ASTS" in result.dropped
    assert "WDC" in result.dropped


def test_pearson_handles_constant_series():
    """All-zero variance → corrcoef returns nan; helper coerces to 0."""
    a = np.array([0.01, 0.01, 0.01, 0.01])
    b = np.array([0.02, 0.03, 0.01, 0.05])
    assert _pearson(a, b) == 0.0


def test_format_gate_result_contains_all_tickers():
    """Smoke test the formatter."""
    rng = np.random.default_rng(seed=12)
    common = rng.normal(0, 0.02, 80).tolist()
    recs = [
        _rec("HI", 100.0, common),
        _rec("LO",  50.0, common),
    ]
    gate = gate_by_correlation(recs, threshold=0.85, window_days=60)
    out = format_gate_result(gate, recs)
    assert "HI" in out
    assert "LO" in out
    assert "Accepted" in out
    assert "Dropped" in out


def test_default_config_loads():
    """YAML-loaded config has sensible default values."""
    from src.config import PORTFOLIO_GATE
    assert isinstance(PORTFOLIO_GATE.enabled, bool)
    assert 0.0 < PORTFOLIO_GATE.correlation_threshold < 1.0
    assert 20 <= PORTFOLIO_GATE.correlation_window_days <= 252

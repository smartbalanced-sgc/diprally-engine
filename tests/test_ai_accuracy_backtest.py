"""Tests for the AI accuracy backtest harness — synth data only.

End-to-end with real AI requires FMP + ANTHROPIC keys and is run by the
operator locally. These unit tests lock the math: as-of filtering,
scoring, date selection. So a future refactor that silently breaks the
as-of replay can't slip past CI.
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


# =============================================================================
# Synth price history fixture
# =============================================================================

def _synth_history(start: date, n_days: int, drift_annual: float = 0.0,
                    sigma_annual: float = 0.30, seed: int = 42):
    """Generate a synth daily-OHLC pd.DataFrame with `n_days` business
    days starting at `start`. GBM with the given annualized drift/σ.
    Includes high/low columns derived from close × ±0.5σ_daily."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    s_daily = sigma_annual * np.sqrt(dt)
    z = rng.standard_normal(n_days)
    log_returns = (drift_annual - 0.5 * sigma_annual ** 2) * dt + s_daily * z
    closes = 100.0 * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range(start=start, periods=n_days).date
    highs = closes * np.exp(0.5 * s_daily)
    lows = closes * np.exp(-0.5 * s_daily)
    return pd.DataFrame({
        "date": [d.isoformat() for d in dates],
        "close": closes, "high": highs, "low": lows,
    })


# =============================================================================
# As-of price truncation
# =============================================================================

def test_truncate_history_to_strips_post_as_of():
    from tools.diag.ai_accuracy_backtest import truncate_history_to
    df = _synth_history(date(2026, 1, 5), 60)
    as_of = date(2026, 2, 15)
    out = truncate_history_to(df, as_of)
    # Every retained row must have date ≤ as_of
    for d in pd.to_datetime(out["date"]).dt.date:
        assert d <= as_of
    # The strictly-greater-than rows should be the complement
    assert len(out) + sum(pd.to_datetime(df["date"]).dt.date > as_of) == len(df)


def test_forward_prices_from_returns_strictly_after():
    from tools.diag.ai_accuracy_backtest import forward_prices_from
    df = _synth_history(date(2026, 1, 5), 80)
    as_of = date(2026, 2, 15)
    fwd = forward_prices_from(df, as_of, n_trading_days=20)
    # All rows must be strictly after as_of
    for d in pd.to_datetime(fwd["date"]).dt.date:
        assert d > as_of
    assert len(fwd) <= 20


def test_truncate_raises_on_missing_date_col():
    from tools.diag.ai_accuracy_backtest import truncate_history_to
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="date"):
        truncate_history_to(df, date.today())


# =============================================================================
# Bundle as-of filtering
# =============================================================================

def test_filter_bundle_lists_to_as_of_prunes_correctly():
    from tools.diag.ai_accuracy_backtest import filter_bundle_lists_to_as_of
    bundle = {
        "ticker": "MU",
        "as_of": "2026-05-31",
        "pt_revisions_90d": [
            {"date": "2026-05-28", "firm": "DA Davidson", "action": "raise"},
            {"date": "2026-05-10", "firm": "UBS",         "action": "raise"},
            {"date": "2026-03-15", "firm": "Mizuho",      "action": "raise"},
        ],
        "pt_revisions_90d_count": 3,
        "pt_revisions_90d_raise_cut_ratio": "3 raises / 0 cuts",
        "grade_changes_90d": [
            {"date": "2026-05-20", "firm": "BofA", "action": "maintain"},
            {"date": "2026-05-30", "firm": "MS",   "action": "upgrade"},
        ],
        "grade_actions_summary": {
            "upgrade": 1, "downgrade": 0, "maintain": 1, "initiate": 0,
        },
        "recent_news_30d": [
            {"date": "2026-05-29", "title": "post-cutoff news"},
            {"date": "2026-05-15", "title": "in-window news"},
        ],
        "recent_news_30d_count": 2,
    }
    out = filter_bundle_lists_to_as_of(bundle, date(2026, 5, 25))
    # PT raises: keep only May 10 + March 15 (dropped May 28)
    assert len(out["pt_revisions_90d"]) == 2
    assert out["pt_revisions_90d_count"] == 2
    assert out["pt_revisions_90d_raise_cut_ratio"] == "2 raises / 0 cuts"
    # Grade changes: keep only BofA maintain (dropped MS upgrade)
    assert len(out["grade_changes_90d"]) == 1
    assert out["grade_actions_summary"]["upgrade"] == 0
    assert out["grade_actions_summary"]["maintain"] == 1
    # News: keep only the May-15 in-window item
    assert len(out["recent_news_30d"]) == 1
    assert out["recent_news_30d"][0]["title"] == "in-window news"
    assert out["recent_news_30d_count"] == 1
    # as_of field updated
    assert out["as_of"] == "2026-05-25"
    # Original bundle UNMUTATED
    assert len(bundle["pt_revisions_90d"]) == 3
    assert bundle["recent_news_30d_count"] == 2


def test_filter_bundle_handles_unparseable_dates_gracefully():
    """If a row has a malformed date, the filter drops it by default
    (default_keep=False) rather than crashing."""
    from tools.diag.ai_accuracy_backtest import filter_bundle_lists_to_as_of
    bundle = {
        "as_of": "2026-05-31",
        "pt_revisions_90d": [
            {"date": "not-a-date", "firm": "X"},
            {"date": None, "firm": "Y"},
            {"date": "2026-05-15", "firm": "Z"},
        ],
    }
    out = filter_bundle_lists_to_as_of(bundle, date(2026, 5, 20))
    # Only Z (May 15, valid date, ≤ as_of) survives
    assert len(out["pt_revisions_90d"]) == 1
    assert out["pt_revisions_90d"][0]["firm"] == "Z"


def test_filter_bundle_empty_lists_clean_up_derived():
    """When filtering empties a list, derived counts / summaries clear
    out rather than reporting stale numbers."""
    from tools.diag.ai_accuracy_backtest import filter_bundle_lists_to_as_of
    bundle = {
        "as_of": "2026-05-31",
        "pt_revisions_90d": [{"date": "2026-05-28", "action": "raise"}],
        "pt_revisions_90d_count": 1,
        "pt_revisions_90d_raise_cut_ratio": "1 raises / 0 cuts",
    }
    out = filter_bundle_lists_to_as_of(bundle, date(2026, 5, 25))
    # Single entry was May 28, after May 25 → all dropped.
    assert out["pt_revisions_90d"] == []
    assert out["pt_revisions_90d_count"] == 0
    # raise_cut_ratio should be gone (no entries to ratio)
    assert "pt_revisions_90d_raise_cut_ratio" not in out


# =============================================================================
# As-of state recomputation
# =============================================================================

def test_compute_as_of_state_uses_truncated_data():
    """Computed spot / σ / RSI / mom_30d must reflect the as-of view,
    not the latest. Tests the basic invariant: spot at as_of_date equals
    the close on that day."""
    from tools.diag.ai_accuracy_backtest import compute_as_of_state
    df = _synth_history(date(2026, 1, 5), 100, drift_annual=0.20, sigma_annual=0.4)
    as_of = date(2026, 3, 15)
    state = compute_as_of_state(df, as_of)
    # Spot must equal the actual close on (or just before) the as-of date.
    truncated_dates = pd.to_datetime(df["date"]).dt.date
    last_idx = (truncated_dates <= as_of).values.nonzero()[0][-1]
    expected_spot = df.iloc[last_idx]["close"]
    assert state["spot"] == pytest.approx(expected_spot, rel=1e-9)
    # Sigma must be a finite positive number (realized 30d annualized).
    assert state["sigma_blended"] > 0
    assert state["sigma_blended"] < 5.0  # 500% σ is implausible — bound check
    # mom_30d / mom_5d / rsi should be finite
    for k in ("rsi", "mom_5d", "mom_30d", "ytd_return"):
        assert np.isfinite(state[k])


def test_compute_as_of_state_raises_on_insufficient_history():
    from tools.diag.ai_accuracy_backtest import compute_as_of_state
    df = _synth_history(date(2026, 5, 1), 10)
    # Asking for state with only ~10 days of history before the date —
    # we need ≥35 to compute realized vol + RSI.
    with pytest.raises(ValueError, match="Insufficient history"):
        compute_as_of_state(df, date(2026, 5, 13))


# =============================================================================
# Scoring math
# =============================================================================

def test_score_prediction_direction_match_bullish():
    """AI predicts +30% annualized; forward 20d returns +5% (= ~+63%
    annualized). Sign matches; direction_match = 1."""
    from tools.diag.ai_accuracy_backtest import score_prediction
    forward = pd.DataFrame({
        "close": np.linspace(100, 105, 20),
        "high":  np.linspace(101, 106, 20),
        "low":   np.linspace(99, 104, 20),
    })
    out = score_prediction(
        predicted_drift_annual=0.30, spot_at_as_of=100.0, forward_df=forward,
    )
    assert out["direction_match"] == 1
    assert out["terminal_return"] == pytest.approx(0.05, rel=1e-6)
    # Realized drift = 5% × (252/20) = 63%
    assert out["realized_drift_ann"] == pytest.approx(0.63, rel=1e-3)
    assert out["hit_rally_5pct"] == 1
    assert out["hit_dip_5pct"] == 0


def test_score_prediction_direction_miss():
    """AI predicts +30%; forward goes -3% (= -38% annualized). Sign
    mismatch; direction_match = 0."""
    from tools.diag.ai_accuracy_backtest import score_prediction
    forward = pd.DataFrame({
        "close": np.linspace(100, 97, 20),
        "high":  np.linspace(100, 97, 20),
        "low":   np.linspace(100, 97, 20),
    })
    out = score_prediction(
        predicted_drift_annual=0.30, spot_at_as_of=100.0, forward_df=forward,
    )
    assert out["direction_match"] == 0
    assert out["realized_drift_ann"] < 0
    # Overshoot ratio: predicted/realized = 0.30 / -0.38 ≈ -0.79
    assert out["overshoot_ratio"] < 0


def test_score_prediction_no_forward_data_returns_nones():
    from tools.diag.ai_accuracy_backtest import score_prediction
    out = score_prediction(
        predicted_drift_annual=0.10, spot_at_as_of=100.0,
        forward_df=pd.DataFrame({"close": []}),
    )
    assert out["n_forward"] == 0
    assert out["direction_match"] is None
    assert out["realized_drift_ann"] is None


# =============================================================================
# Date selection
# =============================================================================

def test_pick_default_as_of_dates_respects_forward_horizon():
    """Auto-picked dates must each have ≥20 trading days of forward
    history available."""
    from tools.diag.ai_accuracy_backtest import pick_default_as_of_dates
    df = _synth_history(date(2026, 1, 5), 200)
    dates = pick_default_as_of_dates(df, lookback_td=60, forward_td=20, spacing_td=5)
    available_dates = pd.to_datetime(df["date"]).dt.date.tolist()
    last_available = available_dates[-1]
    for d in dates:
        # Every picked as-of must be ≤ last_available - 20 trading days
        idx_d = available_dates.index(d)
        assert idx_d + 20 <= len(available_dates) - 1, (
            f"{d}: not enough forward history"
        )


def test_pick_default_as_of_dates_returns_empty_on_short_history():
    from tools.diag.ai_accuracy_backtest import pick_default_as_of_dates
    df = _synth_history(date(2026, 5, 1), 30)
    dates = pick_default_as_of_dates(df, lookback_td=60, forward_td=20)
    # Not enough history → empty list, not crash
    assert dates == []


# =============================================================================
# Aggregate scoring
# =============================================================================

def test_aggregate_results_zero_events():
    from tools.diag.ai_accuracy_backtest import aggregate_results
    out = aggregate_results([])
    assert out["n_events"] == 0


def test_aggregate_results_hit_rate_and_bias():
    from tools.diag.ai_accuracy_backtest import aggregate_results
    rows = [
        # All 4 directionally-bullish predictions; 3 realized bullish, 1 bearish.
        # Hit rate = 3/4 = 0.75. AI mean-overshoots realized.
        {"ai_predicted_drift": 0.30, "realized_drift_ann": 0.20,
         "direction_match": 1, "overshoot_ratio": 1.5,
         "truncated_forward": False},
        {"ai_predicted_drift": 0.40, "realized_drift_ann": 0.10,
         "direction_match": 1, "overshoot_ratio": 4.0,
         "truncated_forward": False},
        {"ai_predicted_drift": 0.20, "realized_drift_ann": 0.05,
         "direction_match": 1, "overshoot_ratio": 4.0,
         "truncated_forward": False},
        {"ai_predicted_drift": 0.30, "realized_drift_ann": -0.10,
         "direction_match": 0, "overshoot_ratio": -3.0,
         "truncated_forward": False},
    ]
    agg = aggregate_results(rows)
    assert agg["n_events"] == 4
    assert agg["n_scored"] == 4
    assert agg["directional_hit_rate"] == pytest.approx(0.75)
    # Mean predicted = 0.30, mean realized = 0.0625 → bias = +0.2375
    assert agg["mean_predicted_drift"] == pytest.approx(0.30, rel=1e-6)
    assert agg["mean_realized_drift"] == pytest.approx(0.0625, rel=1e-6)
    assert agg["magnitude_bias"] == pytest.approx(0.2375, rel=1e-6)
    # Overshoot median across [1.5, 4.0, 4.0, -3.0] is the middle of
    # sorted [-3.0, 1.5, 4.0, 4.0] = (1.5 + 4.0) / 2 = 2.75
    assert agg["overshoot_median_ratio"] == pytest.approx(2.75, rel=1e-6)


def test_aggregate_results_excludes_truncated_forwards():
    """Events whose forward window was truncated below the 20-trading-day
    minimum must not contribute to summary stats — they're unscored."""
    from tools.diag.ai_accuracy_backtest import aggregate_results
    rows = [
        {"ai_predicted_drift": 0.30, "realized_drift_ann": 0.20,
         "direction_match": 1, "overshoot_ratio": 1.5,
         "truncated_forward": True},  # ← excluded
        {"ai_predicted_drift": 0.30, "realized_drift_ann": -0.10,
         "direction_match": 0, "overshoot_ratio": -3.0,
         "truncated_forward": False},  # ← included
    ]
    agg = aggregate_results(rows)
    assert agg["n_scored"] == 1

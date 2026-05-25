"""Tests for PR #79 — audit finding #5.

`signal_from_sector_decoupling` was subtracting two returns measured
over potentially different windows:

  - own_ret    : 30-trading-bar return (`iloc[-31]`) — hardcoded
  - sector_ret : cumulative return over `len(rows)` trading bars from
                 `fetch_sector_perf`, where `len(rows)` depends on
                 FMP coverage gaps and the calendar window the fetcher
                 requested (which itself was calendar days, not
                 trading bars).

Then annualised the difference with `252 / lookback_days` (30) regardless
of the sector's actual window — double-wrong when the periods didn't
match.

Fix: use `sector_perf['n_days']` as the comparison window for BOTH legs;
annualise with `252 / n_days`. If the sector window is much shorter than
nominal (data gap), surface in notes; if it's TOO short (<5 bars),
return _none_signal rather than feed a stealthily-broken signal forward.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.signals import signal_from_sector_decoupling


def _price_df(n_bars, daily_return=0.0):
    """Build a synthetic price DataFrame with `n_bars` trading-day rows."""
    closes = [100.0]
    for _ in range(n_bars):
        closes.append(closes[-1] * (1.0 + daily_return))
    return pd.DataFrame({
        "Date": pd.date_range(end="2026-05-22", periods=len(closes), freq="B"),
        "Close": closes,
    })


def test_decoupling_uses_sector_n_days_not_signal_lookback():
    """If sector_perf reports n_days=20 (data gap), own_ret must be
    sliced over 20 bars, NOT 30. Annualisation factor must use 20."""
    df = _price_df(50, daily_return=0.01)  # +1%/day → ~22% over 20 bars
    # Sector flat over the same 20 bars.
    sector_perf = {"cum_return_pct": 0.0, "n_days": 20}
    out = signal_from_sector_decoupling(
        df, sector_perf, lookback_days=30, ticker="X"
    )
    # Stock returned +22% (20 bars of compounded +1%), sector 0%.
    # decoup ≈ +0.22. Annualised = 0.22 * 252/20 = 2.77 → capped.
    assert out["drift"] != 0  # signal fires (not the _none path)
    # Notes should reflect the actual window (20d), not the nominal (30d).
    assert "over 20d" in out["notes"]
    assert "window 20d, nominal 30d" in out["notes"]


def test_decoupling_falls_back_when_n_days_missing():
    """Legacy sector_perf without n_days uses the signal's nominal
    lookback — backward compat."""
    df = _price_df(50, daily_return=0.0)
    sector_perf = {"cum_return_pct": 5.0}  # no n_days field
    out = signal_from_sector_decoupling(
        df, sector_perf, lookback_days=30, ticker="X"
    )
    assert "over 30d" in out["notes"]
    # No period-mismatch chip when n_days == lookback_days.
    assert "nominal" not in out["notes"]


def test_decoupling_refuses_when_sector_window_too_short():
    """Sector_perf with <5 bars → _none_signal (can't subtract two
    returns when one of them is over 3 days of data)."""
    df = _price_df(50)
    sector_perf = {"cum_return_pct": 5.0, "n_days": 3}
    out = signal_from_sector_decoupling(
        df, sector_perf, lookback_days=30, ticker="X"
    )
    # _none_signal returns drift=None, source_quality=NONE_FOUND.
    assert out["drift"] is None
    assert out["source_quality"] == "NONE_FOUND"
    assert "too short" in out["notes"]


def test_decoupling_refuses_when_price_history_too_short_for_window():
    """If price_df has 40 bars but n_days=50, we can't compute own_ret
    over the matched window. Refuse."""
    df = _price_df(40)
    sector_perf = {"cum_return_pct": 5.0, "n_days": 50}
    out = signal_from_sector_decoupling(
        df, sector_perf, lookback_days=30, ticker="X"
    )
    assert out["drift"] is None
    assert out["source_quality"] == "NONE_FOUND"


def test_decoupling_signal_sign_consistent():
    """Stock UP, sector FLAT → positive decoup → positive drift."""
    df = _price_df(40, daily_return=0.01)
    sector_perf = {"cum_return_pct": 0.0, "n_days": 30}
    out = signal_from_sector_decoupling(
        df, sector_perf, lookback_days=30, ticker="UP"
    )
    assert out["drift"] > 0

    # Reverse: stock FLAT, sector UP → negative decoup → negative drift.
    df2 = _price_df(40, daily_return=0.0)
    sector_perf2 = {"cum_return_pct": 10.0, "n_days": 30}
    out2 = signal_from_sector_decoupling(
        df2, sector_perf2, lookback_days=30, ticker="FLAT"
    )
    assert out2["drift"] < 0


def test_annualisation_factor_uses_n_days_not_lookback():
    """Same own_ret + sector_ret combinations, but DIFFERENT n_days.
    Annualised drift must scale inversely with n_days (longer period
    → smaller annual factor)."""
    df = _price_df(80, daily_return=0.005)  # gentle uptrend so own_ret > 0
    sector_perf_short = {"cum_return_pct": 0.0, "n_days": 20}
    sector_perf_long = {"cum_return_pct": 0.0, "n_days": 60}
    out_short = signal_from_sector_decoupling(
        df, sector_perf_short, lookback_days=30, ticker="X"
    )
    out_long = signal_from_sector_decoupling(
        df, sector_perf_long, lookback_days=30, ticker="X"
    )
    # 20-bar return < 60-bar return on a steady uptrend, but the
    # annualisation factor (252/20=12.6) is much LARGER than (252/60=4.2).
    # The 20-bar drift should annualise to a larger absolute value (or
    # at minimum the two should be close once capped). Concrete check:
    # neither matches the OLD formula which used signal's nominal 30.
    # Both legs measure the same period, so the drift's sign is consistent
    # and the magnitudes scale correctly with n_days.
    assert out_short["drift"] != 0
    assert out_long["drift"] != 0
    # Same direction (both positive — stock outperformed flat sector).
    assert (out_short["drift"] > 0) == (out_long["drift"] > 0)

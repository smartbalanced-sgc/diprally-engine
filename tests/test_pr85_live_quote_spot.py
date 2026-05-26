"""Tests for PR #85 — fetch_live_quote replaces stale daily-bar close as spot.

Pre-PR-#85 the engine set:
    spot = float(history_df["Close"].iloc[-1])

FMP's `/stable/historical-price-eod` (the source of `history_df`) lags
1-2 hours after market close. The 2026-05-26 16:50 ET cycle therefore
used FRIDAY's close as spot for SNDK ($1,478 vs Tue close $1,589 +7.5%)
and MU ($751 vs Tue close $895 +19.3%). Every BUY in that cycle was
computed against stale spots.

PR #85 fix:
  - `fetch_live_quote(ticker, api_key)` uses `/stable/quote` which
    returns current intraday price (verified on Starter plan tier).
  - Engine spot now sources from live_quote, with daily-bar close as
    fallback. CSV captures `spot_source` per row.
  - Dashboard `_spot_source_line` reports the actual breakdown
    (live_quote vs fallback) instead of an unconditional "Live quote".

Regression tests:
  - Confirm fetch_live_quote parses the documented FMP /stable/quote
    schema (the exact response shape from 2026-05-27 diagnostic run).
  - Confirm None on empty list / non-list / missing price / non-numeric.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# fetch_live_quote — parser tests against the real FMP /stable/quote shape
# =============================================================================

# Real response captured from FMP on 2026-05-27 for SNDK. Keeping this
# in tree as ground-truth schema reference.
SNDK_QUOTE = [{
    "symbol": "SNDK",
    "name": "Sandisk Corporation",
    "price": 1589.55,
    "changePercentage": 7.49718,
    "change": 110.86,
    "volume": 12666761,
    "dayLow": 1520,
    "dayHigh": 1641.74,
    "yearHigh": 1641.74,
    "yearLow": 36.207,
    "marketCap": 235396062113,
    "priceAvg50": 1005.0656,
    "priceAvg200": 456.54022,
    "exchange": "NASDAQ",
    "open": 1535.645,
    "previousClose": 1478.69,
    "timestamp": 1779825600,
}]


def test_fetch_live_quote_parses_real_fmp_response():
    """Round-trip the exact shape from FMP /stable/quote."""
    from src import data_fetch
    with patch.object(data_fetch, "_fmp_get", return_value=SNDK_QUOTE):
        out = data_fetch.fetch_live_quote("SNDK", api_key="dummy")
    assert out is not None
    assert out["symbol"] == "SNDK"
    assert out["price"] == pytest.approx(1589.55)
    assert out["previous_close"] == pytest.approx(1478.69)
    assert out["day_high"] == pytest.approx(1641.74)
    assert out["day_low"] == pytest.approx(1520.0)
    assert out["open"] == pytest.approx(1535.645)
    assert out["exchange"] == "NASDAQ"
    assert out["change_pct"] == pytest.approx(7.49718)
    assert out["timestamp"] == 1779825600


def test_fetch_live_quote_returns_none_on_empty_list():
    from src import data_fetch
    with patch.object(data_fetch, "_fmp_get", return_value=[]):
        assert data_fetch.fetch_live_quote("XYZ", "k") is None


def test_fetch_live_quote_returns_none_on_dict_payload():
    """FMP error payloads come back as {'Error Message': '...'} — a
    dict, not a list. Parser must handle gracefully."""
    from src import data_fetch
    with patch.object(data_fetch, "_fmp_get",
                       return_value={"Error Message": "rate limit"}):
        assert data_fetch.fetch_live_quote("XYZ", "k") is None


def test_fetch_live_quote_returns_none_on_missing_price():
    from src import data_fetch
    with patch.object(data_fetch, "_fmp_get",
                       return_value=[{"symbol": "XYZ"}]):
        assert data_fetch.fetch_live_quote("XYZ", "k") is None


def test_fetch_live_quote_returns_none_on_zero_price():
    """Defensive: a zero/negative price is nonsense — don't propagate it
    into the engine where it'd corrupt the entire MC."""
    from src import data_fetch
    with patch.object(data_fetch, "_fmp_get",
                       return_value=[{"symbol": "XYZ", "price": 0}]):
        assert data_fetch.fetch_live_quote("XYZ", "k") is None
    with patch.object(data_fetch, "_fmp_get",
                       return_value=[{"symbol": "XYZ", "price": -5.0}]):
        assert data_fetch.fetch_live_quote("XYZ", "k") is None


def test_fetch_live_quote_returns_none_on_non_numeric_price():
    from src import data_fetch
    with patch.object(data_fetch, "_fmp_get",
                       return_value=[{"symbol": "XYZ", "price": "n/a"}]):
        assert data_fetch.fetch_live_quote("XYZ", "k") is None


# =============================================================================
# Spot-source CSV column and dashboard reporting
# =============================================================================

def test_csv_columns_include_spot_source():
    """The new column must be in CSV_COLUMNS so DictWriter doesn't drop it."""
    from src.engine import CSV_COLUMNS
    assert "spot_source" in CSV_COLUMNS


def test_orchestrator_spot_source_counts_live_and_fallback(tmp_path, monkeypatch):
    """_spot_source_counts reads today's CSV rows across all tickers and
    tallies the spot_source field."""
    from src import orchestrator
    from src.engine import CSV_COLUMNS
    import csv as _csv
    from datetime import datetime

    monkeypatch.setattr(orchestrator, "_OUTPUT_ROOT", tmp_path)
    today = datetime.now().strftime("%Y-%m-%d")

    for ticker, source in [("AAA", "live_quote"),
                            ("BBB", "live_quote"),
                            ("CCC", "daily_bar_fallback")]:
        path = tmp_path / f"round_trip_history_{ticker}.csv"
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            row = {c: "" for c in CSV_COLUMNS}
            row["date"] = today
            row["spot"] = "100.00"
            row["spot_source"] = source
            w.writerow(row)

    counts = orchestrator._spot_source_counts()
    assert counts["live_quote"] == 2
    assert counts["daily_bar_fallback"] == 1


def test_spot_source_line_reports_actual_breakdown_on_trading_day(monkeypatch, tmp_path):
    """When tickers used live_quote, the dashboard line says so. When
    SOME fell back to daily-bar, the line surfaces a warning with the
    count — institutional honesty about data freshness."""
    from src import orchestrator
    from src.engine import CSV_COLUMNS
    import csv as _csv
    from datetime import datetime, date

    monkeypatch.setattr(orchestrator, "_OUTPUT_ROOT", tmp_path)
    # Force "today" to a known trading day (Tue 2026-05-26).
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None): return datetime(2026, 5, 26, 14, 0)
    monkeypatch.setattr(orchestrator, "datetime", _DT)

    for ticker, source in [("AAA", "live_quote"),
                            ("BBB", "live_quote"),
                            ("CCC", "daily_bar_fallback")]:
        path = tmp_path / f"round_trip_history_{ticker}.csv"
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            row = {c: "" for c in CSV_COLUMNS}
            row["date"] = "2026-05-26"
            row["spot"] = "100.00"
            row["spot_source"] = source
            w.writerow(row)

    line = orchestrator._spot_source_line()
    # Surfaces both counts.
    assert "2/3" in line
    assert "fell back" in line  # one ticker used daily-bar fallback
    # Honest data-freshness caveat about the fallback path.
    assert "may be stale" in line


def test_spot_source_line_clean_when_all_live(monkeypatch, tmp_path):
    """All tickers used live quote → no warning chip, no '⚠'."""
    from src import orchestrator
    from src.engine import CSV_COLUMNS
    import csv as _csv
    from datetime import datetime

    monkeypatch.setattr(orchestrator, "_OUTPUT_ROOT", tmp_path)
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None): return datetime(2026, 5, 26, 14, 0)
    monkeypatch.setattr(orchestrator, "datetime", _DT)

    for ticker in ["AAA", "BBB"]:
        path = tmp_path / f"round_trip_history_{ticker}.csv"
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            row = {c: "" for c in CSV_COLUMNS}
            row["date"] = "2026-05-26"
            row["spot"] = "100.00"
            row["spot_source"] = "live_quote"
            w.writerow(row)

    line = orchestrator._spot_source_line()
    assert "2/2" in line
    assert "fell back" not in line
    # Holiday/weekend warning marker should NOT appear here (it's a
    # trading day with all-live spot).
    assert "⚠" not in line

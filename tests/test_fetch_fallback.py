"""Tests for D-W2-14 — yfinance fallback when FMP fails.

Verifies the resilience capstone for W2: one bad FMP response (402, 404,
429, 5xx, timeout) no longer kills the pipeline. Engine retries via
yfinance with proper per-provider symbol translation.

These tests use mocks (no live FMP / yfinance calls). Pure Python.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import requests

from src.data_fetch import FetchError, fetch_history
from src.registry import provider_symbol


# ---------- Per-provider symbol translation ----------

def test_provider_symbol_defaults_to_canonical_when_no_override():
    """Today's universe uses dash-form on both providers. provider_symbol
    returns the canonical symbol as-is when no override is configured."""
    assert provider_symbol("INTC", "fmp") == "INTC"
    assert provider_symbol("INTC", "yfinance") == "INTC"
    assert provider_symbol("MOG-A", "fmp") == "MOG-A"
    assert provider_symbol("MOG-A", "yfinance") == "MOG-A"


def test_provider_symbol_case_insensitive():
    """Operator-friendly: lowercase input still works."""
    assert provider_symbol("intc", "fmp") == "INTC"
    assert provider_symbol("intc", "YFINANCE") == "INTC"


def test_provider_symbol_unknown_ticker_returns_canonical():
    """Tickers not in the universe still get a callable result (uppercased
    canonical). Caller is responsible for either adding to registry or
    handling FetchError downstream."""
    assert provider_symbol("UNKNOWN", "fmp") == "UNKNOWN"
    assert provider_symbol("UNKNOWN", "yfinance") == "UNKNOWN"


def test_provider_symbol_rejects_unknown_provider():
    """Defensive — caller passing wrong provider name fails fast."""
    try:
        provider_symbol("INTC", "bloomberg")
        assert False, "should have raised"
    except ValueError as e:
        assert "bloomberg" in str(e)


# ---------- Yfinance fallback behavior ----------

def _fake_fmp_response(status_code=200, json_data=None):
    """Mock requests.Response for FMP calls."""
    resp = MagicMock()
    resp.status_code = status_code
    if status_code >= 400:
        http_err = requests.exceptions.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_err
    else:
        resp.raise_for_status.return_value = None
    resp.json.return_value = json_data if json_data is not None else []
    return resp


def _fake_yf_dataframe():
    """Mock yfinance .history() return value — pandas DataFrame with
    Close column and a DatetimeIndex named Date."""
    dates = pd.date_range("2024-01-01", periods=500, freq="B")
    df = pd.DataFrame({
        "Open": [100.0] * 500,
        "Close": [100.0 + i * 0.1 for i in range(500)],
        "High": [101.0] * 500,
        "Low": [99.0] * 500,
        "Volume": [1000000] * 500,
    }, index=pd.DatetimeIndex(dates, name="Date"))
    return df


def test_fmp_success_uses_fmp_no_fallback():
    """When FMP returns clean data, yfinance is NEVER called.
    df.attrs['data_source'] == 'fmp'."""
    fake_data = [
        {"date": "2024-01-01", "close": 100.0},
        {"date": "2024-01-02", "close": 101.0},
        {"date": "2024-01-03", "close": 102.0},
    ]
    with patch("src.data_fetch.requests.get") as mock_get:
        mock_get.return_value = _fake_fmp_response(200, fake_data)
        df = fetch_history("TEST", "fakekey", 730)
    assert df is not None
    assert df.attrs.get("data_source") == "fmp"
    assert len(df) == 3


def test_fmp_402_triggers_yfinance_fallback():
    """The exact scenario that originally killed the MOG.A smoke. With
    fallback, FMP 402 is logged + yfinance is retried + pipeline continues."""
    with patch("src.data_fetch.requests.get") as mock_get, \
         patch("yfinance.Ticker") as mock_yf:
        mock_get.return_value = _fake_fmp_response(402)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = _fake_yf_dataframe()
        mock_yf.return_value = mock_ticker_instance
        df = fetch_history("TEST", "fakekey", 730)
    assert df is not None
    assert df.attrs.get("data_source") == "yfinance"
    assert "Close" in df.columns


def test_fmp_404_triggers_yfinance_fallback():
    """Same fallback path for ticker-not-found errors."""
    with patch("src.data_fetch.requests.get") as mock_get, \
         patch("yfinance.Ticker") as mock_yf:
        mock_get.return_value = _fake_fmp_response(404)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = _fake_yf_dataframe()
        mock_yf.return_value = mock_ticker_instance
        df = fetch_history("TEST", "fakekey", 730)
    assert df.attrs.get("data_source") == "yfinance"


def test_fmp_429_rate_limit_triggers_yfinance_fallback():
    """Rate limit — fall back to yfinance which has different rate limits."""
    with patch("src.data_fetch.requests.get") as mock_get, \
         patch("yfinance.Ticker") as mock_yf:
        mock_get.return_value = _fake_fmp_response(429)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = _fake_yf_dataframe()
        mock_yf.return_value = mock_ticker_instance
        df = fetch_history("TEST", "fakekey", 730)
    assert df.attrs.get("data_source") == "yfinance"


def test_both_providers_fail_raises_combined_fetcherror():
    """When BOTH FMP and yfinance fail, raise a single FetchError with
    both reasons in the message — debugging needs to know which failed
    and why."""
    with patch("src.data_fetch.requests.get") as mock_get, \
         patch("yfinance.Ticker") as mock_yf:
        mock_get.return_value = _fake_fmp_response(402)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("yfinance also failed")
        mock_yf.return_value = mock_ticker_instance
        try:
            fetch_history("TEST", "fakekey", 730)
            assert False, "should have raised"
        except FetchError as e:
            # Combined error must mention both failures
            assert e.source == "all-providers"
            reason = str(e)
            assert "FMP failed" in reason
            assert "yfinance" in reason


def test_yfinance_empty_response_raises_in_fallback():
    """Even after fallback, an empty yfinance response → combined error."""
    with patch("src.data_fetch.requests.get") as mock_get, \
         patch("yfinance.Ticker") as mock_yf:
        mock_get.return_value = _fake_fmp_response(402)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()  # empty
        mock_yf.return_value = mock_ticker_instance
        try:
            fetch_history("TEST", "fakekey", 730)
            assert False, "should have raised"
        except FetchError as e:
            assert e.source == "all-providers"


def test_yfinance_normalizes_to_fmp_shape():
    """yfinance returns a DatetimeIndex-keyed df with Close column. fetch_history
    must normalize to FMP-compatible shape: Date column (not index) + Close.
    Otherwise downstream signal_from_peer_rs etc. break on missing columns."""
    with patch("src.data_fetch.requests.get") as mock_get, \
         patch("yfinance.Ticker") as mock_yf:
        mock_get.return_value = _fake_fmp_response(402)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = _fake_yf_dataframe()
        mock_yf.return_value = mock_ticker_instance
        df = fetch_history("TEST", "fakekey", 730)
    # Must have Date column (not index)
    assert "Date" in df.columns
    # Date column must be datetime-typed
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    # Must have Close column
    assert "Close" in df.columns
    # Must be sorted ascending by date
    assert df["Date"].is_monotonic_increasing


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in sorted(inspect.getmembers(sys.modules[__name__], inspect.isfunction)):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except (AssertionError, Exception) as e:
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            fails += 1
    if fails:
        print(f"\n{fails} test(s) failed")
        sys.exit(1)
    print("\nALL TESTS PASSED")

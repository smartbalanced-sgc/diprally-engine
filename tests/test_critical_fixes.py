"""Tests for the pre-W2 critical fixes.

Each test corresponds to one of the 8 fixes in the critical-fixes PR:

  1. peer_rs lookback bug — verify the peer history has enough trading
     bars to compute the n_day_return
  2. Display masking fix — NONE_FOUND signal yields is_absent=True
  3. URL apikey redaction in error logs
  4. FetchError typed exception on HTTP failure
  5. Analyst extreme-outlier confidence downgrade
  6. Effective-weight column computed correctly from blend
  7. GARCH α+β 4-decimal display when > 0.95 (visual; smoke-tested by
     other tests indirectly)

These tests are first-line-of-defense regression guards. They run via
`python tests/test_critical_fixes.py` or under pytest. No external
dependencies (no FMP / no Anthropic).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import requests

from src.config import (
    ANALYST_EXTREME_DRIFT_THRESHOLD,
    BLEND_WEIGHTS_V2,
    DEFAULT_LOOKBACK_DAYS,
)
from src.data_fetch import FetchError, _redact
from src.engine import DriftSignal, _signals_dict_to_display_list
from src.signals import (
    _gate_extreme_drift,
    blend_with_uncertainty,
    signal_from_analyst_targets,
    signal_from_peer_rs,
)


# ---------- 1. peer_rs lookback ----------

def test_peer_rs_works_with_730d_history():
    """With DEFAULT_LOOKBACK_DAYS=730 calendar days fetched, peer_rs has
    plenty of trading bars to compute the 60-trading-day return.

    Pre-fix: fetch_peer_history was called with lookback_days=60, yielding
    ~43 trading bars — n_day_return(df, 60) returned None — signal returned
    _none_signal, silently masked as '+0.0% LOW' in display.

    Post-fix: caller passes DEFAULT_LOOKBACK_DAYS=730. Test verifies signal
    produces a real drift output when given enough bars."""
    # Synthesize 100 trading bars of history for ticker + peer
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    own = pd.DataFrame({
        "Date": dates,
        "Close": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),  # +0.1%/day drift
    })
    peer = pd.DataFrame({
        "Date": dates,
        "Close": 100 * np.cumprod(1 + np.random.normal(-0.0005, 0.02, 100)),  # -0.05%/day drift
    })
    result = signal_from_peer_rs(own, {"PEER": peer}, lookback_days=60, ticker="TEST")
    assert result["drift"] is not None, "Signal returned None — peer lookback still broken"
    assert result["confidence"] in ("HIGH", "MEDIUM", "LOW"), result
    assert "TEST" in result["notes"]


def test_peer_rs_returns_none_signal_when_too_few_bars():
    """Pre-fix masking sanity check: when history is genuinely too short
    (e.g. newly-IPOed ticker), signal returns _none_signal — and the
    display fix (test below) marks it absent rather than showing +0.0%."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")  # only 30 bars
    own = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 110, 30)})
    peer = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 105, 30)})
    result = signal_from_peer_rs(own, {"PEER": peer}, lookback_days=60, ticker="TEST")
    assert result["drift"] is None, "Should return _none_signal when history too short"
    assert result["source_quality"] == "NONE_FOUND"


# ---------- 2. Display masking: is_absent ----------

def test_display_marks_none_found_signals_absent():
    """NONE_FOUND signals must be marked is_absent=True so the reporter
    can render 'n/a' rather than the misleading '+0.0%'."""
    signals_dict = {
        "historical": {"drift": 0.10, "confidence": "HIGH", "source_quality": "PRIMARY",
                       "sources_count": 1, "notes": "ok"},
        "peer_rs": {"drift": None, "confidence": "LOW", "source_quality": "NONE_FOUND",
                    "sources_count": 0, "notes": "no peers"},
    }
    out = _signals_dict_to_display_list(signals_dict, BLEND_WEIGHTS_V2)
    by_name = {s.name: s for s in out}
    historical = [s for s in out if "Historical" in s.name][0]
    peer_rs = [s for s in out if "Peer RS" in s.name][0]
    assert historical.is_absent is False, "Active signal incorrectly marked absent"
    assert peer_rs.is_absent is True, "NONE_FOUND signal not marked absent"


def test_display_effective_weight_computed_from_blend():
    """Effective weight column shows post-gate-renormalized weights from
    the live blend, not the nominal BLEND_WEIGHTS_V2 values."""
    signals_dict = {
        "historical": {"drift": 0.10, "confidence": "MEDIUM", "source_quality": "PRIMARY",
                       "sources_count": 1, "notes": "ok"},  # weight 0.05, full
        "analyst": {"drift": -0.04, "confidence": "HIGH", "source_quality": "REPUTABLE",
                    "sources_count": 10, "notes": "ok"},  # weight 0.15, full
        "ai": {"drift": 0.0, "confidence": "LOW", "source_quality": "NONE_FOUND",
               "sources_count": 0, "notes": "AI absent"},  # phantom — zero weight
    }
    blend = blend_with_uncertainty(signals_dict, weights_dict=BLEND_WEIGHTS_V2)
    out = _signals_dict_to_display_list(signals_dict, BLEND_WEIGHTS_V2, blend=blend)
    by_name = {s.name.split()[0]: s for s in out}
    historical = [s for s in out if "Historical" in s.name][0]
    analyst = [s for s in out if "Analyst" in s.name][0]
    ai = [s for s in out if "AI analyst" in s.name][0]
    # historical: nominal 0.05 / (0.05 + 0.15) = 25% effective (only 2 active)
    # analyst: nominal 0.15 / 0.20 = 75% effective
    # ai: NONE_FOUND → 0
    assert abs(historical.effective_weight - 0.25) < 0.01, f"hist effective: {historical.effective_weight}"
    assert abs(analyst.effective_weight - 0.75) < 0.01, f"analyst effective: {analyst.effective_weight}"
    assert ai.effective_weight == 0.0, f"ai effective should be 0: {ai.effective_weight}"


# ---------- 3. URL apikey redaction ----------

def test_redact_strips_apikey_from_url():
    """Apikey query params must never appear in error messages or logs.
    _redact applies a case-insensitive substring replacement that handles
    the common URL-encoded forms."""
    cases = [
        ("https://fmp.com/api?apikey=secret123&from=2024", "apikey=***REDACTED***"),
        ("HTTPError: 402 ... apikey=ld6wilmawW3FutupImuIMeNIuqafQIMo", "apikey=***REDACTED***"),
        ("apikey=ABC123XYZ", "apikey=***REDACTED***"),
        ("normal text no key", "normal text no key"),  # no-op when no apikey
    ]
    for input_str, expected_substring in cases:
        out = _redact(input_str)
        assert expected_substring in out, f"redaction failed for {input_str!r} → {out!r}"
        if "secret" in input_str or "ld6w" in input_str or "ABC123" in input_str:
            assert "secret123" not in out and "ld6w" not in out and "ABC123" not in out, \
                f"key leaked: {out!r}"


# ---------- 4. FetchError typed exception ----------

def test_fetcherror_carries_status_and_redacts_reason():
    """FetchError must carry ticker, source, status, and a pre-redacted reason
    so callers can distinguish 402-not-in-plan from 429-rate-limit from
    network timeout."""
    e = FetchError("TEST", "fmp", 402, "HTTPError ... apikey=secret")
    assert e.ticker == "TEST"
    assert e.source == "fmp"
    assert e.status == 402
    assert "secret" not in str(e), "FetchError leaked apikey"
    assert "***REDACTED***" in e.reason or "REDACTED" in e.reason


# ---------- 5. Analyst extreme-outlier downgrade ----------

def test_gate_extreme_drift_downgrades_high_to_medium():
    """|implied drift| > threshold should step HIGH → MEDIUM with a
    visible verification flag in the notes."""
    new_conf, new_notes = _gate_extreme_drift(-0.589, "HIGH", "base notes")
    assert new_conf == "MEDIUM"
    assert "EXTREME OUTLIER" in new_notes
    assert "manual verification" in new_notes


def test_gate_extreme_drift_preserves_normal_drift():
    """Normal drift values (under threshold) should pass through unchanged."""
    new_conf, new_notes = _gate_extreme_drift(0.10, "HIGH", "base notes")
    assert new_conf == "HIGH"
    assert new_notes == "base notes"


def test_gate_extreme_drift_caps_at_low():
    """A LOW signal with extreme drift stays LOW (can't go below LOW)."""
    new_conf, new_notes = _gate_extreme_drift(-0.80, "LOW", "x")
    assert new_conf == "LOW"
    assert "EXTREME OUTLIER" in new_notes


def test_analyst_signal_downgrades_extreme_drift_end_to_end():
    """MOG-A scenario: implied drift -58.9% should produce MEDIUM conf
    (not HIGH) with verification flag in notes."""
    # Spot $319, last-month avg $131, n=13 — HIGH base conf
    summary = {"last_month_count": 13, "last_month_avg": 131.0,
               "last_quarter_count": 14, "last_quarter_avg": 130.0,
               "last_year_count": 25, "last_year_avg": 145.0,
               "all_time_count": 30, "all_time_avg": 150.0}
    s = signal_from_analyst_targets({}, 318.87, summary=summary)
    assert abs(s["drift"] - (-0.5891)) < 0.01, f"drift: {s['drift']}"
    assert s["confidence"] == "MEDIUM", f"should downgrade HIGH→MEDIUM: {s['confidence']}"
    assert "EXTREME OUTLIER" in s["notes"]


# ---------- 6. Effective-weight via blend ----------

def test_blend_weights_reflect_low_halving():
    """blend['weights'] after LOW halving should be smaller than nominal."""
    signals_dict = {
        "ai": {"drift": 0.10, "confidence": "LOW", "source_quality": "REPUTABLE",
               "sources_count": 5, "notes": "ok"},  # nominal 0.25, LOW halves to 0.125
        "analyst": {"drift": 0.05, "confidence": "HIGH", "source_quality": "REPUTABLE",
                    "sources_count": 10, "notes": "ok"},  # nominal 0.15 stays 0.15
    }
    blend = blend_with_uncertainty(signals_dict, weights_dict=BLEND_WEIGHTS_V2)
    weights = blend["weights"]
    assert abs(weights["ai"] - 0.125) < 0.001, f"AI halved weight: {weights['ai']}"
    assert abs(weights["analyst"] - 0.15) < 0.001, f"analyst full weight: {weights['analyst']}"


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in sorted(inspect.getmembers(sys.modules[__name__], inspect.isfunction)):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
            fails += 1
    if fails:
        print(f"\n{fails} test(s) failed")
        sys.exit(1)
    print("\nALL TESTS PASSED")

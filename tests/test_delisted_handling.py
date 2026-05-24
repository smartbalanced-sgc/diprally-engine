"""Tests for PR #57 — pre-flight ticker liveness check.

VELO3D-class failures (delisted / removed from data provider) should
surface in the dashboard as DELISTED — operator-actionable feedback
("remove from universe") — instead of as a generic "P1 FAIL" alongside
real engine failures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import orchestrator as orch


# =============================================================================
# _looks_delisted pattern matcher
# =============================================================================

def test_yfinance_delisted_marker_matches():
    """yfinance fallback emits 'possibly delisted; no timezone found'
    on bankrupt/delisted names. This is the canonical signal."""
    text = "ERROR: $VELO3D: possibly delisted; no timezone found"
    assert orch._looks_delisted(text, "VELO3D") is True


def test_delisted_substring_alone_matches():
    """Any 'delisted' substring is strong signal — conservative."""
    text = "ERROR: Symbol may be delisted from NASDAQ"
    assert orch._looks_delisted(text, "XXX") is True


def test_fmp_404_on_profile_endpoint_matches():
    """FMP returns 404 on profile when ticker isn't in provider's
    universe — usually means delisted (or never listed)."""
    text = (
        "WARNING: FMP profile?symbol=velo3d failed: "
        "404 Client Error: Not Found"
    )
    assert orch._looks_delisted(text, "VELO3D") is True


def test_fmp_404_on_other_endpoint_does_not_match():
    """A 404 on a non-profile endpoint (e.g. missing earnings calendar
    for a thinly-covered name) is NOT delisted — it's a data gap.
    Conservative matcher must not flip."""
    text = (
        "WARNING: FMP earnings-calendar?symbol=GHM failed: "
        "404 Client Error: Not Found"
    )
    assert orch._looks_delisted(text, "GHM") is False


def test_transient_network_error_does_not_match():
    """Connection timeouts / DNS failures / Anthropic API hiccups
    must NOT flag as DELISTED — they're transient infrastructure
    issues, not ticker-lifecycle events."""
    for transient in (
        "ERROR: connection timeout to financialmodelingprep.com",
        "ERROR: DNS resolution failed",
        "Anthropic API error: rate limited",
        "FetchError: network unreachable",
    ):
        assert orch._looks_delisted(transient, "INTC") is False, \
            f"transient {transient!r} should not flag as delisted"


def test_empty_input_safe():
    assert orch._looks_delisted("", "INTC") is False
    assert orch._looks_delisted(None, "INTC") is False


# =============================================================================
# Verdict + status routing
# =============================================================================

def _delisted_run(ticker):
    """Synthesize a TickerRun shaped like _phase1_single would
    produce on a delisted ticker."""
    return orch.TickerRun(
        ticker=ticker, phase1_returncode=1, snapshot=None,
        phase1_error="ERROR: $VELO3D: possibly delisted; no timezone found",
        delisted=True,
    )


def _generic_fail_run(ticker):
    return orch.TickerRun(
        ticker=ticker, phase1_returncode=1, snapshot=None,
        phase1_error="ERROR: FetchError: connection refused",
        delisted=False,
    )


def test_delisted_decision_verdict_is_DELISTED():
    decision = orch._decision_from_run(_delisted_run("VELO3D"))
    assert decision.verdict == "DELISTED"
    assert "remove from" in decision.status_note.lower() or \
           "delisted" in decision.status_note.lower()


def test_generic_fail_decision_stays_FAIL():
    decision = orch._decision_from_run(_generic_fail_run("INTC"))
    assert decision.verdict == "FAIL"


def test_delisted_color_distinct_from_fail():
    """DELISTED is operator-actionable (not trader-actionable), so
    it should look different from FAIL in the dashboard. FAIL is
    red (urgent investigation); DELISTED is darker gray (queue for
    universe cleanup)."""
    assert orch._VERDICT_COLORS["DELISTED"] != orch._VERDICT_COLORS["FAIL"]


def test_summary_includes_delisted_status(tmp_path, monkeypatch):
    """format_summary surfaces DELISTED as a distinct status line
    instead of muddling it into 'P1 FAIL' counts."""
    runs = [
        _generic_fail_run("BUSTED"),
        _delisted_run("VELO3D"),
    ]
    summary = orch.format_summary(runs, None)
    assert "DELISTED" in summary
    assert "remove from universe" in summary


def test_delisted_does_not_count_as_p1_fail_in_dashboard(tmp_path, monkeypatch):
    """The dashboard tile row should split DELISTED from FAIL so
    a universe with 1 delisted ticker doesn't show 1 FAIL."""
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    runs = [_delisted_run("VELO3D")]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = orch.generate_aggregate_dashboard(runs, None, run_dir)
    html = path.read_text()
    # DELISTED tile is present, distinct count from FAIL.
    assert ">DELISTED" in html
    assert "<strong>1</strong>DELISTED" in html
    assert "<strong>0</strong>FAIL" in html

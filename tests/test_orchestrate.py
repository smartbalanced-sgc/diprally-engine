"""Tests for the W5 orchestrator helpers — W5 PR #31.

The full subprocess pipeline needs FMP credentials + network and is
covered by manual smoke runs (see docs/handover/_DEFERRED.md). These
tests cover the pure-Python pieces: snapshot parsing, summary
formatting, run-dir scaffolding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import orchestrator as orch
from src.broker import BrokerSnapshot, BrokerAllocation, allocate


def test_parse_snapshot_extracts_json_line():
    stdout = (
        "lots of report text...\n"
        "more report text\n"
        'BROKER_SNAPSHOT_JSON={"ticker": "INTC", "ambiguity": 0.42, '
        '"qualifies_for_t2_plus": true, "sigma_class": "MID"}\n'
    )
    snap = orch.parse_snapshot(stdout)
    assert snap is not None
    assert snap.ticker == "INTC"
    assert snap.ambiguity == 0.42
    assert snap.qualifies_for_t2_plus is True
    assert snap.sigma_class == "MID"


def test_parse_snapshot_returns_none_when_absent():
    stdout = "no snapshot in this output\nsome other text\n"
    assert orch.parse_snapshot(stdout) is None


def test_parse_snapshot_uppercases_ticker():
    """Engine emits ticker uppercase, but be defensive — broker should
    always see normalized symbols."""
    stdout = ('BROKER_SNAPSHOT_JSON={"ticker": "lwlg", "ambiguity": 0.5, '
              '"qualifies_for_t2_plus": false, "sigma_class": "EXTREME"}\n')
    snap = orch.parse_snapshot(stdout)
    assert snap is not None
    assert snap.ticker == "LWLG"


def test_parse_snapshot_handles_malformed_json():
    stdout = "BROKER_SNAPSHOT_JSON={broken json here\n"
    # Regex matches the line, but json.loads fails — should return None,
    # not raise.
    assert orch.parse_snapshot(stdout) is None


def test_parse_snapshot_handles_missing_fields():
    stdout = 'BROKER_SNAPSHOT_JSON={"ticker": "X"}\n'
    assert orch.parse_snapshot(stdout) is None


def _make_run(ticker, ambiguity=None, qualifies=True, sigma_class="MID",
               phase1_error=None, phase2_rc=None, assigned_tier="T0"):
    snap = None
    if ambiguity is not None:
        snap = BrokerSnapshot(
            ticker=ticker, ambiguity=ambiguity,
            qualifies_for_t2_plus=qualifies, sigma_class=sigma_class,
        )
    return orch.TickerRun(
        ticker=ticker, phase1_returncode=0 if snap else 1,
        snapshot=snap, phase1_error=phase1_error,
        assigned_tier=assigned_tier, phase2_returncode=phase2_rc,
    )


def test_summary_includes_all_tickers():
    runs = [
        _make_run("LWLG", 0.78, True, "EXTREME"),
        _make_run("INTC", 0.30, True, "MID"),
        _make_run("BROKEN", phase1_error="FetchError: 404"),
    ]
    snapshots = [r.snapshot for r in runs if r.snapshot]
    alloc = allocate(snapshots)
    summary = orch.format_summary(runs, alloc)
    for t in ("LWLG", "INTC", "BROKEN"):
        assert t in summary
    assert "P1 FAIL" in summary  # BROKEN's failure surfaces


def test_summary_sorts_by_ambiguity_desc():
    runs = [
        _make_run("TLOW",  0.20, True, "MID"),
        _make_run("THIGH", 0.80, True, "EXTREME"),
        _make_run("TMID",  0.50, True, "HIGH"),
    ]
    snapshots = [r.snapshot for r in runs]
    alloc = allocate(snapshots)
    summary = orch.format_summary(runs, alloc)
    # THIGH (0.80) should appear before TMID (0.50), which appears before TLOW (0.20).
    pos_high = summary.find("THIGH")
    pos_mid = summary.find("TMID")
    pos_low = summary.find("TLOW")
    assert pos_high > 0 and pos_mid > 0 and pos_low > 0
    assert pos_high < pos_mid < pos_low


def test_summary_handles_no_allocation():
    """When every ticker fails Phase 1, allocation is None — summary
    should still render without crashing."""
    runs = [_make_run("FAIL1", phase1_error="boom"),
            _make_run("FAIL2", phase1_error="kaboom")]
    summary = orch.format_summary(runs, None)
    assert "FAIL1" in summary and "FAIL2" in summary
    assert "Broker" not in summary  # no broker line when allocation is None

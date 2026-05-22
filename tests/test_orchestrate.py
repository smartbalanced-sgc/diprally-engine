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


# =============================================================================
# W5 PR #32 — aggregate dashboard
# =============================================================================

def test_decision_from_failed_run():
    """A Phase-1 failed ticker → verdict FAIL, no targets, dashboard
    link still pointed at the would-be per-ticker HTML."""
    run = _make_run("BUSTED", phase1_error="HTTP 502")
    d = orch._decision_from_run(run)
    assert d.ticker == "BUSTED"
    assert d.verdict == "FAIL"
    assert d.dip_target is None
    assert "502" in d.status_note


def test_render_dashboard_html_contains_all_tickers(tmp_path):
    """Sanity: each ticker shows up by name in the rendered HTML."""
    runs = [_make_run(t, 0.5, True, "MID") for t in ("AAA", "BBB", "CCC")]
    decisions = [orch._decision_from_run(r) for r in runs]
    html_str = orch._render_dashboard_html(decisions, None, href_prefix="")
    for t in ("AAA", "BBB", "CCC"):
        assert t in html_str
    # Verdict CSS classes present.
    assert 'class="verdict"' in html_str


def test_render_dashboard_links_use_href_prefix(tmp_path):
    """run_dir copy uses ../; stable copy uses bare name. Verify both."""
    runs = [_make_run("INTC", 0.3, True, "MID")]
    decisions = [orch._decision_from_run(r) for r in runs]
    inside = orch._render_dashboard_html(decisions, None, href_prefix="../")
    stable = orch._render_dashboard_html(decisions, None, href_prefix="")
    assert '"../intc_dipnrally_dashboard.html"' in inside
    assert '"intc_dipnrally_dashboard.html"' in stable


def test_generate_dashboard_writes_both_copies(tmp_path, monkeypatch):
    """generate_aggregate_dashboard writes index.html in BOTH run_dir
    and output/. Verify both exist and contain the ticker."""
    # Redirect _OUTPUT_ROOT to a temp dir so the test doesn't touch the
    # repo's real output/ directory.
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    run_dir = tmp_path / "run_xyz"
    run_dir.mkdir()
    runs = [_make_run("INTC", 0.3, True, "MID")]
    snapshots = [r.snapshot for r in runs]
    alloc = allocate(snapshots)
    path = orch.generate_aggregate_dashboard(runs, alloc, run_dir)
    assert path == run_dir / "index.html"
    assert path.exists()
    assert (tmp_path / "index.html").exists()
    assert "INTC" in path.read_text()
    assert "INTC" in (tmp_path / "index.html").read_text()


def test_dashboard_sort_fails_at_bottom(tmp_path, monkeypatch):
    """FAIL tickers should appear AFTER successful tickers in the
    rendered HTML regardless of ambiguity."""
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    run_dir = tmp_path / "run_xyz"
    run_dir.mkdir()
    runs = [
        _make_run("FAIL_HIGH", phase1_error="boom"),  # would-be top by ambig if it had one
        _make_run("OK_LOW", 0.20, True, "MID"),
    ]
    snapshots = [r.snapshot for r in runs if r.snapshot]
    alloc = allocate(snapshots)
    path = orch.generate_aggregate_dashboard(runs, alloc, run_dir)
    html_str = path.read_text()
    pos_ok = html_str.find("OK_LOW")
    pos_fail = html_str.find("FAIL_HIGH")
    assert pos_ok > 0 and pos_fail > 0
    assert pos_ok < pos_fail  # OK before FAIL


def test_dashboard_summary_counts(tmp_path, monkeypatch):
    """The four big-number tiles (BUY/WAIT/REFUSED/FAIL) reflect the
    actual verdict mix."""
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    run_dir = tmp_path / "run_xyz"
    run_dir.mkdir()
    # Two FAIL, one WAIT (no history CSV → verdict WAIT for snapshotted-but-
    # no-pair tickers; but here we only have snapshotted ones, and no CSV
    # exists in tmp_path → row=None → dip_target=None → verdict=WAIT).
    runs = [
        _make_run("WAIT1", 0.30, True, "MID"),
        _make_run("FAIL1", phase1_error="boom"),
        _make_run("FAIL2", phase1_error="kaboom"),
    ]
    snapshots = [r.snapshot for r in runs if r.snapshot]
    alloc = allocate(snapshots)
    path = orch.generate_aggregate_dashboard(runs, alloc, run_dir)
    html_str = path.read_text()
    # Tile counts encoded as `<strong>N</strong>VERDICT_NAME` in the HTML.
    assert "<strong>1</strong>WAIT" in html_str
    assert "<strong>2</strong>FAIL" in html_str
    assert "<strong>0</strong>BUY" in html_str

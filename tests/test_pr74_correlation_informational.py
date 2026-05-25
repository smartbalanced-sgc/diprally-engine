"""Tests for PR #74 — correlation gate flipped from EXCLUSIONARY to
INFORMATIONAL.

Background: PR #49 introduced the portfolio correlation gate to drop
correlated BUYs as "substitute ideas." That framing made sense for
buy-and-hold portfolio construction. For SWING TRADING (this engine's
actual purpose), correlated dip-and-rally setups are independent
EVENTS, not "one bet doubled." Sacred #6 says operator sizes externally
— silencing signals via the gate makes a sizing decision FOR the
operator.

PR #74 keeps all BUYs visible. The correlation analysis still runs;
results are annotated as CORRELATED notes in the status_note field;
verdict stays BUY (not REFUSED-CORRELATED). Operator decides whether
to take both, scale down, or pick one per cluster.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import orchestrator as orch


# Reuse the helpers from test_portfolio_gate_wiring.py via direct
# import to avoid fixture duplication.
sys.path.insert(0, str(_REPO_ROOT / "tests"))
from test_portfolio_gate_wiring import _make_run, _build_correlated_history


# =============================================================================
# 1. Correlated BUYs both retain verdict=BUY
# =============================================================================

def test_correlated_buys_both_keep_buy_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    _build_correlated_history(tmp_path, "HI_EV", "LO_EV", n=100)
    runs = [_make_run("HI_EV", 0.40), _make_run("LO_EV", 0.30)]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    orig = orch._decision_from_run

    def _patched(run):
        d = orig(run)
        d.ev_bps_of_dip = 100.0 if d.ticker == "HI_EV" else 80.0
        d.verdict = "BUY"
        return d
    monkeypatch.setattr(orch, "_decision_from_run", _patched)

    path = orch.generate_aggregate_dashboard(runs, None, run_dir)
    html = path.read_text()

    # Both rows present as BUYs (green pill, not purple REFUSED-CORRELATED)
    # Table rows use _VERDICT_COLORS["BUY"] = #1a7f37 (legend uses
    # a slightly different shade for the chip)
    buy_pill_count = html.count(
        'class="verdict" style="background:#1a7f37">BUY'
    )
    assert buy_pill_count >= 2  # both HI_EV and LO_EV rows

    # No row carries REFUSED-CORRELATED verdict
    refused_pill = (
        'class="verdict" style="background:#8957e5">REFUSED-CORRELATED'
    )
    assert refused_pill not in html


# =============================================================================
# 2. Lower-EV ticker gets the CORRELATED annotation in status_note
# =============================================================================

def test_lower_ev_ticker_status_note_contains_correlation(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    _build_correlated_history(tmp_path, "HI_EV", "LO_EV", n=100)
    runs = [_make_run("HI_EV", 0.40), _make_run("LO_EV", 0.30)]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    orig = orch._decision_from_run

    def _patched(run):
        d = orig(run)
        d.ev_bps_of_dip = 100.0 if d.ticker == "HI_EV" else 80.0
        d.verdict = "BUY"
        return d
    monkeypatch.setattr(orch, "_decision_from_run", _patched)

    path = orch.generate_aggregate_dashboard(runs, None, run_dir)
    html = path.read_text()
    assert "CORRELATED" in html


# =============================================================================
# 3. Original BUY status note (Dip → Rally) is preserved alongside
# =============================================================================

def test_correlation_annotation_appended_not_replacing_buy_note(tmp_path, monkeypatch):
    """The pre-gate status_note for a BUY is 'Dip $X → Rally $Y,
    P(round-trip) = Z%'. PR #74 appends correlation info; the
    original note must still be present."""
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    _build_correlated_history(tmp_path, "TOP_BUY", "LOW_BUY", n=100)
    runs = [_make_run("TOP_BUY", 0.40), _make_run("LOW_BUY", 0.30)]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    orig = orch._decision_from_run

    def _patched(run):
        d = orig(run)
        d.ev_bps_of_dip = 100.0 if d.ticker == "TOP_BUY" else 80.0
        d.verdict = "BUY"
        d.dip_target = 100.0
        d.rally_target = 110.0
        d.p_round_trip = 0.55
        d.status_note = (
            "Dip $100.00 → Rally $110.00, P(round-trip) = 55%"
        )
        return d
    monkeypatch.setattr(orch, "_decision_from_run", _patched)

    path = orch.generate_aggregate_dashboard(runs, None, run_dir)
    html = path.read_text()
    # Original BUY note preserved
    assert "Dip $100.00 → Rally $110.00" in html
    # Correlation note appended after separator
    assert "CORRELATED" in html


# =============================================================================
# 4. Log line indicates informational mode
# =============================================================================

def test_gate_log_says_informational(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    _build_correlated_history(tmp_path, "A_T", "B_T", n=100)
    runs = [_make_run("A_T", 0.40), _make_run("B_T", 0.30)]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    orig = orch._decision_from_run

    def _patched(run):
        d = orig(run)
        d.ev_bps_of_dip = 100.0 if d.ticker == "A_T" else 80.0
        d.verdict = "BUY"
        return d
    monkeypatch.setattr(orch, "_decision_from_run", _patched)

    orch.generate_aggregate_dashboard(runs, None, run_dir)
    captured = capsys.readouterr().out
    # Old log: "dropped N as substitute ideas"
    # New log: "INFORMATIONAL ... noted N correlation pair(s) ...
    # All BUYs remain visible"
    assert "INFORMATIONAL" in captured
    assert "noted" in captured.lower()
    assert "all buys remain visible" in captured.lower()


# =============================================================================
# 5. Uncorrelated BUYs have no annotation
# =============================================================================

def test_uncorrelated_buys_have_no_correlation_note(tmp_path, monkeypatch):
    """Two BUYs whose returns are uncorrelated → no CORRELATED note
    on either. Sanity check that the annotation only fires when
    there's actual correlation."""
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    # Build de-correlated price histories
    import csv as _csv
    from datetime import datetime, timedelta
    from src.engine import CSV_COLUMNS
    import random
    rng = random.Random(42)
    base = datetime(2026, 1, 1).date()

    def _write(ticker, mults):
        path = tmp_path / f"round_trip_history_{ticker}.csv"
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            price = 100.0
            for i, m in enumerate(mults):
                price *= (1 + m)
                row = {c: "" for c in CSV_COLUMNS}
                row["date"] = (base + timedelta(days=i)).strftime("%Y-%m-%d")
                row["spot"] = f"{price:.2f}"
                row["ev_pct_of_dip"] = "0.0060"
                w.writerow(row)

    # 100 days each, independent random walks
    a_mults = [rng.gauss(0, 0.02) for _ in range(100)]
    b_mults = [rng.gauss(0, 0.02) for _ in range(100)]
    _write("INDEP_A", a_mults)
    _write("INDEP_B", b_mults)

    runs = [_make_run("INDEP_A", 0.40), _make_run("INDEP_B", 0.30)]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    orig = orch._decision_from_run

    def _patched(run):
        d = orig(run)
        d.ev_bps_of_dip = 100.0 if d.ticker == "INDEP_A" else 80.0
        d.verdict = "BUY"
        return d
    monkeypatch.setattr(orch, "_decision_from_run", _patched)

    path = orch.generate_aggregate_dashboard(runs, None, run_dir)
    html = path.read_text()
    # Verify both stayed BUY
    buy_count = html.count(
        'class="verdict" style="background:#1a7f37">BUY'
    )
    assert buy_count == 2
    # No "⚠ CORRELATED:" annotation in row status_notes (the legend
    # text mentions CORRELATED so we can't just search for the word —
    # check for the row-annotation pattern specifically)
    assert "⚠ CORRELATED:" not in html


# =============================================================================
# 6. Legend updated to reflect informational mode
# =============================================================================

def test_legend_describes_correlated_as_informational():
    """Legend text should reflect that correlation is a NOTE, not a
    refusal. PR #74."""
    # Minimal render to grab legend HTML
    decisions = [
        orch.TickerDecision(
            ticker="X", sigma_class="MID", tier="T0",
            ambiguity=0.1, qualifies_for_t2_plus=True,
            spot=100.0, dip_target=98.0, rally_target=110.0,
            p_round_trip=0.6, ev_bps_of_dip=120.0,
            verdict="BUY", status_note="x",
        )
    ]
    html = orch._render_dashboard_html(decisions, None)
    # Legend text no longer says "one bet expressed twice"
    assert "one bet expressed twice" not in html
    # New language present
    assert "CORRELATED note" in html or "engine surfaces the correlation" in html
    # Indicates operator decides
    assert "operator decides" in html.lower() or "operator" in html.lower()

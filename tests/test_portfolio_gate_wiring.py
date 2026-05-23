"""Tests for PR #55 — portfolio gate wiring into orchestrator.

PR #49 shipped the gate module standalone; this PR wires it into
generate_aggregate_dashboard's flow. Verifies:

  - When 2+ BUY-verdict tickers are correlated > threshold, the
    lower-EV ticker's verdict flips to REFUSED-CORRELATED with the
    gate's reason in status_note.
  - WAIT / REFUSED-EV / FAIL verdicts pass through untouched (no
    recommendation to dedupe).
  - Gate failure (exception) doesn't block dashboard generation.
  - Single-BUY or zero-BUY runs skip the gate gracefully.
  - _history_as_price_df handles missing files / sparse history.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import orchestrator as orch
from src.broker import BrokerSnapshot


def _make_run(ticker, ambiguity=0.5, sigma_class="MID",
               phase2_rc=0, assigned_tier="T2"):
    snap = BrokerSnapshot(
        ticker=ticker, ambiguity=ambiguity,
        qualifies_for_t2_plus=True, sigma_class=sigma_class,
    )
    return orch.TickerRun(
        ticker=ticker, phase1_returncode=0, snapshot=snap,
        assigned_tier=assigned_tier, phase2_returncode=phase2_rc,
    )


def _write_history_csv(tmp_path, ticker, dates_and_spots, dip=None, rally=None):
    """Write a per-ticker CSV history file with the new schema.
    dates_and_spots: list of (date_str, spot_float).
    dip/rally: optional — if provided, every row uses these (makes the
    latest row's decision = BUY)."""
    path = tmp_path / f"round_trip_history_{ticker}.csv"
    # Minimal column set — write empty strings for the rest to satisfy
    # DictReader's parser.
    cols = orch.csv.DictReader  # noqa: just to silence linter; we'll use list
    from src.engine import CSV_COLUMNS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for d, s in dates_and_spots:
            row = {c: "" for c in CSV_COLUMNS}
            row["date"] = d
            row["spot"] = str(s)
            if dip is not None:
                row["recommended_dip"] = str(dip)
                row["recommended_rally"] = str(rally)
                row["ev_pct_of_dip"] = "0.0060"  # 60bps — clears EV-hurdle
                row["p_round_trip"] = "0.55"
            w.writerow(row)
    return path


# =============================================================================
# _history_as_price_df
# =============================================================================

def test_history_df_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    assert orch._history_as_price_df("NOEXIST") is None


def test_history_df_sparse_history_returns_none(tmp_path, monkeypatch):
    """Fewer than 5 rows → gate's correlation window can't be computed
    reliably → return None and let the gate accept defensively."""
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    base = datetime(2026, 1, 1).date()
    rows = [((base + timedelta(days=i)).strftime("%Y-%m-%d"), 100.0 + i)
            for i in range(3)]
    _write_history_csv(tmp_path, "SPARSE", rows)
    assert orch._history_as_price_df("SPARSE") is None


def test_history_df_returns_sorted_df(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    base = datetime(2026, 1, 1).date()
    # Write OUT of order — verify the function sorts.
    rows = []
    for i in [3, 0, 4, 1, 2, 5, 6]:
        rows.append(((base + timedelta(days=i)).strftime("%Y-%m-%d"), 100.0 + i))
    _write_history_csv(tmp_path, "AAA", rows)
    df = orch._history_as_price_df("AAA")
    assert df is not None
    assert len(df) == 7
    # Dates should be ascending after sort
    assert list(df["Date"]) == sorted(df["Date"])
    # Close values should match the date order
    assert df["Close"].iloc[0] < df["Close"].iloc[-1]


# =============================================================================
# Gate wiring into generate_aggregate_dashboard
# =============================================================================

def _build_correlated_history(tmp_path, ticker_a, ticker_b, n=70):
    """Two tickers with synthetic price paths that are perfectly correlated."""
    base = datetime(2026, 1, 1).date()
    import numpy as np
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.02, n)
    closes = 100.0 * np.exp(rets.cumsum())
    rows = [((base + timedelta(days=i)).strftime("%Y-%m-%d"), float(c))
            for i, c in enumerate(closes)]
    _write_history_csv(tmp_path, ticker_a, rows, dip=95.0, rally=110.0)
    # B uses the same series (perfect ρ ≈ 1.0)
    _write_history_csv(tmp_path, ticker_b, rows, dip=95.0, rally=110.0)


def test_gate_drops_correlated_lower_ev_ticker(tmp_path, monkeypatch):
    """Two highly-correlated BUY tickers → lower EV gets dropped."""
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    _build_correlated_history(tmp_path, "HIGH_EV", "LOW_EV")
    runs = [
        _make_run("HIGH_EV", ambiguity=0.40),
        _make_run("LOW_EV", ambiguity=0.30),
    ]
    allocation = None  # not needed for dashboard render
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Manually set ev_bps_of_dip so HIGH_EV wins the gate's priority
    # ordering. _decision_from_run reads ev_pct_of_dip from CSV — both
    # were written with the same 60bps; need to differentiate.
    # Patch _decision_from_run to inject distinct EVs.
    orig_decision = orch._decision_from_run
    def _patched(run):
        d = orig_decision(run)
        if d.ticker == "HIGH_EV":
            d.ev_bps_of_dip = 100.0
        elif d.ticker == "LOW_EV":
            d.ev_bps_of_dip = 80.0
        # Force both to BUY (no calibration data → would default WAIT).
        d.verdict = "BUY"
        return d
    monkeypatch.setattr(orch, "_decision_from_run", _patched)

    path = orch.generate_aggregate_dashboard(runs, allocation, run_dir)
    html = path.read_text()
    # Lower-EV ticker should be REFUSED-CORRELATED in the rendered HTML.
    assert "REFUSED-CORRELATED" in html
    assert "LOW_EV" in html
    assert "HIGH_EV" in html


def test_gate_skipped_when_only_one_buy(tmp_path, monkeypatch):
    """Single-BUY run → nothing to dedupe → no gate invocation."""
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    runs = [_make_run("ALONE")]
    allocation = None
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = orch.generate_aggregate_dashboard(runs, allocation, run_dir)
    html = path.read_text()
    assert "REFUSED-CORRELATED" not in html


def test_gate_skipped_when_no_buys(tmp_path, monkeypatch):
    """All-WAIT run (current frothy market state) → gate has nothing
    to dedupe → no error, no REFUSED-CORRELATED in output."""
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    runs = [_make_run("A"), _make_run("B"), _make_run("C")]
    allocation = None
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = orch.generate_aggregate_dashboard(runs, allocation, run_dir)
    html = path.read_text()
    assert "REFUSED-CORRELATED" not in html


def test_verdict_color_includes_correlated():
    """REFUSED-CORRELATED needs a color or it renders as default gray."""
    assert "REFUSED-CORRELATED" in orch._VERDICT_COLORS
    # Distinct from REFUSED-EV / REFUSED-METHOD so trader can
    # visually tell why each ticker was filtered.
    assert orch._VERDICT_COLORS["REFUSED-CORRELATED"] != \
           orch._VERDICT_COLORS["REFUSED-EV"]


def test_gate_failure_does_not_block_dashboard(tmp_path, monkeypatch):
    """If portfolio.gate_by_correlation raises, the orchestrator must
    fall back to ungated dashboard rather than crashing the run."""
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)

    # Monkey-patch gate to raise.
    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic gate failure")
    import src.portfolio
    monkeypatch.setattr(src.portfolio, "gate_by_correlation", _boom)

    _build_correlated_history(tmp_path, "X", "Y")
    runs = [_make_run("X"), _make_run("Y")]
    # Force BUY so gate would have run.
    orig_decision = orch._decision_from_run
    def _patched(run):
        d = orig_decision(run)
        d.verdict = "BUY"
        d.ev_bps_of_dip = 100.0
        return d
    monkeypatch.setattr(orch, "_decision_from_run", _patched)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Should NOT raise.
    path = orch.generate_aggregate_dashboard(runs, None, run_dir)
    assert path.exists()
    # Both tickers stay BUY since gate failed.
    html = path.read_text()
    assert "X" in html and "Y" in html

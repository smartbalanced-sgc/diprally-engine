"""Tests for PR #77 — audit findings #6 and #8.

#6: Portfolio gate silently accepts on insufficient history.
    Pre-fix: a ticker with too few bars to compute correlation was
    appended to `accepted` with no record. Operator saw a BUY pass the
    gate when in fact the gate could not evaluate it (PR #75-class
    silent failure). New-listing names (SNDK, CRWV, NBIS, ARM, VELO)
    were the most affected.
    Fix: GateResult gains `bypassed: dict[str, str]`. The orchestrator
    annotates the BUY's status_note as "⚠ LIMITED-HISTORY: ..." so the
    dashboard makes the bypass visible.

#8: AI cache replay ignores tier.
    Pre-fix: cache keyed only on (ticker, last_trading_day) with spot-
    move invalidation. A T1 cached payload (Pass 1 only) would happily
    replay for a later T3 run; engine logged tier=T3 at $0.00 while
    actually serving T1 data (no Pass 2 / verification / stress).
    Fix: cache persists `tier_name`; `get_cached(..., current_tier=T)`
    refuses a strictly-lower-tier payload.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# Finding #6 — gate.bypassed surfaced and orchestrator annotates row
# =============================================================================

from src.portfolio import (
    GateResult,
    PortfolioRecommendation,
    format_gate_result,
    gate_by_correlation,
)


def _rec(ticker, ev_bps, rets):
    """Build a PortfolioRecommendation from explicit daily-return series."""
    closes = [100.0]
    for r in rets:
        closes.append(closes[-1] * np.exp(r))
    df = pd.DataFrame({
        "Date": pd.date_range(end="2026-05-22", periods=len(closes), freq="B"),
        "Close": closes,
    })
    return PortfolioRecommendation(ticker=ticker, ev_bps=ev_bps, history_df=df)


def test_short_history_populates_gate_bypassed():
    """5 bars, 60d window → gate cannot compute correlation → ticker
    goes to gate.bypassed AND remains in gate.accepted (defensive)."""
    rng = np.random.default_rng(seed=11)
    short_rets = rng.normal(0, 0.02, 5).tolist()
    long_rets = rng.normal(0, 0.02, 80).tolist()
    recs = [
        _rec("LONG", 100.0, long_rets),
        _rec("SHORT", 50.0, short_rets),
    ]
    result = gate_by_correlation(recs, threshold=0.85, window_days=60)
    assert "SHORT" in result.bypassed
    assert "SHORT" in result.accepted  # still surfaced to operator
    assert "insufficient history" in result.bypassed["SHORT"]
    # The full-history ticker is NOT bypassed.
    assert "LONG" not in result.bypassed


def test_format_gate_result_includes_bypassed_section():
    """Operator-readable summary must list the bypassed tickers so the
    bypass isn't silent in stdout / orchestrator log."""
    result = GateResult(
        accepted=["FULL", "NEW_IPO"],
        dropped={},
        bypassed={"NEW_IPO": "insufficient history (15 bars, need 91)"},
    )
    rec_full = _rec("FULL", 100.0, [0.01] * 80)
    rec_new = _rec("NEW_IPO", 50.0, [0.01] * 15)
    txt = format_gate_result(result, [rec_full, rec_new])
    assert "Bypassed" in txt
    assert "NEW_IPO" in txt
    assert "insufficient history" in txt


def test_orchestrator_annotates_limited_history_on_dashboard(tmp_path, monkeypatch):
    """End-to-end: a BUY whose history_df is shorter than the gate's
    window must get a '⚠ LIMITED-HISTORY' annotation in its row, so
    the trader sees that the correlation check did NOT run for it."""
    from src import orchestrator as orch
    from src.broker import BrokerSnapshot

    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    # Build two BUYs:
    #   FULL_HIST — 100 bars of synthetic price history (gate evaluates).
    #   IPO       — 10 bars only (gate bypasses).
    import csv as _csv
    from datetime import datetime, timedelta
    from src.engine import CSV_COLUMNS
    base = datetime(2026, 1, 1).date()

    def _write(ticker, n_bars):
        path = tmp_path / f"round_trip_history_{ticker}.csv"
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for i in range(n_bars):
                row = {c: "" for c in CSV_COLUMNS}
                row["date"] = (base + timedelta(days=i)).strftime("%Y-%m-%d")
                row["spot"] = f"{100.0 + i * 0.1:.2f}"
                row["ev_pct_of_dip"] = "0.006"
                w.writerow(row)

    _write("FULL_HIST", 100)
    _write("IPO", 10)  # below the 30-bar floor → _history_as_price_df returns None

    def _make_run(t):
        snap = BrokerSnapshot(
            ticker=t, ambiguity=0.4,
            qualifies_for_t2_plus=True, sigma_class="MID",
        )
        return orch.TickerRun(
            ticker=t, phase1_returncode=0, snapshot=snap,
            assigned_tier="T2", phase2_returncode=0,
        )

    runs = [_make_run("FULL_HIST"), _make_run("IPO")]

    orig = orch._decision_from_run

    def _patched(run):
        d = orig(run)
        d.ev_bps_of_dip = 100.0 if d.ticker == "FULL_HIST" else 80.0
        d.verdict = "BUY"
        return d

    monkeypatch.setattr(orch, "_decision_from_run", _patched)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = orch.generate_aggregate_dashboard(runs, None, run_dir)
    html = path.read_text()

    # The IPO row must carry the LIMITED-HISTORY badge.
    assert "LIMITED-HISTORY" in html
    # Both stay BUY (sacred #6, PR #74 — informational only).
    assert html.count(
        'class="verdict" style="background:#1a7f37">BUY'
    ) == 2


def test_default_gate_result_has_empty_bypassed():
    """Backwards compat — older callers building GateResult directly
    should still work (bypassed defaults to empty dict)."""
    result = GateResult(accepted=["A"], dropped={})
    assert result.bypassed == {}


# =============================================================================
# Finding #8 — AI cache tier validation
# =============================================================================

from src import ai_cache


@pytest.fixture
def cache_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_cache, "_cache_dir", lambda: tmp_path)
    return tmp_path


def test_cache_hit_when_tier_matches(cache_tmp):
    """Same-tier replay is the typical case (e.g. T2 caches its payload,
    same-day T2 rerun replays it)."""
    ai_cache.save("AMAT", 100.0, {
        "tier_name": "T2",
        "pass1_raw": {"x": 1},
        "pass2_raw": {"y": 2},
    })
    result = ai_cache.get_cached("AMAT", 100.0, current_tier="T2")
    assert result is not None
    assert result["tier_name"] == "T2"


def test_cache_hit_when_cached_tier_higher(cache_tmp):
    """A T3 cache contains everything a T2 run needs → valid replay."""
    ai_cache.save("AMAT", 100.0, {
        "tier_name": "T3",
        "pass1_raw": {"x": 1},
        "pass2_raw": {"y": 2},
        "stress_results": [{"z": 3}],
    })
    result = ai_cache.get_cached("AMAT", 100.0, current_tier="T2")
    assert result is not None
    assert result["tier_name"] == "T3"


def test_cache_miss_when_cached_tier_lower(cache_tmp, capsys):
    """T1 cache cannot serve T3 — Pass 2 / verification / stress missing.
    Pre-PR-#77 the engine would replay the T1 payload and claim tier=T3
    at $0.00 cost. Now it invalidates and forces a fresh run."""
    ai_cache.save("AMAT", 100.0, {
        "tier_name": "T1",
        "pass1_raw": {"x": 1},
        "pass2_raw": None,
    })
    result = ai_cache.get_cached("AMAT", 100.0, current_tier="T3")
    assert result is None
    captured = capsys.readouterr().out
    assert "cache invalidated" in captured.lower()
    assert "tier" in captured.lower()


def test_cache_miss_when_legacy_payload_has_no_tier(cache_tmp, capsys):
    """Pre-PR-#77 caches have no `tier_name` field — be safe, invalidate
    rather than risk replaying an unknown payload."""
    # Write a payload directly (bypassing ai_cache.save which now sets
    # the tier_name on demand — except we DON'T pass one).
    import json as _json
    path = cache_tmp / "AMAT_2026-05-22.json"
    path.write_text(_json.dumps({
        "spot": 100.0,
        "ticker": "AMAT",
        "date": "2026-05-22",
        # NO tier_name
        "pass1_raw": {"legacy": True},
    }))
    # ai_cache.today_str() returns last_trading_day → 2026-05-22 on a
    # Mon 2026-05-25 holiday. Use the filename match by passing
    # date_str explicitly to avoid wall-clock fragility.
    result = ai_cache.get_cached("AMAT", 100.0, date_str="2026-05-22",
                                  current_tier="T2")
    assert result is None
    captured = capsys.readouterr().out
    assert "cache invalidated" in captured.lower()


def test_cache_works_without_current_tier_arg(cache_tmp):
    """Backwards compat: if no current_tier is passed (e.g. older
    caller), the tier check is skipped — same behavior as before."""
    ai_cache.save("AMAT", 100.0, {
        "tier_name": "T1",
        "pass1_raw": {"x": 1},
    })
    result = ai_cache.get_cached("AMAT", 100.0)  # no current_tier
    assert result is not None


def test_cache_save_persists_tier_name(cache_tmp):
    """ai_cache.save preserves the tier_name field operators rely on
    in PR #77's invalidation logic."""
    ai_cache.save("AMAT", 100.0, {"tier_name": "T3", "pass1_raw": {}})
    # Read the file directly to confirm persistence.
    path = next(cache_tmp.glob("AMAT_*.json"))
    raw = json.loads(path.read_text())
    assert raw["tier_name"] == "T3"


def test_tier_satisfies_helper_orderings():
    """T3 ≥ T2 ≥ T1 ≥ T0. Strict — T2 cache cannot serve T3."""
    from src.ai_cache import _tier_satisfies
    assert _tier_satisfies("T3", "T2") is True
    assert _tier_satisfies("T2", "T2") is True
    assert _tier_satisfies("T1", "T2") is False
    assert _tier_satisfies("T0", "T1") is False
    # None cached_tier (legacy) → never satisfies.
    assert _tier_satisfies(None, "T1") is False
    # None current_tier → caller opted out.
    assert _tier_satisfies("T1", None) is True

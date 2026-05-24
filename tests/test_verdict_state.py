"""Tests for the engine-level verdict_state computation + CSV
persistence (2026-05-24 audit fix).

Audit found that the orchestrator's aggregate dashboard reconstructed
each ticker's verdict from CSV dip/EV alone, which silently
misclassified sacred-#14 (trend filter), sacred-#18 (parabola filter),
and sacred-#16 (method-disagreement) refusals as BUY or WAIT. The
engine now writes a `verdict_state` column to the CSV row directly,
mirroring the reporter headline_card decision tree — single source of
truth across per-ticker report + aggregate dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.engine import _compute_verdict_state, ENGINE_VERDICT_STATES, CSV_COLUMNS


def _best(net_ev=1.0, ev_pct=0.01, p_dip=0.7, p_rally=0.8):
    """Stand-in for the AltDipRallyPair returned by scan_dip_rally_grid."""
    return SimpleNamespace(
        dip_price=100.0, rally_price=120.0,
        p_dip_touched=p_dip, p_rally_given_dip=p_rally,
        p_round_trip=p_dip * p_rally,
        net_ev_per_share=net_ev, ev_pct_of_dip=ev_pct,
    )


# =============================================================================
# Verdict computation — sacred-decision refusals
# =============================================================================

def test_trend_filter_refused_wins():
    """Sacred #14 trend filter has highest priority — even with parabola
    flag also set (impossible in practice; defensive), trend wins per
    headline_card priority order."""
    v = _compute_verdict_state(
        best=_best(), met_threshold_strict=True, method_check={},
        trend_filter_refused=True, parabola_filter_refused=False,
        ev_hurdle_refused=False,
    )
    assert v == "REFUSED-TREND"


def test_parabola_filter_refused():
    """Sacred #18 parabola filter."""
    v = _compute_verdict_state(
        best=_best(), met_threshold_strict=True, method_check={},
        trend_filter_refused=False, parabola_filter_refused=True,
        ev_hurdle_refused=False,
    )
    assert v == "REFUSED-PARABOLA"


def test_method_disagreement_refused():
    """Sacred #16 method disagreement: engine sets best=None when refusal
    fires AND method_check is not the anchor-verification path."""
    v = _compute_verdict_state(
        best=None, met_threshold_strict=False,
        method_check={"refused": True, "is_anchor": False},
        trend_filter_refused=False, parabola_filter_refused=False,
        ev_hurdle_refused=False,
    )
    assert v == "REFUSED-METHOD"


def test_method_anchor_not_treated_as_refusal():
    """When best is None because no qualifying pair existed (anchor
    verification path), the verdict is WAIT, NOT REFUSED-METHOD —
    the anchor check is a sacred-#8 verification, not a refusal."""
    v = _compute_verdict_state(
        best=None, met_threshold_strict=False,
        method_check={"refused": False, "is_anchor": True},
        trend_filter_refused=False, parabola_filter_refused=False,
        ev_hurdle_refused=False,
    )
    assert v == "WAIT"


def test_ev_hurdle_refused():
    """Sacred #13 EV-hurdle (50bps of dip)."""
    v = _compute_verdict_state(
        best=_best(ev_pct=0.001), met_threshold_strict=True,
        method_check={},
        trend_filter_refused=False, parabola_filter_refused=False,
        ev_hurdle_refused=True,
    )
    assert v == "REFUSED-EV"


def test_clean_buy():
    """All gates clear, conviction strict, EV positive."""
    v = _compute_verdict_state(
        best=_best(net_ev=2.5, ev_pct=0.02),
        met_threshold_strict=True, method_check={},
        trend_filter_refused=False, parabola_filter_refused=False,
        ev_hurdle_refused=False,
    )
    assert v == "BUY"


def test_below_threshold_fallback():
    """Best-EV fallback when no pair met strict conviction. Operator
    sees the pair but is told not to trade."""
    v = _compute_verdict_state(
        best=_best(), met_threshold_strict=False, method_check={},
        trend_filter_refused=False, parabola_filter_refused=False,
        ev_hurdle_refused=False,
    )
    assert v == "BELOW-THRESHOLD"


def test_negative_ev_warning():
    """Conviction met, EV-hurdle didn't fire (above 50bps? edge case),
    but net_ev_per_share went negative. SKIP signal."""
    v = _compute_verdict_state(
        best=_best(net_ev=-0.5), met_threshold_strict=True,
        method_check={},
        trend_filter_refused=False, parabola_filter_refused=False,
        ev_hurdle_refused=False,
    )
    assert v == "NEGATIVE-EV"


def test_wait_no_qualifying_pair():
    """Grid scan produced no candidate at all."""
    v = _compute_verdict_state(
        best=None, met_threshold_strict=False, method_check={},
        trend_filter_refused=False, parabola_filter_refused=False,
        ev_hurdle_refused=False,
    )
    assert v == "WAIT"


# =============================================================================
# Verdict-state CSV plumbing
# =============================================================================

def test_verdict_state_in_csv_columns():
    """The 2026-05-24 audit added this column; it must be in CSV_COLUMNS
    so DictWriter writes it on every engine row."""
    assert "verdict_state" in CSV_COLUMNS


def test_all_engine_verdict_states_covered():
    """Each of the 8 engine verdict states must be producible by
    _compute_verdict_state. Doc-and-implementation sync check."""
    seen = set()
    inputs = [
        # (kwargs, expected)
        (dict(best=_best(), met_threshold_strict=True, method_check={},
              trend_filter_refused=True, parabola_filter_refused=False,
              ev_hurdle_refused=False), "REFUSED-TREND"),
        (dict(best=_best(), met_threshold_strict=True, method_check={},
              trend_filter_refused=False, parabola_filter_refused=True,
              ev_hurdle_refused=False), "REFUSED-PARABOLA"),
        (dict(best=None, met_threshold_strict=False,
              method_check={"refused": True, "is_anchor": False},
              trend_filter_refused=False, parabola_filter_refused=False,
              ev_hurdle_refused=False), "REFUSED-METHOD"),
        (dict(best=_best(), met_threshold_strict=True, method_check={},
              trend_filter_refused=False, parabola_filter_refused=False,
              ev_hurdle_refused=True), "REFUSED-EV"),
        (dict(best=None, met_threshold_strict=False, method_check={},
              trend_filter_refused=False, parabola_filter_refused=False,
              ev_hurdle_refused=False), "WAIT"),
        (dict(best=_best(), met_threshold_strict=False, method_check={},
              trend_filter_refused=False, parabola_filter_refused=False,
              ev_hurdle_refused=False), "BELOW-THRESHOLD"),
        (dict(best=_best(net_ev=-1.0), met_threshold_strict=True,
              method_check={},
              trend_filter_refused=False, parabola_filter_refused=False,
              ev_hurdle_refused=False), "NEGATIVE-EV"),
        (dict(best=_best(), met_threshold_strict=True, method_check={},
              trend_filter_refused=False, parabola_filter_refused=False,
              ev_hurdle_refused=False), "BUY"),
    ]
    for kw, expected in inputs:
        actual = _compute_verdict_state(**kw)
        assert actual == expected, f"{kw} → {actual}, expected {expected}"
        seen.add(expected)
    assert seen == set(ENGINE_VERDICT_STATES)


# =============================================================================
# Orchestrator consumption of verdict_state
# =============================================================================

def test_orchestrator_reads_verdict_state_from_row():
    """When CSV has verdict_state, orchestrator uses it directly
    (no reconstruction). Each engine state surfaces with its
    operator-friendly status_note."""
    from src.orchestrator import _verdict_from_row

    # Mock TickerDecision
    decision = SimpleNamespace(
        dip_target=100.0, rally_target=120.0,
        ev_bps_of_dip=200.0, p_round_trip=0.55,
    )
    for state in ("REFUSED-TREND", "REFUSED-PARABOLA", "REFUSED-METHOD",
                  "REFUSED-EV", "WAIT", "BELOW-THRESHOLD", "NEGATIVE-EV", "BUY"):
        row = {"verdict_state": state}
        v, note = _verdict_from_row(row, decision)
        assert v == state
        assert note, f"empty note for {state}"


def test_orchestrator_falls_back_to_legacy_when_column_missing():
    """Pre-audit CSV rows have no verdict_state. Dashboard must still
    work — falls back to reconstructing from dip/EV. The fallback IS
    the old buggy logic, but it only fires on history rows written
    before the audit fix landed; new rows always have the column."""
    from src.orchestrator import _verdict_from_row

    # Row with dip and good EV → fallback says BUY
    decision = SimpleNamespace(
        dip_target=100.0, rally_target=120.0, ev_bps_of_dip=200.0,
        p_round_trip=0.55,
    )
    v, note = _verdict_from_row({"verdict_state": ""}, decision)
    assert v == "BUY"

    # Row with no dip → fallback says WAIT
    decision_no_dip = SimpleNamespace(
        dip_target=None, rally_target=None, ev_bps_of_dip=None,
        p_round_trip=None,
    )
    v, note = _verdict_from_row({"verdict_state": ""}, decision_no_dip)
    assert v == "WAIT"


def test_orchestrator_color_table_covers_engine_states():
    """All 8 engine verdict_states + orchestrator-level states must
    have colors. Missing slot → black/undefined in HTML."""
    from src.orchestrator import _VERDICT_COLORS
    for state in ENGINE_VERDICT_STATES:
        assert state in _VERDICT_COLORS, f"{state} missing color"
    for orch_state in ("FAIL", "DELISTED", "REFUSED-CORRELATED"):
        assert orch_state in _VERDICT_COLORS

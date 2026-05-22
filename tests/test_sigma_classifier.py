"""Tests for src/sigma_classifier.py — W3 σ-class auto-detection.

Sacred references:
  - #17: boundaries + class-specific conviction live in config/diprally.yaml
  - W3 plan: data wins (auto), registry hint is advisory

Boundary semantics (>= for the lower bound):
  σ < high_min                            → MID
  high_min <= σ < extreme_min             → HIGH
  σ >= extreme_min                        → EXTREME
  σ is None / σ <= 0                      → MID  (GARCH-failure fallback)

With config/diprally.yaml defaults (extreme_min=0.95, high_min=0.50):
  σ = 0.49  → MID
  σ = 0.50  → HIGH  (boundary, inclusive)
  σ = 0.94  → HIGH
  σ = 0.95  → EXTREME (boundary, inclusive)
  σ = 1.20  → EXTREME
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import SIGMA_CLASS_BOUNDARIES, SIGMA_CLASSES
from src.sigma_classifier import (
    class_conviction,
    classify_sigma,
    reconcile_with_registry,
)


def test_classify_below_high_min_is_mid():
    assert classify_sigma(0.49) == "MID"
    assert classify_sigma(0.30) == "MID"
    assert classify_sigma(0.10) == "MID"


def test_classify_at_high_min_is_high():
    # Inclusive >= boundary.
    assert classify_sigma(SIGMA_CLASS_BOUNDARIES.high_min) == "HIGH"


def test_classify_between_high_and_extreme_is_high():
    assert classify_sigma(0.60) == "HIGH"
    assert classify_sigma(0.80) == "HIGH"
    assert classify_sigma(0.94) == "HIGH"


def test_classify_at_extreme_min_is_extreme():
    # σ exactly at extreme_min classifies as EXTREME (conservative).
    assert classify_sigma(SIGMA_CLASS_BOUNDARIES.extreme_min) == "EXTREME"


def test_classify_above_extreme_min_is_extreme():
    assert classify_sigma(1.20) == "EXTREME"
    assert classify_sigma(2.50) == "EXTREME"


def test_classify_none_falls_back_to_mid():
    # GARCH-fit-failure path — sacred conservative fallback.
    assert classify_sigma(None) == "MID"


def test_classify_zero_or_negative_falls_back_to_mid():
    assert classify_sigma(0.0) == "MID"
    assert classify_sigma(-0.5) == "MID"


def test_class_conviction_returns_yaml_values():
    # The numbers come from config/diprally.yaml; we assert via the loader
    # so the test stays valid if Jesse retunes the table.
    for cls in ("EXTREME", "HIGH", "MID"):
        dip, rally = class_conviction(cls)
        entry = SIGMA_CLASSES[cls]
        assert dip == entry.conviction.dip
        assert rally == entry.conviction.rally_conditional


def test_class_conviction_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        class_conviction("LOW")


def test_reconcile_match_returns_no_note():
    # INTC is MID in the registry per CLAUDE.md. classify_sigma(0.30) → MID.
    auto = classify_sigma(0.30)
    effective, note = reconcile_with_registry("INTC", auto)
    assert effective == auto
    assert note is None


def test_reconcile_mismatch_returns_note():
    # Force a mismatch: classify INTC as EXTREME even though registry says MID.
    effective, note = reconcile_with_registry("INTC", "EXTREME")
    assert effective == "EXTREME"  # auto wins
    assert note is not None
    assert "EXTREME" in note
    assert "MID" in note


def test_reconcile_unknown_ticker_no_mismatch():
    # Ticker not in universe: no hint, so no mismatch advisory.
    effective, note = reconcile_with_registry("NOTREAL", "HIGH")
    assert effective == "HIGH"
    assert note is None


# =============================================================================
# W3 PR #22 — per-class grid sizing (D-W3-1)
# =============================================================================

def test_per_class_grid_fields_present():
    """Each class entry exposes the full grid sizing block."""
    for cls in ("EXTREME", "HIGH", "MID"):
        g = SIGMA_CLASSES[cls].grid
        assert 0.0 < g.dip_step_pct < 1.0
        assert 0.0 < g.rally_step_pct < 1.0
        assert 0.0 < g.dip_max_depth_pct < 1.0
        assert g.rally_max_reach_pct > 0.0


def test_per_class_depth_widens_with_volatility():
    """EXTREME class scans deeper than HIGH, which scans deeper than MID.
    Sanity-check the σ-class hierarchy in the YAML — if Jesse retunes,
    the relative ordering should still hold or this test must be
    updated alongside it."""
    extreme = SIGMA_CLASSES["EXTREME"].grid
    high = SIGMA_CLASSES["HIGH"].grid
    mid = SIGMA_CLASSES["MID"].grid
    assert extreme.dip_max_depth_pct > high.dip_max_depth_pct > mid.dip_max_depth_pct
    assert extreme.rally_max_reach_pct > high.rally_max_reach_pct > mid.rally_max_reach_pct
    assert extreme.dip_step_pct > high.dip_step_pct > mid.dip_step_pct


# =============================================================================
# W3 PR #23 — per-class friction in bps
# =============================================================================

def test_per_class_friction_present_and_ordered():
    """Each class exposes friction_bps_round_trip; EXTREME has the
    highest friction (widest ticks, lowest liquidity, deepest impact)
    and MID the lowest."""
    extreme = SIGMA_CLASSES["EXTREME"].friction_bps_round_trip
    high = SIGMA_CLASSES["HIGH"].friction_bps_round_trip
    mid = SIGMA_CLASSES["MID"].friction_bps_round_trip
    assert extreme > 0 and high > 0 and mid > 0
    assert extreme > high > mid, (
        f"Friction ordering wrong: EXTREME={extreme} HIGH={high} MID={mid}"
    )


def test_friction_in_bps_is_price_agnostic():
    """The whole point of moving from $2/share to bps: same fractional
    cost regardless of price level. Apply the EXTREME bps to a $13
    stock and a $1500 stock at symmetric dip/rally pcts and verify the
    friction expressed as a fraction of average notional is identical."""
    bps = SIGMA_CLASSES["EXTREME"].friction_bps_round_trip
    # Stock A: $13 spot, 30% dip / 40% rally
    dipA, rallyA = 13.0 * 0.70, 13.0 * 1.40
    fA = (dipA + rallyA) / 2.0 * bps / 10000.0
    fracA = fA / ((dipA + rallyA) / 2.0)
    # Stock B: $1500 spot, same fractional dip/rally
    dipB, rallyB = 1500.0 * 0.70, 1500.0 * 1.40
    fB = (dipB + rallyB) / 2.0 * bps / 10000.0
    fracB = fB / ((dipB + rallyB) / 2.0)
    assert abs(fracA - fracB) < 1e-12
    assert abs(fracA - bps / 10000.0) < 1e-12


def test_per_class_grid_yields_tractable_point_count():
    """Step + depth combination yields a tractable candidate-pair count.
    Lower bound (≥30 cells per dimension) ensures a fine-enough EV
    gradient that the optimum isn't a discretisation artifact (per
    D-W3-1 acceptance criteria). Upper bound (≤200 cells per
    dimension) guards against accidental grid explosion."""
    for cls in ("EXTREME", "HIGH", "MID"):
        g = SIGMA_CLASSES[cls].grid
        n_dip_points = int(g.dip_max_depth_pct / g.dip_step_pct)
        n_rally_points = int(g.rally_max_reach_pct / g.rally_step_pct)
        assert 30 <= n_dip_points <= 200, (
            f"{cls} dip grid yields {n_dip_points} points — out of safe range"
        )
        assert 30 <= n_rally_points <= 200, (
            f"{cls} rally grid yields {n_rally_points} points — out of safe range"
        )

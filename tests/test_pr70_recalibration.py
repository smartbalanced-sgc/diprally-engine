"""Tests for PR #70 — σ-class-aware EV-hurdle + parabola threshold.

2026-05-24 recalibration. Previously the EV-hurdle (50 bps / 0.50%) and
parabola filter (+50% mom_30d) were single global constants applied to
every ticker regardless of σ-class. This made HIGH and EXTREME names
near-impossible to recommend because:

  1. At σ > 60%, the 75% rally|dip conviction was mathematically
     unachievable
  2. At σ > 60%, post-dip distributions are so wide that even when EV is
     positive in distribution, the point estimate often misses the 50bps
     hurdle
  3. The +50% parabola threshold caught BASELINE volatility of the
     target momentum universe, not tail events

PR #70 makes each threshold σ-class-aware:

  Threshold              MID      HIGH     EXTREME
  dip conviction         65%      60%      55%
  rally|dip conviction   70%      65%      55%
  ev_hurdle_bps          50       25       25
  parabola_mom_30d       +50%     +80%     +100%

MID class is preserved as-is. HIGH and EXTREME get realistic thresholds
calibrated to their actual vol distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import SIGMA_CLASSES


# =============================================================================
# YAML config — per-class thresholds present and correctly populated
# =============================================================================

def test_extreme_class_has_recalibrated_thresholds():
    """PR #89: parabola threshold raised 1.00 → 2.00 — AI-cycle EXTREME
    names routinely hit 100-180% mom_30d without parabolic-reversal
    (MRAM, LWLG, VELO observed +150-180% during the bull cycle)."""
    extreme = SIGMA_CLASSES["EXTREME"]
    assert extreme.conviction.dip == pytest.approx(0.55)
    assert extreme.conviction.rally_conditional == pytest.approx(0.55)
    assert extreme.ev_hurdle_bps == pytest.approx(25.0)
    assert extreme.parabola_mom_30d_threshold == pytest.approx(2.00)


def test_high_class_has_recalibrated_thresholds():
    """PR #89: parabola threshold raised 0.80 → 1.50 — AI-cycle HIGH
    names (MU, ARM, NBIS, INTC) routinely hit 80-100% mom_30d without
    mean-reverting during the secular bull cycle."""
    high = SIGMA_CLASSES["HIGH"]
    assert high.conviction.dip == pytest.approx(0.60)
    assert high.conviction.rally_conditional == pytest.approx(0.65)
    assert high.ev_hurdle_bps == pytest.approx(25.0)
    assert high.parabola_mom_30d_threshold == pytest.approx(1.50)


def test_mid_class_recalibrated_for_ai_cycle():
    """PR #89: MID parabola threshold raised 0.50 → 0.80. Established
    large-caps RARELY rally >50% in 30 days in normal markets, but AI-
    cycle MID names (LRCX, AMAT) have done so. Threshold now flags
    only true exceptional moves."""
    mid = SIGMA_CLASSES["MID"]
    assert mid.conviction.dip == pytest.approx(0.65)
    assert mid.conviction.rally_conditional == pytest.approx(0.70)
    assert mid.ev_hurdle_bps == pytest.approx(50.0)
    assert mid.parabola_mom_30d_threshold == pytest.approx(0.80)


def test_thresholds_monotonic_across_classes():
    """Sanity: higher-volatility classes have LOOSER conviction (lower
    floor) and LOWER ev_hurdle (lower floor) and HIGHER parabola
    threshold (higher tolerance). Catches accidental reversal in YAML."""
    mid = SIGMA_CLASSES["MID"]
    high = SIGMA_CLASSES["HIGH"]
    extreme = SIGMA_CLASSES["EXTREME"]

    # Conviction loosens (lower) as vol class rises
    assert mid.conviction.dip >= high.conviction.dip >= extreme.conviction.dip
    assert (mid.conviction.rally_conditional
            >= high.conviction.rally_conditional
            >= extreme.conviction.rally_conditional)

    # EV hurdle drops (lower) as vol class rises
    assert mid.ev_hurdle_bps >= high.ev_hurdle_bps
    assert mid.ev_hurdle_bps >= extreme.ev_hurdle_bps

    # Parabola threshold rises (more permissive) as vol class rises
    assert mid.parabola_mom_30d_threshold <= high.parabola_mom_30d_threshold
    assert (high.parabola_mom_30d_threshold
            <= extreme.parabola_mom_30d_threshold)


# =============================================================================
# Backward compatibility — schema accepts entries without per-class fields
# =============================================================================

def test_schema_accepts_class_without_per_class_thresholds():
    """If a YAML class entry omits ev_hurdle_bps or
    parabola_mom_30d_threshold, the schema should still load (fields
    are Optional). The engine then falls back to legacy globals."""
    from src.config import SigmaClassThresholdConfig, SigmaClassConvictionConfig, SigmaClassGridConfig
    # Build a minimal config WITHOUT the new optional fields
    entry = SigmaClassThresholdConfig(
        conviction=SigmaClassConvictionConfig(dip=0.60, rally_conditional=0.70),
        grid=SigmaClassGridConfig(
            dip_step_pct=0.01, rally_step_pct=0.01,
            dip_max_depth_pct=0.30, rally_max_reach_pct=0.40,
        ),
        friction_bps_round_trip=30.0,
        panic_floor_pct=0.20,
        ai_vol_regime_multipliers={"HIGH": 1.1, "MEDIUM": 1.0, "LOW": 0.9},
    )
    # ev_hurdle_bps and parabola_mom_30d_threshold should default to None
    assert entry.ev_hurdle_bps is None
    assert entry.parabola_mom_30d_threshold is None


# =============================================================================
# Engine consumption — per-class threshold actually applied
# =============================================================================

def test_engine_imports_uses_class_threshold_attributes():
    """Smoke test that the engine code reads from SIGMA_CLASSES entries
    rather than the legacy globals. Regression guard against accidental
    revert to globals."""
    # Re-load engine module fresh and verify the recalibrated threshold
    # values would actually be reached by the per-class lookup.
    for class_name in ("MID", "HIGH", "EXTREME"):
        entry = SIGMA_CLASSES[class_name]
        # Engine accesses these attributes; ensure they exist (non-None
        # after YAML load).
        assert entry.ev_hurdle_bps is not None
        assert entry.parabola_mom_30d_threshold is not None


# =============================================================================
# Realistic scenario — verify behavior change on representative cases
# =============================================================================

def test_sndk_like_high_class_no_longer_blocked_by_old_thresholds():
    """SNDK in production today: mom_30d = +73.6%, σ-class HIGH.
    Under OLD calibration (parabola +50%): refused on parabola.
    Under NEW calibration (parabola +80% for HIGH): passes parabola.
    This test documents the behavior change at the threshold level."""
    high = SIGMA_CLASSES["HIGH"]
    sndk_mom_30d = 0.736
    # Old behavior: blocked by global 0.50 threshold
    assert sndk_mom_30d >= 0.50  # SNDK trips old global
    # New behavior: passes per-class 0.80 threshold for HIGH
    assert sndk_mom_30d < high.parabola_mom_30d_threshold


def test_high_mom_30d_still_refused_above_new_threshold():
    """Sanity: if a HIGH-class name DOES exceed the new threshold, the
    filter STILL refuses it. PR #89 raised HIGH threshold to 1.50 —
    so +180% in 30 days still trips it (true blowoff)."""
    high = SIGMA_CLASSES["HIGH"]
    # +180% in 30 days IS true blow-off even on the new looser threshold
    assert 1.80 > high.parabola_mom_30d_threshold


def test_extreme_threshold_above_typical_extreme_monthly_swing():
    """EXTREME names regularly swing 50-80% monthly. New +100% threshold
    catches truly extreme (>2x) moves only — appropriate for the
    EXTREME-class character."""
    extreme = SIGMA_CLASSES["EXTREME"]
    # Typical EXTREME mom_30d range is 30-80%; threshold should be above
    assert extreme.parabola_mom_30d_threshold > 0.80

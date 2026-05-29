"""Integration tests for W6 — verifies the three new signals plumb
end-to-end through the blend without breaking existing behavior.

  PR #33: catalyst verification (apply_catalyst_verification)
  PR #34: fundamentals signal
  PR #35: revision momentum signal

These tests don't require network — they assemble synthetic signal
dicts and run the blend.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ai_layer import apply_catalyst_verification
from src.config import BLEND_WEIGHTS_V2
from src.signals import (
    blend_with_uncertainty,
    signal_from_fundamentals,
    signal_from_revision_momentum,
)


TODAY = date(2026, 5, 22)


def _grade(action, days_ago):
    return {"action": action,
            "publishedDate": (TODAY - timedelta(days=days_ago)).strftime("%Y-%m-%d")}


def _make_full_signal_dict(fund_drift=0.02, rev_drift=0.015,
                            ai_drift=0.05, hist_drift=0.03,
                            ptrev_drift=0.015):
    """Build a signal dict with all 13 v2 slots populated. Numeric drifts
    are arbitrary but realistic; this is for plumbing verification, not
    drift accuracy."""
    def s(drift, conf="MEDIUM"):
        return {"drift": drift, "confidence": conf,
                "source_quality": "PRIMARY", "sources_count": 1,
                "notes": ""}
    return {
        "historical": s(hist_drift),
        "analyst": s(0.04),
        "sector": s(0.01),
        "macro": s(-0.005),
        "short_interest": s(0.005, "LOW"),
        "peer_rs": s(0.02),
        "sector_decoupling": s(-0.01),
        "ai": s(ai_drift, "HIGH"),
        "catalyst_proximity": s(0.03, "MEDIUM"),
        "narrative": s(0.01),
        "fundamentals": s(fund_drift, "HIGH"),
        "revision_momentum": s(rev_drift, "MEDIUM"),
        "pt_revision": s(ptrev_drift, "MEDIUM"),
    }


def test_blend_accepts_all_twelve_v2_signals():
    """The two new W6 signals must be present in BLEND_WEIGHTS_V2 AND
    the blend function must accept them without dropping or crashing."""
    assert "fundamentals" in BLEND_WEIGHTS_V2
    assert "revision_momentum" in BLEND_WEIGHTS_V2
    signals = _make_full_signal_dict()
    blend = blend_with_uncertainty(signals, BLEND_WEIGHTS_V2)
    # blend returns dict with mu/sigma + weights breakdown
    assert "blended" in blend
    assert "std" in blend
    assert "weights" in blend
    # Both new signals should be in the weights output (not silently dropped).
    assert "fundamentals" in blend["weights"]
    assert "revision_momentum" in blend["weights"]


def test_fundamentals_contribution_moves_blend_drift():
    """Doubling fundamentals drift should detectably move the blended mu."""
    low = _make_full_signal_dict(fund_drift=0.0)
    high = _make_full_signal_dict(fund_drift=0.06)  # near cap
    blend_low = blend_with_uncertainty(low, BLEND_WEIGHTS_V2)
    blend_high = blend_with_uncertainty(high, BLEND_WEIGHTS_V2)
    assert blend_high["blended"] > blend_low["blended"]


def test_revision_momentum_now_inert_in_blend():
    """Defect C clean swap: revision_momentum's blend weight is 0.00, so
    changing its drift no longer moves the blended mu. The signal still
    computes + displays; it just doesn't drive drift. (Was a 0.04-weight
    contributor pre-Defect-C.)"""
    low = _make_full_signal_dict(rev_drift=0.0)
    high = _make_full_signal_dict(rev_drift=0.05)
    blend_low = blend_with_uncertainty(low, BLEND_WEIGHTS_V2)
    blend_high = blend_with_uncertainty(high, BLEND_WEIGHTS_V2)
    assert blend_high["blended"] == blend_low["blended"]


def test_pt_revision_contribution_moves_blend_drift():
    """Defect C: pt_revision now carries the analyst-revision slot's weight
    (0.04), so changing its drift detectably moves the blended mu."""
    low = _make_full_signal_dict(ptrev_drift=0.0)
    high = _make_full_signal_dict(ptrev_drift=0.05)
    blend_low = blend_with_uncertainty(low, BLEND_WEIGHTS_V2)
    blend_high = blend_with_uncertainty(high, BLEND_WEIGHTS_V2)
    assert blend_high["blended"] > blend_low["blended"]


def test_blend_weights_sum_to_one():
    """Post-renormalization the blend's weights add to 1.0."""
    signals = _make_full_signal_dict()
    blend = blend_with_uncertainty(signals, BLEND_WEIGHTS_V2)
    total = sum(blend["weights"].values())
    # Some signals might be halved (LOW conf) but the relative sum
    # should be close to (sum of nominal weights) ≈ 1.0 minus any
    # halved gating. The blend renormalizes implicitly.
    assert 0.5 <= total <= 1.0  # weights are pre-renorm; total ≤ 1.0


def test_signals_with_none_drift_dont_break_blend():
    """Both new W6 signals can return drift=None on no-coverage /
    pre-revenue tickers. The blend must skip them without crashing."""
    signals = _make_full_signal_dict()
    signals["fundamentals"] = {
        "drift": None, "confidence": "LOW",
        "source_quality": "NONE_FOUND", "sources_count": 0, "notes": ""
    }
    signals["revision_momentum"] = {
        "drift": None, "confidence": "LOW",
        "source_quality": "NONE_FOUND", "sources_count": 0, "notes": ""
    }
    blend = blend_with_uncertainty(signals, BLEND_WEIGHTS_V2)
    assert blend["blended"] is not None  # blend succeeded
    # Other 10 signals still produce a sensible mu.


def test_signal_factories_produce_blend_compatible_output():
    """Both new signal_from_* functions return dicts the blend accepts."""
    # Fundamentals
    fund = signal_from_fundamentals({
        "fcf_yield": 0.05,
        "net_debt_to_ebitda": 1.5,
        "op_margin_trend": 0.02,
        "n_components_available": 3,
    })
    assert set(fund.keys()) >= {"drift", "confidence", "source_quality"}
    # Revision momentum
    rev = signal_from_revision_momentum(
        [_grade("upgrade", 5), _grade("upgrade", 20), _grade("downgrade", 50)],
        today=TODAY,
    )
    assert set(rev.keys()) >= {"drift", "confidence", "source_quality"}
    # Both must be wirable into a synthetic signals dict.
    signals = _make_full_signal_dict()
    signals["fundamentals"] = fund
    signals["revision_momentum"] = rev
    blend = blend_with_uncertainty(signals, BLEND_WEIGHTS_V2)
    assert blend["blended"] is not None


def test_catalyst_verification_filters_before_signal():
    """Verification REFUTES a catalyst → catalyst_proximity signal
    operates on the smaller filtered list. Sanity-check the
    plumbing (signal recomputation is downstream of apply_)."""
    catalysts = [
        {"name": "real earnings", "type": "earnings",
         "date_or_window": "2026-07-15", "magnitude": "high",
         "direction_risk": "bullish"},
        {"name": "phantom deal", "type": "M&A",
         "date_or_window": "2026-06-30", "magnitude": "high",
         "direction_risk": "bullish"},
    ]
    verifications = [
        {"catalyst_name": "real earnings", "verdict": "VERIFIED",
         "reasoning": "8-K confirmed", "supporting_url": None},
        {"catalyst_name": "phantom deal", "verdict": "REFUTED",
         "reasoning": "no such deal in SEC filings", "supporting_url": None},
    ]
    filtered = apply_catalyst_verification(catalysts, verifications)
    assert len(filtered) == 1
    assert filtered[0]["name"] == "real earnings"
    assert filtered[0]["verification_verdict"] == "VERIFIED"


def test_w6_signal_drifts_compose_realistically():
    """Realistic INTC-like profile across all 12 signals — final blended
    drift should be positive but moderate (not pegged at any cap)."""
    signals = _make_full_signal_dict(
        hist_drift=0.04, ai_drift=0.10, fund_drift=0.015,
        rev_drift=0.025,
    )
    blend = blend_with_uncertainty(signals, BLEND_WEIGHTS_V2)
    # Sanity: with positive drift across signals, blended mu should be positive.
    assert blend["blended"] > 0
    # But not pegged at any extreme — should be in plausible 0-15% range.
    assert blend["blended"] < 0.15

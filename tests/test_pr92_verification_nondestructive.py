"""Tests for PR #92 — catalyst verification no longer self-sabotages.

ROOT CAUSE (confirmed on MU 2026-05-28 T3 run): the catalyst verification
step runs Haiku WITHOUT web search (PR #52 killed the search path), so it is
bounded by its training cutoff and structurally cannot confirm any recent /
in-horizon event. It returned UNVERIFIED on all 3 of MU's real, institutionally
-sourced catalysts (HBM4 sold out, record Q3 guidance, FY27 capex). The old
rule `UNVERIFIED → magnitude=low` then collapsed the catalyst_proximity drift
signal to +0.0% (at 13.9% weight) on every run, silently discarding Pass 1's
findings and burying the verdict.

FIX: UNVERIFIED is now a NO-OP. Only REFUTED (active contradiction) acts.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _catalyst(name, magnitude="high", direction="bullish"):
    return {
        "name": name, "type": "earnings", "date_or_window": "2026-06-24",
        "magnitude": magnitude, "direction_risk": direction,
        "sources": ["sec.gov", "reuters.com"],
    }


# =============================================================================
# 1. UNVERIFIED must NOT downgrade magnitude (the core regression)
# =============================================================================

def test_unverified_preserves_magnitude():
    """The MU failure mode: a real, high-magnitude, well-sourced catalyst
    marked UNVERIFIED by the blind verifier must KEEP its magnitude — not
    get downgraded to 'low' (which zeroed the drift signal)."""
    from src.ai_layer import apply_catalyst_verification
    catalysts = [_catalyst("HBM4 sold out + record Q3 guidance", magnitude="high")]
    verifications = [{"catalyst_name": "HBM4 sold out + record Q3 guidance",
                      "verdict": "UNVERIFIED", "reasoning": "can't confirm from training"}]
    out = apply_catalyst_verification(catalysts, verifications)
    assert len(out) == 1
    assert out[0]["magnitude"] == "high", (
        "UNVERIFIED must be a no-op — magnitude must survive. If this is "
        "'low', the verifier is again self-sabotaging Pass 1's findings."
    )


def test_three_unverified_keep_all_magnitudes():
    """Reproduces MU exactly: 3 catalysts, all UNVERIFIED, all keep magnitude."""
    from src.ai_layer import apply_catalyst_verification
    catalysts = [
        _catalyst("Q3 FY26 earnings", magnitude="high"),
        _catalyst("HBM4 ramp / pricing", magnitude="high", direction="bullish"),
        _catalyst("FY27 capex step-up", magnitude="med"),
    ]
    verifications = [
        {"catalyst_name": "Q3 FY26 earnings", "verdict": "UNVERIFIED"},
        {"catalyst_name": "HBM4 ramp / pricing", "verdict": "UNVERIFIED"},
        {"catalyst_name": "FY27 capex step-up", "verdict": "UNVERIFIED"},
    ]
    out = apply_catalyst_verification(catalysts, verifications)
    mags = [c["magnitude"] for c in out]
    assert mags == ["high", "high", "med"], (
        f"All magnitudes must survive UNVERIFIED; got {mags}"
    )


# =============================================================================
# 2. REFUTED still drops (protective value retained)
# =============================================================================

def test_refuted_still_drops():
    from src.ai_layer import apply_catalyst_verification
    catalysts = [_catalyst("Real catalyst"), _catalyst("Hallucinated catalyst")]
    verifications = [
        {"catalyst_name": "Real catalyst", "verdict": "VERIFIED"},
        {"catalyst_name": "Hallucinated catalyst", "verdict": "REFUTED"},
    ]
    out = apply_catalyst_verification(catalysts, verifications)
    names = [c["name"] for c in out]
    assert names == ["Real catalyst"], "REFUTED must still drop the catalyst"


# =============================================================================
# 3. VERIFIED keeps as-is
# =============================================================================

def test_verified_keeps_magnitude():
    from src.ai_layer import apply_catalyst_verification
    catalysts = [_catalyst("Confirmed catalyst", magnitude="high")]
    verifications = [{"catalyst_name": "Confirmed catalyst", "verdict": "VERIFIED"}]
    out = apply_catalyst_verification(catalysts, verifications)
    assert out[0]["magnitude"] == "high"


# =============================================================================
# 4. End-to-end: catalyst_proximity drift survives UNVERIFIED
# =============================================================================

def test_catalyst_proximity_drift_survives_unverified():
    """The real win: with magnitudes preserved, the catalyst_proximity signal
    produces non-zero drift for in-window bullish catalysts — instead of the
    +0.0% the MU run showed."""
    from src.ai_layer import apply_catalyst_verification
    from src.signals import signal_from_catalyst_proximity
    catalysts = [
        _catalyst("HBM4 ramp", magnitude="high", direction="bullish"),
        _catalyst("Record guidance", magnitude="high", direction="bullish"),
    ]
    verifications = [
        {"catalyst_name": "HBM4 ramp", "verdict": "UNVERIFIED"},
        {"catalyst_name": "Record guidance", "verdict": "UNVERIFIED"},
    ]
    kept = apply_catalyst_verification(catalysts, verifications)
    mu, conf, rat = signal_from_catalyst_proximity(kept, horizon_days=20)
    # Bullish in-window high-magnitude catalysts must move drift > 0.
    assert mu > 0.0, (
        f"catalyst_proximity drift should be positive after UNVERIFIED no-op; "
        f"got {mu} (this was +0.0% in the MU bug)"
    )


# =============================================================================
# 5. MU reclassified MID → HIGH (Defect G)
# =============================================================================

def test_mu_registry_is_high():
    """MU auto-detects HIGH (σ ~88%) every run; registry hint updated to match."""
    from src.registry import classify
    assert classify("MU") == "HIGH"

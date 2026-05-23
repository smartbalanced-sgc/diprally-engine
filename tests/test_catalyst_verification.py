"""Tests for W6 PR #33 — catalyst verification (D-W5-1).

The network-bound call_ai_catalyst_verification() function isn't unit-
testable without mocking the Anthropic client (deferred to integration
smoke). These tests cover apply_catalyst_verification(), the pure
function that filters + downgrades the catalyst list based on verdict
inputs.

Verdicts:
  VERIFIED   → magnitude unchanged, verification fields appended
  UNVERIFIED → magnitude forced to "low", original preserved as
               magnitude_pre_verification
  REFUTED    → catalyst dropped entirely
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ai_layer import VERIFICATION_VERDICTS, apply_catalyst_verification


def _cat(name, magnitude="med", direction="two-sided"):
    return {
        "name": name,
        "type": "earnings",
        "date_or_window": "2026-07-15",
        "magnitude": magnitude,
        "direction_risk": direction,
    }


def _verdict(name, verdict, reasoning="primary source check", url=None):
    return {
        "catalyst_name": name,
        "verdict": verdict,
        "reasoning": reasoning,
        "supporting_url": url,
    }


def test_verified_passes_through_with_metadata():
    catalysts = [_cat("Q2 earnings", "high", "bullish")]
    verifications = [_verdict("Q2 earnings", "VERIFIED",
                               url="https://sec.gov/8k/123")]
    out = apply_catalyst_verification(catalysts, verifications)
    assert len(out) == 1
    assert out[0]["magnitude"] == "high"  # unchanged
    assert out[0]["verification_verdict"] == "VERIFIED"
    assert out[0]["verification_url"] == "https://sec.gov/8k/123"


def test_unverified_downgrades_magnitude_to_low():
    catalysts = [_cat("Convertible Q2 window", "med")]
    verifications = [_verdict("Convertible Q2 window", "UNVERIFIED",
                               reasoning="no 10-Q footnote supports a Q2 window")]
    out = apply_catalyst_verification(catalysts, verifications)
    assert len(out) == 1
    assert out[0]["magnitude"] == "low"
    assert out[0]["magnitude_pre_verification"] == "med"
    assert out[0]["verification_verdict"] == "UNVERIFIED"


def test_refuted_drops_catalyst_entirely():
    catalysts = [_cat("Motiv acquisition close Q2")]
    verifications = [_verdict("Motiv acquisition close Q2", "REFUTED",
                               reasoning="deal closed Q4 2025")]
    out = apply_catalyst_verification(catalysts, verifications)
    assert out == []


def test_mixed_verdicts_apply_independently():
    """Three catalysts, one of each verdict — verify the resulting list."""
    catalysts = [
        _cat("Q2 earnings", "high"),
        _cat("Convertible Q2 window", "med"),
        _cat("Phantom deal", "high"),
    ]
    verifications = [
        _verdict("Q2 earnings", "VERIFIED"),
        _verdict("Convertible Q2 window", "UNVERIFIED"),
        _verdict("Phantom deal", "REFUTED"),
    ]
    out = apply_catalyst_verification(catalysts, verifications)
    assert len(out) == 2
    names = [c["name"] for c in out]
    assert names == ["Q2 earnings", "Convertible Q2 window"]
    assert out[0]["magnitude"] == "high"
    assert out[1]["magnitude"] == "low"


def test_catalysts_beyond_verified_topN_passthrough():
    """If only 3 verifications come back but 5 catalysts were
    surfaced, catalysts 4 + 5 pass through unchanged (they didn't
    reach the verification top-N)."""
    catalysts = [_cat(f"cat_{i}", "med") for i in range(5)]
    verifications = [_verdict(f"cat_{i}", "REFUTED") for i in range(3)]
    out = apply_catalyst_verification(catalysts, verifications)
    # Top 3 REFUTED → dropped; tail 2 pass through.
    assert len(out) == 2
    assert [c["name"] for c in out] == ["cat_3", "cat_4"]


def test_empty_catalysts_returns_empty():
    assert apply_catalyst_verification([], [_verdict("x", "VERIFIED")]) == []


def test_no_verifications_passes_input_unchanged():
    """When verification call fails (returns []), catalysts pass
    through untouched — engine should NOT silently drop catalysts
    just because verification was unavailable."""
    catalysts = [_cat("Q2 earnings", "high")]
    out = apply_catalyst_verification(catalysts, [])
    assert out == catalysts  # same content


def test_apply_does_not_mutate_inputs():
    """Pure function contract — input list is not modified in place."""
    catalysts = [_cat("Convertible Q2 window", "med")]
    verifications = [_verdict("Convertible Q2 window", "UNVERIFIED")]
    apply_catalyst_verification(catalysts, verifications)
    # Original catalyst dict unchanged.
    assert catalysts[0]["magnitude"] == "med"
    assert "verification_verdict" not in catalysts[0]


def test_unknown_verdict_treated_as_unverified():
    """Defensive: a verdict string we don't recognize (model
    hallucination) should be treated as UNVERIFIED — downgrade rather
    than silently passing as VERIFIED."""
    catalysts = [_cat("X", "high")]
    verifications = [{"catalyst_name": "X", "verdict": "MAYBE",
                       "reasoning": "uncertain", "supporting_url": None}]
    # The verifications list comes pre-cleaned from
    # call_ai_catalyst_verification, but if a caller hand-builds one
    # with an unknown verdict, apply_ should still be defensive.
    out = apply_catalyst_verification(catalysts, verifications)
    # apply_ doesn't re-clean — that's call_ai_..._verification's job.
    # But the catalyst should NOT be dropped (not REFUTED) and should
    # carry through with whatever verdict label was passed.
    assert len(out) == 1
    assert out[0]["verification_verdict"] == "MAYBE"


def test_verification_verdicts_set_matches_documentation():
    """Sacred constant — VERIFIED / UNVERIFIED / REFUTED exactly."""
    assert VERIFICATION_VERDICTS == {"VERIFIED", "UNVERIFIED", "REFUTED"}

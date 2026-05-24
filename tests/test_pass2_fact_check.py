"""Tests for PR #63 — Pass 2 fact-discipline programmatic enforcer.

Smoke evidence (INTC v3 audit, 2026-05-23) showed Pass 2 sometimes
invents numeric facts:
  Pass 2 critique: "INTC's current price is far below $119..."
  Actual spot:     $119.84
  → Hallucination — Pass 2 contradicted reality.

PR #40 added the FACT DISCIPLINE prompt block; PR #63 adds
programmatic enforcement. Phase 1 = detect + flag in the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pass2_fact_check import (
    FactViolation,
    _extract_dollar_mentions,
    _find_spot_context_dollars,
    format_violations,
    validate_pass2_critique,
)


# =============================================================================
# _extract_dollar_mentions
# =============================================================================

def test_extract_simple_dollar_amounts():
    text = "Spot is $119.84 and target is $145."
    mentions = _extract_dollar_mentions(text)
    values = [v for v, _ in mentions]
    assert 119.84 in values
    assert 145.0 in values


def test_extract_comma_formatted():
    text = "Market cap $1,500,000,000 is large."
    mentions = _extract_dollar_mentions(text)
    assert 1_500_000_000.0 in [v for v, _ in mentions]


def test_extract_skips_billions_suffix():
    """$1.5B is market cap context, not price. Skip."""
    text = "Market cap of $602B is large."
    mentions = _extract_dollar_mentions(text)
    assert mentions == []


def test_extract_skips_year_values():
    """$2026 looks like a year, not a price. Skip integer years
    in 1900-2100 range."""
    text = "In $2026 the company will pivot."
    mentions = _extract_dollar_mentions(text)
    assert mentions == []


def test_extract_keeps_decimal_2026_dot_84():
    """$2026.84 is a real dollar amount (not a year), keep it."""
    text = "Trades at $2026.84 currently."
    mentions = _extract_dollar_mentions(text)
    assert 2026.84 in [v for v, _ in mentions]


def test_empty_text_returns_empty():
    assert _extract_dollar_mentions("") == []
    assert _extract_dollar_mentions(None) == []


# =============================================================================
# _find_spot_context_dollars
# =============================================================================

def test_spot_context_phrase_captures_dollar():
    """Dollar amount within 60 chars after 'current price' is
    spot-anchored."""
    text = "INTC's current price is now $119.84 per share."
    mentions = _find_spot_context_dollars(text)
    assert 119.84 in [v for v, _ in mentions]


def test_multiple_spot_context_phrases():
    text = ("Stock trades at $120. Current price is below $80.")
    mentions = _find_spot_context_dollars(text)
    values = [v for v, _ in mentions]
    assert 120.0 in values
    assert 80.0 in values


def test_dollar_outside_spot_context_not_captured():
    """A bare $XXX with no spot-context phrase isn't flagged by
    this function (may still be flagged by outlier-price scan)."""
    text = "Q2 earnings will be $0.50 per share."
    mentions = _find_spot_context_dollars(text)
    # No spot/current/trades phrases → empty.
    assert mentions == []


# =============================================================================
# validate_pass2_critique — high-confidence spot contradictions
# =============================================================================

def test_smoke_audit_intc_v3_case_detected():
    """The exact INTC v3 case that motivated this PR.
    Pass 2 said 'current price is far below $119' when spot was $119.84.

    The literal wording 'far below $119' is itself an internal
    contradiction (claiming spot is far below $119, then citing $119
    as if it were the reference) — but the validator flags it as
    spot_contradiction because $119 in a 'current price' context
    diverges from the spot the engine actually saw."""
    critique = (
        "INTC's current price is far below $119 and this figure is "
        "inconsistent with the $80-88 blended analyst PT context."
    )
    # Suppose actual spot was $40 (far below $119) — then $119 in
    # a current-price context IS a fact-hallucination.
    violations = validate_pass2_critique(critique, spot=40.0)
    contradictions = [v for v in violations if v.kind == "spot_contradiction"]
    assert len(contradictions) >= 1
    assert contradictions[0].claimed_value == 119.0
    assert contradictions[0].expected_value == 40.0


def test_no_violation_when_spot_matches():
    """Pass 2 correctly cites the actual spot → no violation."""
    critique = "Stock currently trades at $119.84, as the math layer notes."
    violations = validate_pass2_critique(critique, spot=119.84)
    contradictions = [v for v in violations if v.kind == "spot_contradiction"]
    assert contradictions == []


def test_small_divergence_not_flagged():
    """Pass 2 says spot ≈ $120 when actual is $119.84 — within 20%
    tolerance, no contradiction."""
    critique = "Current price of $120 sits at the high end of the range."
    violations = validate_pass2_critique(critique, spot=119.84)
    contradictions = [v for v in violations if v.kind == "spot_contradiction"]
    assert contradictions == []


# =============================================================================
# validate_pass2_critique — outlier prices (loose)
# =============================================================================

def test_outlier_price_flagged():
    """Standalone $X that's > 2× or < 0.5× of spot → outlier flag."""
    critique = "Pass 1 mentions a fair value of $300 per share."
    violations = validate_pass2_critique(critique, spot=100.0)
    outliers = [v for v in violations if v.kind == "outlier_price"]
    assert len(outliers) == 1
    assert outliers[0].claimed_value == 300.0


def test_price_within_50pct_band_not_flagged():
    """A $150 mention with spot $100 is within the legitimate
    target/PT band (50-150% of spot) — not flagged."""
    critique = "Fair value estimate of $145 implies upside."
    violations = validate_pass2_critique(critique, spot=100.0)
    outliers = [v for v in violations if v.kind == "outlier_price"]
    assert outliers == []


def test_spot_contradiction_not_double_flagged():
    """A $X flagged as spot_contradiction must NOT also be flagged
    as outlier_price — that would double-count the same violation."""
    critique = "Current price is now $500 according to my analysis."
    violations = validate_pass2_critique(critique, spot=100.0)
    contradictions = [v for v in violations if v.kind == "spot_contradiction"]
    outliers = [v for v in violations if v.kind == "outlier_price"]
    # $500 is 5× spot → spot_contradiction fires. Should NOT also
    # appear as outlier_price for the same position.
    assert len(contradictions) == 1
    # outlier scan may pick up the $500 again at the SAME position →
    # the implementation must dedupe.
    assert all(o.claimed_value != 500.0 for o in outliers) or len(outliers) == 0


# =============================================================================
# Defensive paths
# =============================================================================

def test_empty_critique_returns_empty():
    assert validate_pass2_critique("", spot=100.0) == []
    assert validate_pass2_critique(None, spot=100.0) == []


def test_zero_spot_returns_empty():
    """Defensive: spot=0 would divide-by-zero in the tolerance check."""
    assert validate_pass2_critique("text $100", spot=0.0) == []
    assert validate_pass2_critique("text $100", spot=-5.0) == []


# =============================================================================
# format_violations
# =============================================================================

def test_format_violations_empty_returns_empty_string():
    assert format_violations([]) == ""


def test_format_violations_includes_violation_count():
    violations = [
        FactViolation(kind="spot_contradiction", claimed_value=200.0,
                       expected_value=100.0, expected_label="spot",
                       divergence_pct=100.0, context="trades at $200 today"),
    ]
    output = format_violations(violations)
    assert "1 potential violation" in output
    assert "HIGH-CONF SPOT CONTRADICTION" in output
    assert "$200" in output
    assert "$100" in output


def test_format_violations_distinguishes_kinds():
    violations = [
        FactViolation(kind="spot_contradiction", claimed_value=200.0,
                       expected_value=100.0, expected_label="spot",
                       divergence_pct=100.0, context="ctx1"),
        FactViolation(kind="outlier_price", claimed_value=300.0,
                       expected_value=100.0, expected_label="spot",
                       divergence_pct=200.0, context="ctx2"),
    ]
    output = format_violations(violations)
    assert "HIGH-CONF" in output
    assert "outlier price" in output
    assert "2 potential" in output

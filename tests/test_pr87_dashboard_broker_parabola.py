"""Tests for PR #87 — parabola override + broker T2 relax + dashboard redesign.

The previous live cycle (2026-05-27 07:44) returned ZERO BUYs because:
  1. PR #86 set horizon 20d which is genuinely tight for HIGH-σ names
  2. With T1-only AI, math layer can't find pre-AI positive EV setups
  3. Broker required qualifies_for_t2_plus for T2 → no T2 ran → no
     catalyst overlay → math stayed conservative → REFUSED-EV cascade

PR #87 fixes:
  A. Broker T2 eligibility: ambiguity-driven, not pre-AI-EV gated
  B. Parabola override: structured prompt + code-validation lets
     fundamentally-supported momentum names through sacred #18
  C. Dashboard detail row showing dual-EV breakdown per ticker
  D. Tooltip horizon read dynamically (was hardcoded "60-day")
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# Parabola override validator — hard rules
# =============================================================================

GOOD_BULL_ITEM = {
    "claim": "TSMC announced 40nm capacity expansion in Q1 earnings, increasing MU's contract pricing for Q3 2026",
    "source": "TSMC Q1 2026 earnings call transcript (SeekingAlpha id 12345)",
    "specificity_score": 5,
    "in_horizon_relevance": 4,
}

ANOTHER_GOOD_BULL = {
    "claim": "Micron raised FY guidance to $42-44B on 2026-05-15, up from $38-40B",
    "source": "Micron 8-K filing 2026-05-15",
    "specificity_score": 5,
    "in_horizon_relevance": 5,
}

VAGUE_BULL = {
    "claim": "AI demand is strong",
    "source": "general market sentiment",
    "specificity_score": 2,
    "in_horizon_relevance": 3,
}


def test_validator_rejects_default_vote():
    """vote = REFUSE-BLOWOFF (or anything else) → override invalid."""
    from src.ai_layer import validate_parabola_override
    assert validate_parabola_override({"vote": "REFUSE-BLOWOFF"}) is False
    assert validate_parabola_override({"vote": "MAYBE"}) is False
    assert validate_parabola_override({}) is False
    assert validate_parabola_override(None) is False


def test_validator_requires_two_strong_bull_items():
    """One strong bull item is not enough — need 2+."""
    from src.ai_layer import validate_parabola_override
    response = {
        "vote": "OVERRIDE",
        "bull_evidence": [GOOD_BULL_ITEM],   # only 1
        "bear_evidence": [],
        "valuation_check": {"concern_flag": False},
    }
    assert validate_parabola_override(response) is False


def test_validator_accepts_two_strong_bull_items_no_concerns():
    """Two strong items, no valuation/blowoff concerns → override valid."""
    from src.ai_layer import validate_parabola_override
    response = {
        "vote": "OVERRIDE",
        "bull_evidence": [GOOD_BULL_ITEM, ANOTHER_GOOD_BULL],
        "bear_evidence": [],
        "valuation_check": {"concern_flag": False},
        "blowoff_indicators": {},
    }
    assert validate_parabola_override(response) is True


def test_validator_rejects_vague_bull_items():
    """Bull items with low specificity_score don't count toward the 2-item bar."""
    from src.ai_layer import validate_parabola_override
    response = {
        "vote": "OVERRIDE",
        "bull_evidence": [VAGUE_BULL, VAGUE_BULL, VAGUE_BULL],   # 3 weak items
        "bear_evidence": [],
        "valuation_check": {"concern_flag": False},
    }
    assert validate_parabola_override(response) is False


def test_validator_requires_valuation_response_when_concern_flag():
    """concern_flag=True → at least 1 bull item must explicitly address valuation."""
    from src.ai_layer import validate_parabola_override
    response = {
        "vote": "OVERRIDE",
        "bull_evidence": [GOOD_BULL_ITEM, ANOTHER_GOOD_BULL],  # neither addresses valuation
        "bear_evidence": [],
        "valuation_check": {"concern_flag": True},
        "blowoff_indicators": {},
    }
    assert validate_parabola_override(response) is False


def test_validator_accepts_valuation_response_when_concern_flag():
    """concern_flag=True is OK if a bull item directly addresses valuation."""
    from src.ai_layer import validate_parabola_override
    response = {
        "vote": "OVERRIDE",
        "bull_evidence": [
            GOOD_BULL_ITEM,
            {
                "claim": "fwd PE of 18× is in-line with 5y avg of 17.5× given growth re-acceleration",
                "source": "Bloomberg Terminal — peer comps + Bloomberg consensus",
                "specificity_score": 5,
                "in_horizon_relevance": 5,
            },
        ],
        "bear_evidence": [],
        "valuation_check": {"concern_flag": True},
        "blowoff_indicators": {},
    }
    assert validate_parabola_override(response) is True


def test_validator_rejects_strong_bear_evidence():
    """If bear_evidence has 2+ strong items, the override case is weak."""
    from src.ai_layer import validate_parabola_override
    strong_bear = {
        "claim": "Insider cluster sale of 500K shares 2026-05-20",
        "source": "Form 4 SEC filing",
        "specificity_score": 5,
        "in_horizon_relevance": 5,
    }
    response = {
        "vote": "OVERRIDE",
        "bull_evidence": [GOOD_BULL_ITEM, ANOTHER_GOOD_BULL],
        "bear_evidence": [strong_bear, strong_bear],   # 2 strong bears
        "valuation_check": {"concern_flag": False},
    }
    assert validate_parabola_override(response) is False


def test_validator_rejects_when_source_missing():
    """Bull items must have a source string. Empty or missing → not counted."""
    from src.ai_layer import validate_parabola_override
    no_source = {**GOOD_BULL_ITEM, "source": ""}
    response = {
        "vote": "OVERRIDE",
        "bull_evidence": [no_source, no_source],   # both lack source
        "bear_evidence": [],
        "valuation_check": {"concern_flag": False},
    }
    assert validate_parabola_override(response) is False


# =============================================================================
# AIPassOutput carries parabola override fields
# =============================================================================

def test_ai_pass_output_carries_override_fields():
    """parabola_override_raw and parabola_override_valid on the dataclass."""
    from src.engine import AIPassOutput
    a = AIPassOutput(
        pass_number=2, drift_estimate=0.10, drift_range=(-0.10, 0.30),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[], bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.04, raw_sources_cited=4,
    )
    # Defaults — no override unless explicitly populated by parse_ai_pass2
    assert a.parabola_override_raw is None
    assert a.parabola_override_valid is False


# =============================================================================
# Pass 2 parser populates override fields
# =============================================================================

def test_parse_pass2_captures_valid_override():
    """When AI Pass 2 returns a valid OVERRIDE response, parser sets
    parabola_override_valid=True and stores the raw dict."""
    from src.ai_layer import parse_ai_pass2
    from src.engine import AIPassOutput
    pass1 = AIPassOutput(
        pass_number=1, drift_estimate=0.10, drift_range=(-0.10, 0.30),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[], bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.05, raw_sources_cited=3,
    )
    raw = {
        "revised_drift_estimate": 0.20,
        "primary_critique": "Pass 1 under-weighted catalyst",
        "parabola_override": {
            "vote": "OVERRIDE",
            "bull_evidence": [GOOD_BULL_ITEM, ANOTHER_GOOD_BULL],
            "bear_evidence": [],
            "valuation_check": {"concern_flag": False},
            "blowoff_indicators": {},
        },
    }
    pass2 = parse_ai_pass2(raw, pass1, cost=0.04)
    assert pass2.parabola_override_valid is True
    assert pass2.parabola_override_raw is not None
    assert pass2.parabola_override_raw["vote"] == "OVERRIDE"


def test_parse_pass2_rejects_invalid_override():
    """AI votes OVERRIDE but evidence fails hard-rules → valid=False
    even though raw dict is preserved for audit."""
    from src.ai_layer import parse_ai_pass2
    from src.engine import AIPassOutput
    pass1 = AIPassOutput(
        pass_number=1, drift_estimate=0.10, drift_range=(-0.10, 0.30),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[], bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.05, raw_sources_cited=3,
    )
    raw = {
        "revised_drift_estimate": 0.20,
        "primary_critique": "...",
        "parabola_override": {
            "vote": "OVERRIDE",
            "bull_evidence": [VAGUE_BULL],   # weak → rejected
            "bear_evidence": [],
            "valuation_check": {"concern_flag": False},
        },
    }
    pass2 = parse_ai_pass2(raw, pass1, cost=0.04)
    assert pass2.parabola_override_valid is False
    # Raw still captured for audit transparency
    assert pass2.parabola_override_raw is not None


# =============================================================================
# Dashboard detail row renders for tickers with dual-EV data
# =============================================================================

def test_dual_ev_detail_row_renders_for_buy():
    """Each ticker row in the dashboard gets a follow-on detail row
    showing both entry strategies' EV breakdown."""
    from src import orchestrator as orch
    decisions = [
        orch.TickerDecision(
            ticker="AMAT", sigma_class="MID", tier="T2",
            ambiguity=0.10, qualifies_for_t2_plus=True,
            spot=454.89, dip_target=444.0, rally_target=473.0,
            p_round_trip=0.40, ev_bps_of_dip=120.0,
            verdict="BUY", status_note="setup",
            verdict_subtype="DIRECT",
            ev_direct_bps=120.0, ev_wait_bps=80.0,
            p_dip_filled=0.86, p_rally_hit=0.67,
            expected_rally_date="Jun 22, 2026",
            expected_dip_date="May 28, 2026",
        )
    ]
    html_out = orch._render_dashboard_html(decisions, None)
    # Detail row markers
    assert "dual-ev-detail" in html_out
    assert "DIRECT entry" in html_out
    assert "WAIT-FOR-DIP" in html_out
    # Both EVs surface
    assert "+120 bps" in html_out
    assert "+80 bps" in html_out or "+80.0 bps" in html_out
    # Calendar dates surface
    assert "Jun 22, 2026" in html_out
    assert "May 28, 2026" in html_out
    # PR #88: ★ winner marker removed (was misleading — always WAIT-wins
    # in current data was foolish per operator feedback).
    assert "★ winner" not in html_out


def test_dual_ev_detail_row_skipped_when_no_data():
    """Tickers with no dual-EV data (failed Phase 1 / pre-PR-#86 rows)
    don't get a detail row — keeps the dashboard clean."""
    from src import orchestrator as orch
    decisions = [
        orch.TickerDecision(
            ticker="FAILED_TICKER", sigma_class="?", tier="?",
            ambiguity=None, qualifies_for_t2_plus=None,
            spot=None, dip_target=None, rally_target=None,
            p_round_trip=None, ev_bps_of_dip=None,
            verdict="FAIL", status_note="phase 1 fail",
            ev_direct_bps=None, ev_wait_bps=None,
        )
    ]
    html_out = orch._render_dashboard_html(decisions, None)
    # No detail ROW renders (CSS class definition can appear in <style>,
    # but no actual <tr class="dual-ev-detail"> in the markup).
    assert '<tr class="dual-ev-detail"' not in html_out


# =============================================================================
# Tooltip horizon reads dynamically (was hardcoded 60-day)
# =============================================================================

def test_prt_tooltip_uses_dynamic_horizon():
    """The P(round-trip) tooltip should read the current horizon from
    config, not hardcode '60-day paths'."""
    from src.orchestrator import _prt_tooltip_text
    tt = _prt_tooltip_text(0.55)
    # The DEFAULT_HORIZON_DAYS in config is 20 (per PR #86).
    assert "20-trading-day" in tt
    # And the legacy hardcoded "60-day" must NOT appear.
    assert "60-day" not in tt

"""Tests for the AI thesis surfacing layer (2026-05-31).

Two coupled changes locked here:
  1. Catalyst source-date freshness: AI provides source_date, parser
     computes age + freshness deterministically. AI cannot game the
     staleness signal.
  2. Pass-2 verdict_alignment_signal: 5-level qualitative read on
     whether math's verdict is well-supported. INFORMATIONAL — does
     not override math.

Plus reporter rendering of the AI thesis block (catalysts with
freshness, alignment signal, drift refinement) so the qualitative
work AI was already producing becomes operator-visible.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# Source-date freshness — deterministic, computed in parser not by AI
# =============================================================================

def test_parse_source_date_handles_ok_iso():
    from src.ai_layer import _parse_source_date
    assert _parse_source_date("2026-05-29") == date(2026, 5, 29)


def test_parse_source_date_returns_none_on_bad_input():
    from src.ai_layer import _parse_source_date
    for bad in (None, "", "not a date", "2026/05/29", 123, {}):
        assert _parse_source_date(bad) is None


def test_freshness_label_fresh_aging_stale():
    from src.ai_layer import _freshness_label
    today = date(2026, 5, 31)
    assert _freshness_label(date(2026, 5, 31), today) == ("FRESH", 0)
    assert _freshness_label(date(2026, 5, 24), today) == ("FRESH", 7)
    assert _freshness_label(date(2026, 5, 23), today) == ("AGING", 8)
    assert _freshness_label(date(2026, 5, 1),  today) == ("AGING", 30)
    assert _freshness_label(date(2026, 4, 30), today) == ("STALE", 31)
    assert _freshness_label(date(2026, 6, 15), today) == ("FUTURE", -15)


def test_freshness_label_unknown_when_missing():
    from src.ai_layer import _freshness_label
    today = date(2026, 5, 31)
    assert _freshness_label(None, today) == ("UNKNOWN", None)


def test_enrich_catalyst_with_freshness_adds_fields_in_place():
    from src.ai_layer import _enrich_catalyst_with_freshness
    today = date(2026, 5, 31)
    cat = {
        "name": "HBM tightening", "type": "industry",
        "source_date": "2026-04-24",
    }
    out = _enrich_catalyst_with_freshness(cat, today)
    assert out["source_age_days"] == 37
    assert out["freshness"] == "STALE"


def test_parse_ai_pass1_enriches_catalysts_with_freshness():
    """End-to-end: Pass 1 JSON with source_date → parsed AIPassOutput
    has catalysts carrying freshness + age. Deterministic on today."""
    from src.ai_layer import parse_ai_pass1
    today = date(2026, 5, 31)
    raw = {
        "drift_estimate_annualized": 0.20,
        "drift_range_low_high": [-0.10, 0.50],
        "confidence": "MEDIUM",
        "vol_regime": "MEDIUM",
        "narrative_score": "strong",
        "catalysts": [
            {"name": "fresh news", "source_date": "2026-05-30"},
            {"name": "aging news", "source_date": "2026-05-10"},
            {"name": "stale news", "source_date": "2026-04-01"},
            {"name": "no date provided"},  # source_date absent
        ],
        "bull_factors": [], "bear_factors": [], "key_risks": [],
    }
    result = parse_ai_pass1(raw, sources_count=2, cost=0.05, today=today)
    fresh = [c for c in result.catalysts if c["name"] == "fresh news"][0]
    aging = [c for c in result.catalysts if c["name"] == "aging news"][0]
    stale = [c for c in result.catalysts if c["name"] == "stale news"][0]
    nodate = [c for c in result.catalysts if c["name"] == "no date provided"][0]
    assert fresh["freshness"] == "FRESH"
    assert aging["freshness"] == "AGING"
    assert stale["freshness"] == "STALE"
    assert nodate["freshness"] == "UNKNOWN"
    assert nodate["source_age_days"] is None


# =============================================================================
# Pass-2 verdict alignment signal — parser hardening
# =============================================================================

def test_parse_ai_pass2_extracts_valid_alignment_signals():
    from src.ai_layer import parse_ai_pass1, parse_ai_pass2
    today = date(2026, 5, 31)
    pass1_raw = {
        "drift_estimate_annualized": 0.10,
        "drift_range_low_high": [0.0, 0.20],
        "confidence": "MEDIUM", "vol_regime": "MEDIUM",
        "narrative_score": "neutral",
        "catalysts": [], "bull_factors": [], "bear_factors": [], "key_risks": [],
    }
    pass1 = parse_ai_pass1(pass1_raw, sources_count=1, cost=0.0, today=today)
    for level in ("STRONG_SUPPORT", "SUPPORT", "NEUTRAL",
                  "CAUTION", "STRONG_CAUTION"):
        pass2_raw = {
            "revised_drift_estimate": 0.10,
            "verdict_alignment_signal": level,
            "verdict_alignment_reasoning": "test reason for " + level,
        }
        pass2 = parse_ai_pass2(pass2_raw, pass1, cost=0.0, today=today)
        assert pass2.verdict_alignment_signal == level
        assert pass2.verdict_alignment_reasoning.startswith("test reason for ")


def test_parse_ai_pass2_defaults_bad_alignment_to_neutral():
    """AI may hallucinate signal values outside the 5-level scale. The
    parser must coerce to NEUTRAL rather than crash or pass through
    junk into downstream display."""
    from src.ai_layer import parse_ai_pass1, parse_ai_pass2
    today = date(2026, 5, 31)
    pass1_raw = {
        "drift_estimate_annualized": 0.10,
        "drift_range_low_high": [0.0, 0.20],
        "confidence": "MEDIUM", "vol_regime": "MEDIUM",
        "narrative_score": "neutral",
        "catalysts": [], "bull_factors": [], "bear_factors": [], "key_risks": [],
    }
    pass1 = parse_ai_pass1(pass1_raw, sources_count=1, cost=0.0, today=today)
    for bad in ("YOLO", "very_bullish", "", None, 42):
        pass2_raw = {
            "revised_drift_estimate": 0.10,
            "verdict_alignment_signal": bad,
        }
        pass2 = parse_ai_pass2(pass2_raw, pass1, cost=0.0, today=today)
        assert pass2.verdict_alignment_signal == "NEUTRAL"


def test_parse_ai_pass2_backward_compat_signal_absent():
    """Old cached AI runs (pre-2026-05-31) won't have the new field.
    Parser must default to NEUTRAL silently — same-day cache replay
    (sacred #11/#12) requires backward compat on schema additions."""
    from src.ai_layer import parse_ai_pass1, parse_ai_pass2
    today = date(2026, 5, 31)
    pass1_raw = {
        "drift_estimate_annualized": 0.10,
        "drift_range_low_high": [0.0, 0.20],
        "confidence": "MEDIUM", "vol_regime": "MEDIUM",
        "narrative_score": "neutral",
        "catalysts": [], "bull_factors": [], "bear_factors": [], "key_risks": [],
    }
    pass1 = parse_ai_pass1(pass1_raw, sources_count=1, cost=0.0, today=today)
    pass2_raw = {"revised_drift_estimate": 0.15}  # No alignment fields
    pass2 = parse_ai_pass2(pass2_raw, pass1, cost=0.0, today=today)
    assert pass2.verdict_alignment_signal == "NEUTRAL"
    assert pass2.verdict_alignment_reasoning == ""


# =============================================================================
# Reporter — AI thesis block rendering
# =============================================================================

def test_ai_thesis_block_renders_catalysts_with_freshness():
    from src.engine import AIPassOutput
    from src.reporter import _ai_thesis_block
    pass1 = AIPassOutput(
        pass_number=1, drift_estimate=0.18, drift_range=(0.0, 0.4),
        confidence="MEDIUM", vol_regime="MEDIUM",
        narrative_score="strong",
        catalysts=[
            {"name": "fresh PT hike", "direction_risk": "bullish",
             "magnitude": "med", "date_or_window": "2026-05-30",
             "source_date": "2026-05-30", "source_url": "https://example.com/a",
             "source_age_days": 1, "freshness": "FRESH"},
        ],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.05, raw_sources_cited=2,
    )
    pass2 = AIPassOutput(
        pass_number=2, drift_estimate=0.22, drift_range=(0.10, 0.34),
        confidence="MEDIUM", vol_regime="MEDIUM",
        narrative_score="strong",
        catalysts=pass1.catalysts,
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=0.04, cost_usd=0.08, raw_sources_cited=0,
        agreement_with_pass1="agree",
        verdict_alignment_signal="STRONG_SUPPORT",
        verdict_alignment_reasoning="Catalysts validated, no near-horizon risk.",
    )
    out = _ai_thesis_block(pass1, pass2)
    text = "\n".join(out)
    assert "AI THESIS" in text
    assert "STRONG SUPPORT" in text
    assert "Catalysts validated" in text
    assert "fresh PT hike" in text
    assert "FRESH" in text
    assert "Pass 1 +18.0%" in text  # drift refinement display
    assert "Pass 2 +22.0%" in text


def test_ai_thesis_block_handles_pass2_absent():
    """Pass 1 ran but Pass 2 didn't (DEGRADED tier or cache partial).
    Block still renders Pass 1 catalysts; alignment signal section
    explicitly notes Pass 2 absence."""
    from src.engine import AIPassOutput
    from src.reporter import _ai_thesis_block
    pass1 = AIPassOutput(
        pass_number=1, drift_estimate=0.10, drift_range=(0.0, 0.2),
        confidence="MEDIUM", vol_regime="MEDIUM",
        narrative_score="neutral",
        catalysts=[
            {"name": "stale rumor", "direction_risk": "bearish",
             "magnitude": "low",
             "source_date": "2026-03-01",
             "source_age_days": 91, "freshness": "STALE"},
        ],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.02, raw_sources_cited=1,
    )
    out = _ai_thesis_block(pass1, None)
    text = "\n".join(out)
    assert "Pass 2 absent" in text
    assert "stale rumor" in text
    assert "STALE" in text


def test_ai_thesis_block_empty_when_no_ai():
    """T0 math-only runs (no AI tier) — _ai_thesis_block returns []
    so the report stays silent on AI surfaces. Don't render an empty
    header for runs where AI didn't participate."""
    from src.reporter import _ai_thesis_block
    assert _ai_thesis_block(None, None) == []


def test_headline_card_appends_ai_signal_to_buy_line():
    """The 5-level signal renders as a one-segment suffix on the
    headline BUY line. NEUTRAL with no reasoning renders nothing
    (don't add noise to the headline when there's no signal)."""
    from src.engine import AIPassOutput, JointConditionalResult
    from src.reporter import _headline_card

    class _Snap:
        mom_30d = 0.10

    best = JointConditionalResult(
        dip_price=100.0, rally_price=110.0,
        p_dip_touched=0.7, p_rally_given_dip=0.6, p_round_trip=0.35,
        p_bag_hold=0.1, p_no_trade_rally_first=0.1, p_neither=0.1,
        expected_days_to_dip=5.0, expected_days_dip_to_rally=8.0,
        expected_gain_per_share=5.0, expected_bag_hold_loss=2.0,
        net_ev_per_share=0.05, ev_pct_of_dip=0.005,
        verdict_subtype="WAIT-FOR-DIP",
        p_profitable=0.35,
    )
    pass2_caution = AIPassOutput(
        pass_number=2, drift_estimate=0.10, drift_range=(0.0, 0.2),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="neutral",
        catalysts=[], bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=0.0, cost_usd=0.05, raw_sources_cited=0,
        verdict_alignment_signal="CAUTION",
        verdict_alignment_reasoning="Earnings inside horizon.",
    )
    line = _headline_card(
        ticker="TST", spot=105.0, horizon_days=20, best=best,
        ev_pct_of_dip=0.005, met_threshold_strict=True,
        method_check={}, parabola_filter_refused=False,
        trend_filter_refused=False, ev_hurdle_refused=False,
        snapshot=_Snap(), pass2=pass2_caution,
    )
    assert "BUY" in line
    assert "CAUTION" in line
    assert "Earnings inside horizon" in line

    # NEUTRAL + no reasoning → no suffix on headline
    pass2_neutral = AIPassOutput(
        pass_number=2, drift_estimate=0.10, drift_range=(0.0, 0.2),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="neutral",
        catalysts=[], bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=0.0, cost_usd=0.05, raw_sources_cited=0,
    )
    line_neutral = _headline_card(
        ticker="TST", spot=105.0, horizon_days=20, best=best,
        ev_pct_of_dip=0.005, met_threshold_strict=True,
        method_check={}, parabola_filter_refused=False,
        trend_filter_refused=False, ev_hurdle_refused=False,
        snapshot=_Snap(), pass2=pass2_neutral,
    )
    assert "AI:" not in line_neutral


def test_headline_card_no_pass2_no_suffix():
    """Pass 2 absent → headline renders without AI segment, no crash."""
    from src.engine import JointConditionalResult
    from src.reporter import _headline_card

    class _Snap:
        mom_30d = 0.10

    best = JointConditionalResult(
        dip_price=100.0, rally_price=110.0,
        p_dip_touched=0.7, p_rally_given_dip=0.6, p_round_trip=0.35,
        p_bag_hold=0.1, p_no_trade_rally_first=0.1, p_neither=0.1,
        expected_days_to_dip=5.0, expected_days_dip_to_rally=8.0,
        expected_gain_per_share=5.0, expected_bag_hold_loss=2.0,
        net_ev_per_share=0.05, ev_pct_of_dip=0.005,
        verdict_subtype="WAIT-FOR-DIP", p_profitable=0.35,
    )
    line = _headline_card(
        ticker="TST", spot=105.0, horizon_days=20, best=best,
        ev_pct_of_dip=0.005, met_threshold_strict=True,
        method_check={}, parabola_filter_refused=False,
        trend_filter_refused=False, ev_hurdle_refused=False,
        snapshot=_Snap(), pass2=None,
    )
    assert "BUY" in line
    assert "AI:" not in line

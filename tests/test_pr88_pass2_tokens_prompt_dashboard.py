"""Tests for PR #88 — Pass 2 token cap + bidirectional prompt + dashboard UX.

== Root-cause data the user ran ==
Cycle 2026-05-27 09:30 with PR #87 produced 0 BUYs. The CSV diagnostic
revealed:
  - 10 of 26 T2 tickers had Pass 2 drift BLANK (Pass 2 failed silently)
  - Pass 2 logs showed "JSON parse error after sanitisation: Expecting
    ',' delimiter: line 98 column 6 (char 7806)" for parabola-flagged
    tickers — output hit pass2_max_tokens=2500 and was truncated.
  - The 16 tickers where Pass 2 succeeded ALL had drift revised DOWN
    — adversarial framing was being interpreted as reflexive pessimism.

== PR #88 fixes ==
  1. pass2_max_tokens 2500 → 6000 (T2 + T3): truncation root cause.
  2. Pass 2 prompt opening rewritten: "PRECISION REVIEW" not
     "adversarial critic"; explicit instruction to revise BIDIRECTIONALLY
     and to KEEP Pass 1 when well-calibrated.
  3. Dashboard UX (operator critiques):
     a. Main row trimmed (Spot/Dip/Rally/P(RT)/EV bps moved to detail row)
     b. Ticker-navigation strip above the table
     c. ★ winner marker removed (always WAIT-wins was misleading)
     d. Sort: BUYs by P(rally)×gain desc (with EV bps fallback)
     e. Plain-English tooltips for P(RT), Ambiguity, EV
     f. Per-ticker dashboard link → 📊 icon next to ticker name
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# 1. pass2_max_tokens raised to prevent truncation
# =============================================================================

def test_pass2_max_tokens_sufficient_for_full_response():
    """T2 and T3 must have pass2_max_tokens ≥ 4500 so the structured
    Pass 2 response (drift + reasoning fields + catalysts +
    parabola_override block) doesn't truncate mid-JSON."""
    from src.config import AI_TIERS
    for tier_name in ("T2", "T3"):
        tier = AI_TIERS[tier_name]
        assert tier.pass2_max_tokens >= 4500, (
            f"{tier_name} pass2_max_tokens={tier.pass2_max_tokens} is too "
            f"low — PR #87's parabola override section + audit-round "
            f"reasoning fields push actual output to 7000-8500 chars. "
            f"Pass 2 will silently truncate on parabola-flagged tickers."
        )


# =============================================================================
# 2. Pass 2 prompt is BIDIRECTIONAL not adversarial
# =============================================================================

def test_pass2_prompt_no_adversarial_framing():
    """The prompt must NOT open with 'adversarial critic' framing — that
    was being interpreted as 'always revise down', producing 16/16
    downward revisions in cycle 09:30. PR #88 uses 'precision review'
    with explicit anti-pessimism instruction."""
    from src.ai_layer import build_ai_pass2_prompt
    from src.engine import AIPassOutput
    from types import SimpleNamespace

    pass1 = AIPassOutput(
        pass_number=1, drift_estimate=0.10, drift_range=(-0.10, 0.30),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[], bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.05, raw_sources_cited=4,
    )
    snapshot = SimpleNamespace(spot=100.0)
    prompt = build_ai_pass2_prompt(
        "TEST", snapshot, pass1,
        mc_marginal_summary={"p_up": "60%", "p_down": "40%", "bracket_pct_str": "10%"},
        sigma_triangulation_summary={"blended": 0.30, "divergence": 5.0, "anchors": {}},
        prior_posterior_drift=None,
    )
    # Should NOT contain adversarial framing
    assert "adversarial critic" not in prompt.lower()
    # SHOULD contain precision/calibration framing
    assert "PRECISION REVIEW" in prompt
    # SHOULD contain explicit anti-pessimism instruction
    assert "reflexive pessimism" in prompt.lower()
    # SHOULD instruct to KEEP when well-calibrated
    assert "KEEP" in prompt or "keep" in prompt.lower()


# =============================================================================
# 3. Dashboard main row is trimmed (PR #88)
# =============================================================================

def _make_buy(orch, ticker="X", **overrides):
    """Local fixture — populates dual-EV fields for PR #88 sort testing."""
    defaults = dict(
        sigma_class="MID", tier="T2", ambiguity=0.30,
        qualifies_for_t2_plus=True, spot=100.0,
        dip_target=98.0, rally_target=110.0,
        p_round_trip=0.40, ev_bps_of_dip=100.0,
        verdict="BUY", status_note="test",
        verdict_subtype="DIRECT", ev_direct_bps=100.0, ev_wait_bps=80.0,
        p_dip_filled=0.85, p_rally_hit=0.65,
        expected_rally_date="Jun 15, 2026", expected_dip_date="May 30, 2026",
    )
    defaults.update(overrides)
    return orch.TickerDecision(ticker=ticker, **defaults)


def test_main_row_columns_trimmed():
    """PR #88: Spot, Dip, Rally, P(RT), EV bps columns REMOVED from
    main row. They're in the detail row now."""
    from src import orchestrator as orch
    html = orch._render_dashboard_html([_make_buy(orch, "AMAT")], None)
    # Main-row column headers gone
    assert "<th>Spot</th>" not in html
    assert "<th>Dip</th>" not in html
    assert "<th>Rally</th>" not in html
    assert "<th>P(RT)</th>" not in html
    assert "<th>EV bps (%)</th>" not in html
    # Remaining headers present
    assert "<th>Ticker</th>" in html
    assert "<th>σ-class</th>" in html
    assert "<th>Tier</th>" in html
    assert "<th>Ambiguity</th>" in html
    assert "<th>Verdict</th>" in html


# =============================================================================
# 4. Ticker navigation strip above the table
# =============================================================================

def test_ticker_nav_strip_present():
    """Above the table: 'Jump to:' label + chip per ticker."""
    from src import orchestrator as orch
    decisions = [_make_buy(orch, t) for t in ("AMAT", "LRCX", "MU")]
    html = orch._render_dashboard_html(decisions, None)
    assert 'id="ticker-nav"' in html
    assert "Jump to:" in html
    # Each ticker has a nav chip linking to its row anchor
    for t in ("AMAT", "LRCX", "MU"):
        assert f'href="#row-{t}"' in html


def test_ticker_row_has_anchor_id():
    """Each table row gets id='row-<TICKER>' so the nav chip anchors work."""
    from src import orchestrator as orch
    html = orch._render_dashboard_html([_make_buy(orch, "AMAT")], None)
    assert 'id="row-AMAT"' in html


# =============================================================================
# 5. Sort: BUYs by P(rally) × gain (with EV fallback)
# =============================================================================

def test_buys_sort_by_conviction_x_gain():
    """High P(rally) × big gain ranks first within the BUY band."""
    from src import orchestrator as orch
    decisions = [
        _make_buy(orch, "LOW_SCORE", spot=100, rally_target=102,
                  p_rally_hit=0.30, ev_bps_of_dip=20),
        _make_buy(orch, "HIGH_SCORE", spot=100, rally_target=115,
                  p_rally_hit=0.70, ev_bps_of_dip=15),
        _make_buy(orch, "MID_SCORE", spot=100, rally_target=108,
                  p_rally_hit=0.50, ev_bps_of_dip=18),
    ]
    html = orch._render_dashboard_html(decisions, None)
    top = html.find('id="row-HIGH_SCORE"')
    mid = html.find('id="row-MID_SCORE"')
    low = html.find('id="row-LOW_SCORE"')
    assert top < mid < low, (
        f"PR #88 expected sort by P(rally)×gain: HIGH(0.7×15%)=10.5 > "
        f"MID(0.5×8%)=4.0 > LOW(0.3×2%)=0.6. Got positions: "
        f"HIGH={top}, MID={mid}, LOW={low}"
    )


# =============================================================================
# 6. ★ winner marker is GONE
# =============================================================================

def test_no_star_winner_marker():
    from src import orchestrator as orch
    html = orch._render_dashboard_html([_make_buy(orch, "AMAT")], None)
    assert "★ winner" not in html


# =============================================================================
# 7. Plain-English tooltips
# =============================================================================

def test_prt_tooltip_explains_conditional_relationship():
    """PR #88: tooltip should explain why P(RT) ≠ P(dip)×P(rally)."""
    from src.orchestrator import _prt_tooltip_text
    tip = _prt_tooltip_text(0.38)
    assert "NOT equal to" in tip or "Not equal to" in tip
    assert "conditional" in tip.lower() or "downward drift" in tip.lower()


def test_ambiguity_tooltip_clarifies_not_touch_probability():
    """PR #88: ambiguity tooltip must clarify it's NOT a touch probability."""
    from src.orchestrator import _ambiguity_tooltip_text
    tip = _ambiguity_tooltip_text(0.34, "T2")
    assert "NOT a touch probability" in tip or "not a touch probability" in tip.lower()
    assert "SELF" in tip or "self-confidence" in tip.lower()


def test_ev_tooltip_uses_100_trades_framing():
    """PR #88: EV tooltip must explain the '100 trades average' framing."""
    from src.orchestrator import _ev_tooltip_text
    tip = _ev_tooltip_text(-57.0)
    assert "100 times" in tip
    assert "average" in tip.lower()


# =============================================================================
# 8. Per-ticker dashboard link as 📊 icon
# =============================================================================

def test_per_ticker_link_icon_renders():
    """PR #88: per-ticker analysis link is now a 📊 icon next to the
    ticker name (was on the Dip cell, which got removed)."""
    from src import orchestrator as orch
    html = orch._render_dashboard_html([_make_buy(orch, "LRCX")], None)
    assert "ticker-details-link" in html
    assert "lrcx_dipnrally_dashboard.html" in html
    # Icon emoji
    assert "📊" in html

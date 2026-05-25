"""Tests for the 2026-05-25 dashboard cosmetic refresh.

Changes covered:
  1. Title is "Dip and Rally Engine"
  2. Collapsible legend with verdict + column meanings
  3. Default sort: BUYs first by EV desc, then refusals, then WAIT/FAIL
  4. Ticker cells link to Trading212 (new tab)
  5. Dip cells link to per-ticker engine dashboard
  6. Edge-aware tooltips on Ambiguity / P(RT) / EV bps
  7. Spot price source line (weekend vs weekday-live)
  8. Run cost + annual projection
  9. Scroll-to-top button
  10. Mobile responsive (viewport meta + media query)
  11. DELISTED removed from summary tiles (operator said one-off)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import orchestrator as orch


def _make_buy(ticker, ev_bps=100.0, ambiguity=0.15):
    """Convenience: instantiate a BUY-verdict TickerDecision."""
    return orch.TickerDecision(
        ticker=ticker, sigma_class="MID", tier="T0",
        ambiguity=ambiguity, qualifies_for_t2_plus=True,
        spot=100.0, dip_target=98.0, rally_target=110.0,
        p_round_trip=0.62, ev_bps_of_dip=ev_bps,
        verdict="BUY", status_note="dip → rally",
    )


def _make_refused(ticker, verdict, ev_bps=-50.0):
    return orch.TickerDecision(
        ticker=ticker, sigma_class="HIGH", tier="T1",
        ambiguity=0.50, qualifies_for_t2_plus=False,
        spot=200.0, dip_target=195.0, rally_target=220.0,
        p_round_trip=0.50, ev_bps_of_dip=ev_bps,
        verdict=verdict, status_note=f"refused on {verdict}",
    )


# =============================================================================
# 1. Title
# =============================================================================
def test_title_is_dip_and_rally_engine():
    html = orch._render_dashboard_html([_make_buy("X")], None)
    assert "<h1>Dip and Rally Engine</h1>" in html
    assert "<title>Dip and Rally Engine" in html


# =============================================================================
# 2. Collapsible legend
# =============================================================================
def test_legend_collapsible_present():
    html = orch._render_dashboard_html([_make_buy("X")], None)
    assert 'id="legendWrapper"' in html
    assert 'id="legendToggle"' in html
    assert "Legend — verdicts" in html


def test_legend_explains_verdicts_in_plain_english():
    html = orch._render_dashboard_html([_make_buy("X")], None)
    # First-investor-friendly descriptions for the main verdicts
    assert "positive-expected-return swing setup" in html
    assert "Don't trade — the math says you lose money" in html
    assert "Buying a blow-off move without a thesis" in html
    assert "three independent math models" in html
    assert "one bet expressed twice" in html


def test_legend_explains_column_headers():
    html = orch._render_dashboard_html([_make_buy("X")], None)
    assert "volatility bucket" in html  # σ-class
    assert "AI compute level" in html   # Tier
    assert "math layer uncertainty score" in html  # Ambiguity
    assert "joint probability of round-trip" in html  # P(RT)
    assert "expected return per share after friction" in html  # EV bps


# =============================================================================
# 3. Default sort
# =============================================================================
def test_default_sort_buys_first_by_ev_desc():
    """BUY rows should appear before refusals; among BUYs, higher EV first."""
    decisions = [
        _make_refused("LOSER", "REFUSED-EV", ev_bps=-200.0),
        _make_buy("MID_BUY", ev_bps=100.0),
        _make_buy("TOP_BUY", ev_bps=180.0),
        _make_buy("LOW_BUY", ev_bps=60.0),
    ]
    html = orch._render_dashboard_html(decisions, None)
    # Find row order via positions of ticker names
    top = html.find(">TOP_BUY<")
    mid = html.find(">MID_BUY<")
    low = html.find(">LOW_BUY<")
    loser = html.find(">LOSER<")
    assert top < mid < low < loser, \
        f"order wrong: TOP={top}, MID={mid}, LOW={low}, LOSER={loser}"


def test_refused_correlated_sorts_after_buys_before_other_refusals():
    """REFUSED-CORRELATED was a BUY before the gate dropped it — it
    should sort just after the BUYs (substitute idea slot)."""
    decisions = [
        _make_refused("CORR", "REFUSED-CORRELATED", ev_bps=120.0),
        _make_refused("EV_REF", "REFUSED-EV", ev_bps=-50.0),
        _make_buy("REAL_BUY", ev_bps=150.0),
    ]
    html = orch._render_dashboard_html(decisions, None)
    real = html.find(">REAL_BUY<")
    corr = html.find(">CORR<")
    ev = html.find(">EV_REF<")
    assert real < corr < ev


# =============================================================================
# 4 + 5. Ticker → Trading212 (new tab), Dip → engine dashboard
# =============================================================================
def test_ticker_links_to_trading212_in_new_tab():
    html = orch._render_dashboard_html([_make_buy("LRCX")], None)
    assert "trading212.com/trading-instruments/invest/LRCX.US" in html
    # New-tab attributes
    assert 'target="_blank"' in html
    assert 'rel="noopener"' in html


def test_dip_cell_links_to_per_ticker_dashboard():
    html = orch._render_dashboard_html([_make_buy("LRCX")], None, href_prefix="")
    assert 'href="lrcx_dipnrally_dashboard.html"' in html


def test_dip_cell_uses_href_prefix_for_audit_copy():
    html = orch._render_dashboard_html([_make_buy("LRCX")], None, href_prefix="../")
    assert 'href="../lrcx_dipnrally_dashboard.html"' in html


# =============================================================================
# 6. Edge-aware tooltips
# =============================================================================
def test_tooltips_present_on_ambiguity_prt_ev():
    html = orch._render_dashboard_html([_make_buy("X")], None)
    # Three tooltip wrappers per BUY row (Ambiguity, P(RT), EV)
    assert html.count('class="tt"') >= 3


def test_tooltip_text_includes_bps_and_percent():
    html = orch._render_dashboard_html([_make_buy("X", ev_bps=182.0)], None)
    assert "+182 bps EV (+1.82%)" in html


def test_tooltip_text_per_ambiguity_range():
    """LOW / MEDIUM / HIGH text varies by ambiguity value."""
    low = orch._render_dashboard_html([_make_buy("X", ambiguity=0.05)], None)
    high = orch._render_dashboard_html([_make_buy("X", ambiguity=0.75)], None)
    assert "VERY LOW" in low
    assert "HIGH" in high


def test_edge_aware_tooltip_script_present():
    """Inline JS positions tooltips via getBoundingClientRect — clamps to viewport."""
    html = orch._render_dashboard_html([_make_buy("X")], None)
    assert "getBoundingClientRect" in html
    assert "window.innerWidth" in html


# =============================================================================
# 7. Spot price source line
# =============================================================================
def test_spot_source_line_present():
    html = orch._render_dashboard_html([_make_buy("X")], None)
    assert "Spot prices:" in html
    # One of the two states (weekend close OR live quote) must be present
    assert (
        "markets closed" in html
        or "Live FMP quote" in html
    )


def test_spot_source_weekend_logic():
    """On weekends, indicate prior close. _spot_source_line() reads
    datetime.now() at module evaluation, so this test verifies the
    function returns the right string for both states."""
    from datetime import datetime
    from unittest.mock import patch

    # Mock a Saturday
    with patch("src.orchestrator.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 5, 23, 12, 0)  # Saturday
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        line = orch._spot_source_line()
        assert "markets closed weekend" in line

    # Mock a Tuesday
    with patch("src.orchestrator.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 5, 26, 14, 0)  # Tuesday
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        line = orch._spot_source_line()
        assert "Live FMP quote" in line


# =============================================================================
# 8. Run cost + annual projection
# =============================================================================
def test_cost_line_shows_annual_projection():
    """Engine should show 'Run cost $X · ~$Y p.a. if run every trading day'.
    Y = X * 252 (trading days)."""
    from dataclasses import dataclass

    @dataclass
    class FakeAlloc:
        spent_usd: float
        cap_usd: float = 2.0
        assignments: dict = None
        notes: list = None

    alloc = FakeAlloc(spent_usd=0.76)
    html = orch._render_dashboard_html([_make_buy("X")], alloc)
    assert "Run cost" in html
    assert "$0.76" in html
    expected_annual = 0.76 * 252  # = 191.52
    assert f"${expected_annual:.0f}" in html  # $192
    assert "p.a. if run every trading day" in html


# =============================================================================
# 9. Scroll-to-top button
# =============================================================================
def test_scroll_top_button_present():
    html = orch._render_dashboard_html([_make_buy("X")], None)
    assert 'id="scrollTopBtn"' in html
    assert 'class="scroll-top"' in html
    # JS toggles visibility on scroll
    assert "window.scrollY" in html


# =============================================================================
# 10. Mobile responsive
# =============================================================================
def test_mobile_viewport_meta_present():
    html = orch._render_dashboard_html([_make_buy("X")], None)
    assert 'name="viewport"' in html
    assert "width=device-width" in html


def test_mobile_media_query_present():
    html = orch._render_dashboard_html([_make_buy("X")], None)
    assert "@media (max-width: 768px)" in html


def test_mobile_data_label_attrs():
    """Each data cell needs data-label='...' so CSS pseudo-elements
    render the column name in the mobile card view."""
    html = orch._render_dashboard_html([_make_buy("X")], None)
    for label in ("σ-class", "Tier", "Ambiguity", "Verdict", "Spot",
                   "Dip", "Rally", "P(RT)", "EV bps"):
        assert f'data-label="{label}"' in html


# =============================================================================
# 11. DELISTED removed from summary tiles
# =============================================================================
def test_summary_tiles_no_delisted():
    """Operator said DELISTED was a one-off (VELO3D). No tile needed."""
    html = orch._render_dashboard_html([_make_buy("X")], None)
    # Other tiles present
    assert "<strong>1</strong>BUY" in html
    assert "<strong>0</strong>WAIT" in html
    assert "<strong>0</strong>REFUSED" in html
    assert "<strong>0</strong>FAIL" in html
    # No DELISTED tile
    assert "<strong>0</strong>DELISTED" not in html


# =============================================================================
# Dark theme + subtle pattern
# =============================================================================
def test_dark_theme_palette_present():
    html = orch._render_dashboard_html([_make_buy("X")], None)
    assert "--bg-dark: #0d1117" in html  # GitHub dark
    assert "--text-primary: #f0f6fc" in html  # Bright text


def test_subtle_pattern_overlay():
    html = orch._render_dashboard_html([_make_buy("X")], None)
    # Radial-gradient dot pattern + soft blue accent
    assert "radial-gradient(circle at 1px 1px" in html

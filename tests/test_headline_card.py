"""Tests for PR #56 — trader headline card.

Single-line verdict at the top of every report so operator sees
bottom-line at-a-glance without scrolling the dense math. Verdict
precedence must mirror the headline section below.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.reporter import _headline_card


@dataclass
class _Snap:
    ticker: str = "INTC"
    spot: float = 119.84
    mom_30d: float = 0.92
    timestamp: datetime = datetime(2026, 5, 23, 12, 0)


@dataclass
class _Best:
    dip_price: float = 117.0
    rally_price: float = 128.0
    p_round_trip: float = 0.54
    net_ev_per_share: float = 1.50
    ev_pct_of_dip: float = 0.0080


def _card(**overrides):
    """Default kwargs for headline card; override per test."""
    defaults = dict(
        ticker="INTC",
        spot=119.84,
        horizon_days=60,
        best=_Best(),
        ev_pct_of_dip=0.0080,
        met_threshold_strict=True,
        method_check={"refused": False},
        parabola_filter_refused=False,
        trend_filter_refused=False,
        ev_hurdle_refused=False,
        snapshot=_Snap(),
    )
    defaults.update(overrides)
    return _headline_card(**defaults)


# =============================================================================
# Refusal precedence
# =============================================================================

def test_trend_filter_takes_precedence_over_parabola_and_ev_hurdle():
    """Sacred #14 trend filter is the strictest refusal; should win
    when multiple gates fire simultaneously."""
    card = _card(
        trend_filter_refused=True,
        parabola_filter_refused=True,
        ev_hurdle_refused=True,
    )
    assert "trend filter" in card
    assert "parabola" not in card
    assert "EV/dip" not in card


def test_parabola_filter_takes_precedence_over_ev_hurdle():
    card = _card(
        parabola_filter_refused=True,
        ev_hurdle_refused=True,
    )
    assert "parabola filter" in card
    assert "EV/dip" not in card


def test_method_disagreement_when_best_none_and_refused():
    """Sacred #16: math methods disagree → best set to None at engine
    layer, method_check.refused=True. Headline must surface this."""
    card = _card(
        best=None,
        method_check={"refused": True, "refusals": ["MC vs PDE diverge"]},
    )
    assert "math methods disagree" in card


def test_ev_hurdle_refused_shows_bps():
    """Sacred #13: when EV-hurdle fires, the bps value should be in
    the headline (-72bps tells trader why)."""
    card = _card(
        ev_hurdle_refused=True,
        ev_pct_of_dip=-0.0072,  # -72bps
    )
    assert "EV/dip" in card
    assert "-72.0bps" in card or "-72.0" in card


def test_wait_when_best_none_no_refusal():
    """No qualifying pair found, no refusal fired → WAIT."""
    card = _card(best=None, method_check={"refused": False})
    assert "WAIT" in card
    assert "no dip/rally pair" in card


# =============================================================================
# Below-threshold / negative-EV warnings
# =============================================================================

def test_below_threshold_shows_dip_rally():
    """met_threshold_strict=False → BELOW-THRESHOLD warning with the
    best-by-EV fallback prices visible."""
    card = _card(met_threshold_strict=False)
    assert "BELOW-THRESHOLD" in card
    assert "$117" in card
    assert "$128" in card


def test_negative_ev_explicit():
    """met_threshold_strict=True but net_ev_per_share < 0 → warn
    explicitly that average outcome loses money."""
    best = _Best(net_ev_per_share=-0.50)
    card = _card(best=best)
    assert "NEGATIVE-EV" in card


# =============================================================================
# BUY verdict
# =============================================================================

def test_buy_verdict_shows_full_actionable_data():
    """Clean BUY verdict — every piece a trader needs in one line."""
    card = _card()
    assert "BUY" in card
    assert "117" in card  # dip
    assert "128" in card  # rally
    assert "P(RT)" in card
    assert "54%" in card  # round-trip probability
    assert "EV" in card
    assert "60d" in card  # horizon


def test_buy_uses_check_emoji():
    """Visual signal — ✅ for BUY, ⛔ for refused, ⚠ for warnings."""
    card = _card()
    assert "✅" in card


def test_refused_uses_block_emoji():
    card = _card(ev_hurdle_refused=True)
    assert "⛔" in card


def test_warning_uses_warning_emoji():
    card = _card(met_threshold_strict=False)
    assert "⚠" in card


# =============================================================================
# Format / readability
# =============================================================================

def test_card_is_single_line():
    """Headline card must be ONE line — trader scans top of report
    and gets verdict immediately."""
    card = _card()
    assert "\n" not in card


def test_card_includes_ticker_name():
    card = _card(ticker="RKLB")
    assert "RKLB" in card


def test_card_format_within_terminal_width():
    """Reasonable terminal width target: ≤ 200 chars (handles wide
    catalyst names etc.). Hard cap so we don't surprise narrow
    terminals."""
    card = _card()
    assert len(card) <= 200


def test_all_refusal_paths_distinct_text():
    """The 4 refusal paths must produce visibly distinct headlines so
    operator can recognize the reason at-a-glance."""
    cards = {
        "trend": _card(trend_filter_refused=True),
        "parabola": _card(parabola_filter_refused=True),
        "method": _card(best=None, method_check={"refused": True}),
        "ev": _card(ev_hurdle_refused=True),
    }
    # All distinct.
    assert len(set(cards.values())) == 4

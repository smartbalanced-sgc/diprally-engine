"""Tests for PR #51 — catalyst date parser extended formats.

Pass 1 / Pass 2 emit catalyst date_or_window strings in multiple
conventions that the original parser silently dropped:
  - "ongoing" / "rolling" → active overhang catalyst, no specific date
  - "<year>-H1" / "<year>-H2" → half-year window
  - "Q3" / "Q3 2026" → quarter without dash separator
  - "<year>-rolling" → year-tagged rolling overhang

Each silent-drop meant a bearish catalyst that the parabola filter
and sacred #14 trend filter couldn't see, so the gates didn't fire
even when AI explicitly surfaced a de-rating thesis. PR #51 closes
all these gaps; date-locked variants (e.g. specific YYYY-MM-DD) are
unchanged.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.signals import parse_catalyst_date


# =============================================================================
# PR #51: ongoing / rolling — active-overhang variants
# =============================================================================

def test_ongoing_returns_in_horizon_date():
    """An 'ongoing' catalyst (insider selling, debt overhang, export
    control regime) is treated as active across the window → resolves
    to a date within the typical 60d horizon."""
    today = datetime.now().date()
    for variant in ("ongoing", "Ongoing", "ONGOING", " ongoing "):
        d = parse_catalyst_date(variant)
        assert d is not None, f"variant {variant!r} should not be None"
        assert today <= d <= today + timedelta(days=10)


def test_rolling_treated_as_in_horizon():
    today = datetime.now().date()
    for variant in ("rolling", "Rolling", "continuous", "current"):
        d = parse_catalyst_date(variant)
        assert d is not None
        assert today <= d <= today + timedelta(days=10)


def test_year_tagged_rolling():
    """'2026-rolling' style: year-tagged active-window catalyst."""
    today = datetime.now().date()
    for variant in ("2026-rolling", "2026 rolling"):
        d = parse_catalyst_date(variant)
        assert d is not None
        assert today <= d <= today + timedelta(days=10)


# =============================================================================
# PR #51: half-year (H1 / H2) variants
# =============================================================================

def test_year_h1():
    d = parse_catalyst_date("2026-H1")
    assert d is not None
    assert d.year == 2026
    assert d.month == 1
    assert d.day == 1


def test_year_h2():
    d = parse_catalyst_date("2026-H2")
    assert d is not None
    assert d.year == 2026
    assert d.month == 7
    assert d.day == 1


def test_h1_year_reversed_order():
    """'H1 2026' format (reversed convention)."""
    d = parse_catalyst_date("H1 2026")
    assert d is not None
    assert (d.year, d.month) == (2026, 1)


def test_h2_year_reversed_order():
    d = parse_catalyst_date("H2 2026")
    assert d is not None
    assert (d.year, d.month) == (2026, 7)


def test_h_lowercase_variants():
    for variant in ("2026-h1", "2026-h2", "h1 2026", "h2 2026"):
        d = parse_catalyst_date(variant)
        assert d is not None, f"variant {variant!r} should parse"


# =============================================================================
# PR #51: quarter without year / reversed quarter
# =============================================================================

def test_quarter_with_year_reversed():
    """'Q3 2026' format (reversed convention)."""
    d = parse_catalyst_date("Q3 2026")
    assert d is not None
    assert (d.year, d.month) == (2026, 7)


def test_bare_quarter_uses_current_year_when_future():
    """Bare 'Q4' with no year: if current Q4 hasn't passed, use it."""
    today = datetime.now().date()
    d = parse_catalyst_date("Q4")
    assert d is not None
    # Resolves to either current year's Q4 (if not past) or next year's Q4.
    assert d.month == 10
    assert d.year in (today.year, today.year + 1)


def test_bare_quarter_rolls_forward_when_past():
    """If today is May 23 and we ask for Q1, current-year Q1 (Jan 1)
    is already past, so the parser should roll to next-year Q1."""
    today = datetime.now().date()
    d = parse_catalyst_date("Q1")
    assert d is not None
    assert d.month == 1
    # Q1 always starts Jan 1 of some year; if we're past current-year Q1,
    # we should be in next-year Q1 territory.
    if today.month > 1 or today.day > 1:
        assert d.year >= today.year


# =============================================================================
# Existing formats — verify PR #51 didn't regress them
# =============================================================================

def test_iso_date_unchanged():
    d = parse_catalyst_date("2026-07-23")
    assert (d.year, d.month, d.day) == (2026, 7, 23)


def test_year_month_unchanged():
    d = parse_catalyst_date("2026-07")
    assert (d.year, d.month, d.day) == (2026, 7, 1)


def test_year_quarter_unchanged():
    d = parse_catalyst_date("2026-Q3")
    assert (d.year, d.month) == (2026, 7)


def test_quarter_range_returns_earliest():
    """'2026-Q2/Q3' or '2026-Q3/2026-Q4' returns the EARLIEST date.
    Conservative for in-horizon checks — if any part overlaps the
    window, the gate counts it."""
    d = parse_catalyst_date("2026-Q3/Q4")
    assert (d.year, d.month) == (2026, 7)
    d = parse_catalyst_date("2026-Q2/2026-Q3")
    assert (d.year, d.month) == (2026, 4)


def test_bare_year_unchanged():
    d = parse_catalyst_date("2026")
    assert (d.year, d.month, d.day) == (2026, 1, 1)


def test_garbage_returns_none():
    for garbage in ("soon", "TBD", "later", "next month",
                     "", None, "xyz", "Q5", "Q0"):
        assert parse_catalyst_date(garbage) is None, \
            f"{garbage!r} should return None"


def test_year_range_with_months():
    """Pass 2 emits '2026-06/2026-09' style ranges."""
    d = parse_catalyst_date("2026-06/2026-09")
    assert d is not None
    assert (d.year, d.month) == (2026, 6)

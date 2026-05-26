"""Tests for PR #81 — audit cleanup (findings #12 + #13).

#12 (revision_momentum calendar-day lookback) — documentation only.
    The audit graded this informational and explicitly defensible
    (analyst events are calendar-time-natural). Test just guards
    against an accidental "fix" to trading days.

#13 (verdict precedence buries EV refusal when method also refuses) —
    new helper `_all_refusal_reasons` collects ALL triggered refusals
    so the CSV's new `refusal_reasons_all` column captures joint
    failures. `verdict_state` keeps its single-headline label for the
    operator-facing dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# #12 — calendar-day lookback in revision momentum is intentional
# =============================================================================

def test_revision_momentum_uses_calendar_days_documented():
    """Anti-fix guard: the function MUST keep using calendar-day
    `age_days = (today - gdate).days`. If a future contributor
    'fixes' this to trading days via market_calendar, the docstring's
    explicit note should have caught them — but assert the behavior
    too so the test ratchet stays."""
    from src import signals
    src = (_REPO_ROOT / "src" / "signals.py").read_text()
    assert "age_days = (today - gdate).days" in src, (
        "revision-momentum age must remain calendar-day arithmetic "
        "(analyst events are calendar-natural, see PR #81 docstring)"
    )
    # Docstring note must be present so the next reader knows.
    doc = signals.signal_from_revision_momentum.__doc__ or ""
    assert "calendar" in doc.lower()


# =============================================================================
# #13 — _all_refusal_reasons captures joint refusals
# =============================================================================

def test_all_refusal_reasons_empty_when_nothing_refused():
    from src.engine import _all_refusal_reasons
    out = _all_refusal_reasons(
        best=object(),
        method_check=None,
        trend_filter_refused=False,
        parabola_filter_refused=False,
        ev_hurdle_refused=False,
    )
    assert out == []


def test_all_refusal_reasons_records_method_and_ev_jointly():
    """The flagship case: method-disagreement nullified `best` AND
    ev_hurdle was already refused before that. Pre-PR-#81 only
    verdict_state=REFUSED-METHOD was recorded — EV failure invisible
    to W10 aggregates. Now both surface in `refusal_reasons_all`."""
    from src.engine import _all_refusal_reasons
    out = _all_refusal_reasons(
        best=None,  # method check set this to None
        method_check={"refused": True, "is_anchor": False, "refusals": ["x"]},
        trend_filter_refused=False,
        parabola_filter_refused=False,
        ev_hurdle_refused=True,
    )
    assert "METHOD" in out
    assert "EV" in out


def test_all_refusal_reasons_anchor_method_check_doesnt_count():
    """method_check.is_anchor=True means the pair didn't qualify and
    we re-ran the check against a class-anchor pair — that's a
    sanity diagnostic, NOT a refusal of an actual recommendation."""
    from src.engine import _all_refusal_reasons
    out = _all_refusal_reasons(
        best=None,
        method_check={"refused": True, "is_anchor": True, "refusals": ["x"]},
        trend_filter_refused=False,
        parabola_filter_refused=False,
        ev_hurdle_refused=False,
    )
    assert "METHOD" not in out


def test_all_refusal_reasons_records_all_four():
    """Stress case: TREND + PARABOLA + METHOD + EV all fire — list
    contains all four labels in priority order."""
    from src.engine import _all_refusal_reasons
    out = _all_refusal_reasons(
        best=None,
        method_check={"refused": True, "is_anchor": False},
        trend_filter_refused=True,
        parabola_filter_refused=True,
        ev_hurdle_refused=True,
    )
    assert out == ["TREND", "PARABOLA", "METHOD", "EV"]


def test_verdict_state_still_single_label_when_multiple_refusals():
    """verdict_state (operator-facing) keeps its 'show one headline
    label' behavior. The joint-refusal info goes in refusal_reasons_all,
    not in verdict_state."""
    from src.engine import _compute_verdict_state
    state = _compute_verdict_state(
        best=None,
        met_threshold_strict=False,
        method_check={"refused": True, "is_anchor": False},
        trend_filter_refused=False,
        parabola_filter_refused=False,
        ev_hurdle_refused=True,
    )
    assert state == "REFUSED-METHOD"
    # Operator dashboard renders only this label; no compound string.


def test_csv_schema_includes_refusal_reasons_all():
    """The new CSV column must be in the schema so DictWriter doesn't
    silently drop it via extrasaction='ignore'."""
    from src.engine import CSV_COLUMNS
    assert "refusal_reasons_all" in CSV_COLUMNS

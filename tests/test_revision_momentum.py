"""Tests for signal_from_revision_momentum — W6 PR #35.

Pure-function tests against the YAML-loaded thresholds. The FMP-bound
fetch_grades_history is exercised in integration smoke runs.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import SIGNAL_REVISION_MOMENTUM
from src.signals import signal_from_revision_momentum


TODAY = date(2026, 5, 22)


def _grade(action, days_ago):
    """Build a single FMP-shaped grade-change row."""
    d = TODAY - timedelta(days=days_ago)
    return {"action": action, "publishedDate": d.strftime("%Y-%m-%d")}


def test_empty_grades_returns_none_signal():
    result = signal_from_revision_momentum([], today=TODAY)
    assert result["drift"] is None
    assert result["confidence"] == "LOW"


def test_no_directional_actions_returns_none_signal():
    """maintain / init / reiterated don't carry direction → _none_signal."""
    grades = [_grade("maintain", 5), _grade("init", 10),
              _grade("reiterated", 20)]
    result = signal_from_revision_momentum(grades, today=TODAY)
    assert result["drift"] is None


def test_three_recent_upgrades_is_bullish():
    grades = [_grade("upgrade", 5), _grade("upgrade", 10),
              _grade("upgrade", 15)]
    result = signal_from_revision_momentum(grades, today=TODAY)
    assert result["drift"] > 0
    # Three actions = MEDIUM confidence per default thresholds.
    assert result["confidence"] in {"MEDIUM", "HIGH"}
    assert "3 upgrades" in result["notes"]


def test_three_recent_downgrades_is_bearish():
    grades = [_grade("downgrade", 5), _grade("downgrade", 10),
              _grade("downgrade", 15)]
    result = signal_from_revision_momentum(grades, today=TODAY)
    assert result["drift"] < 0
    assert "3 downgrades" in result["notes"]


def test_mixed_actions_can_cancel_out():
    """One upgrade + one downgrade in same bucket → zero weighted score."""
    grades = [_grade("upgrade", 5), _grade("downgrade", 5)]
    result = signal_from_revision_momentum(grades, today=TODAY)
    assert result["drift"] == 0.0


def test_recent_actions_weighted_more_than_old():
    """One upgrade 5 days ago vs one downgrade 70 days ago → bullish net.
    Recent_weight is full (1.0); older_weight is 0.3."""
    grades = [_grade("upgrade", 5), _grade("downgrade", 70)]
    result = signal_from_revision_momentum(grades, today=TODAY)
    assert result["drift"] > 0


def test_outside_lookback_window_ignored():
    """Action older than lookback_days has no effect."""
    too_old = SIGNAL_REVISION_MOMENTUM.lookback_days + 30
    grades = [_grade("upgrade", too_old)]
    result = signal_from_revision_momentum(grades, today=TODAY)
    # Only out-of-window action → no in-window directional actions →
    # _none_signal.
    assert result["drift"] is None


def test_drift_capped_at_configured_max():
    """Many upgrades shouldn't push drift past drift_cap_abs."""
    grades = [_grade("upgrade", 5) for _ in range(20)]
    result = signal_from_revision_momentum(grades, today=TODAY)
    cap = SIGNAL_REVISION_MOMENTUM.drift_cap_abs
    assert result["drift"] <= cap + 1e-9
    assert result["drift"] >= -cap - 1e-9


def test_confidence_ladder():
    cfg = SIGNAL_REVISION_MOMENTUM
    one = signal_from_revision_momentum([_grade("upgrade", 5)], today=TODAY)
    medium = [_grade("upgrade", 5) for _ in range(cfg.conf_medium_count)]
    high = [_grade("upgrade", 5) for _ in range(cfg.conf_high_count)]
    assert signal_from_revision_momentum(one if isinstance(one, list) else
                                          [_grade("upgrade", 5)],
                                          today=TODAY)["confidence"] == "LOW"
    assert signal_from_revision_momentum(medium, today=TODAY)["confidence"] in {"MEDIUM", "HIGH"}
    assert signal_from_revision_momentum(high, today=TODAY)["confidence"] == "HIGH"


def test_malformed_rows_skipped_gracefully():
    """Missing dates / non-dict rows / unparseable dates don't crash."""
    grades = [
        {"action": "upgrade"},                      # no date
        {"action": "upgrade", "publishedDate": ""}, # empty date
        "not a dict",                                # garbage
        _grade("upgrade", 10),                      # one valid
    ]
    result = signal_from_revision_momentum(grades, today=TODAY)
    # Only the valid one counts → LOW confidence, positive drift.
    assert result["drift"] > 0
    assert result["confidence"] == "LOW"
    assert result["sources_count"] == 1


def test_future_dated_actions_ignored():
    """Dates in the future are not in-window (age_days < 0)."""
    grades = [_grade("upgrade", -5)]  # 5 days in the future
    result = signal_from_revision_momentum(grades, today=TODAY)
    assert result["drift"] is None  # no in-window directional actions

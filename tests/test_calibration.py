"""Tests for src/calibration.py — W10 PR #47 outcome harness.

Pure-function tests against synthetic price histories. The full
engine-CSV-update integration is exercised by daily orchestrator runs;
this suite verifies the resolver's logic in isolation.

Sacred resolution rules (verify each):
  - OPEN until row_date + horizon_days has passed
  - dip must come BEFORE rally for round_trip_completed=True
  - bag_hold = dip_touched AND no rally_after AND terminal < dip
  - resolution is idempotent — calling twice doesn't double-count
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.calibration import (
    STATUS_OPEN,
    STATUS_RESOLVED,
    apply_outcome_to_row,
    resolve_history,
    resolve_one_row,
)


PREDICTION_DATE = date(2026, 1, 15)
TODAY_AFTER_HORIZON = date(2026, 5, 15)  # ~120 days later → past 60d horizon


def _row(spot=100.0, dip=85.0, rally=120.0, horizon=60,
         pred_date=PREDICTION_DATE, **extra):
    """Build a synthetic CSV row dict."""
    return {
        "date": pred_date.strftime("%Y-%m-%d"),
        "spot": str(spot),
        "recommended_dip": str(dip),
        "recommended_rally": str(rally),
        "horizon_days": str(horizon),
        **extra,
    }


def _hist(closes, start=PREDICTION_DATE):
    """Build a synthetic history DataFrame. closes is a list of daily
    closes starting AT the prediction date (closes[0] = spot at
    prediction date). Returned df has trading days only (skips
    weekends) using a simple weekday filter."""
    dates = []
    d = start
    while len(dates) < len(closes):
        if d.weekday() < 5:  # Mon-Fri
            dates.append(d)
        d += timedelta(days=1)
    return pd.DataFrame({"Date": pd.to_datetime(dates), "Close": closes})


def test_open_when_window_not_closed():
    """Row that's only 10 days old with 60-day horizon → still OPEN."""
    row = _row()
    today = PREDICTION_DATE + timedelta(days=10)
    outcome = resolve_one_row(row, _hist([100] * 80), today=today)
    assert outcome.status == STATUS_OPEN
    assert outcome.dip_touched is None


def test_round_trip_completed_dip_then_rally():
    """Path: spot $100 → dip to $82 (touched dip $85) day 10 → rally to
    $125 (touched rally $120) day 40. Round-trip completes."""
    closes = [100] * 5 + [95, 90, 87, 84, 82, 86, 92, 100, 108, 115, 122, 125]
    closes += [125] * (65 - len(closes))  # pad to 65 bars (PR #76: ≥60 needed)
    row = _row()
    outcome = resolve_one_row(row, _hist(closes), today=TODAY_AFTER_HORIZON)
    assert outcome.status == STATUS_RESOLVED
    assert outcome.dip_touched is True
    assert outcome.rally_touched_after_dip is True
    assert outcome.round_trip_completed is True
    assert outcome.bag_hold_realized is False
    # Dip touched at bar index 8 (close=82, first ≤ 85 is the 84 at idx 8
    # in the post-prediction window — actual depends on slice; just
    # assert it's a sensible positive integer).
    assert outcome.dip_touch_day is not None and outcome.dip_touch_day > 0
    assert outcome.rally_touch_day > outcome.dip_touch_day


def test_rally_then_dip_is_NOT_round_trip():
    """Path: spot $100 → rally to $125 day 5 (sacred: dip MUST come
    first) → dip to $80 day 30. Even though both targets are touched,
    round_trip_completed is False because dip wasn't first."""
    closes = [100] * 2 + [110, 118, 125] + [120, 110, 100, 92, 85, 78, 80]
    closes += [80] * (65 - len(closes))
    row = _row()
    outcome = resolve_one_row(row, _hist(closes), today=TODAY_AFTER_HORIZON)
    assert outcome.status == STATUS_RESOLVED
    # Rally first → first time dip is touched (day ~10), but rally
    # was already touched BEFORE dip → rally-after-dip = False.
    assert outcome.dip_touched is True
    assert outcome.rally_touched_after_dip is False
    assert outcome.round_trip_completed is False


def test_bag_hold_realized():
    """Dip touched, no rally, terminal close below dip → bag_hold."""
    # Dip target $85, rally target $120. Path: drops to $78, recovers
    # to $90 but never touches rally, ends at $82 (below dip).
    closes = [100, 95, 88, 82, 78, 80, 85, 90, 88, 85, 82, 80, 82]
    closes += [82] * (65 - len(closes))
    row = _row()
    outcome = resolve_one_row(row, _hist(closes), today=TODAY_AFTER_HORIZON)
    assert outcome.status == STATUS_RESOLVED
    assert outcome.dip_touched is True
    assert outcome.rally_touched_after_dip is False
    assert outcome.round_trip_completed is False
    assert outcome.bag_hold_realized is True
    assert outcome.realized_terminal_return < 0


def test_neither_touched():
    """Path stays in [86, 119] — neither dip nor rally touched."""
    closes = [100] * 65
    row = _row()
    outcome = resolve_one_row(row, _hist(closes), today=TODAY_AFTER_HORIZON)
    assert outcome.status == STATUS_RESOLVED
    assert outcome.dip_touched is False
    assert outcome.rally_touched_after_dip is False
    assert outcome.round_trip_completed is False
    assert outcome.bag_hold_realized is False


def test_no_prediction_returns_open():
    """Row with empty dip/rally targets (no qualifying pair) → OPEN
    (nothing to resolve, math layer didn't produce a prediction)."""
    row = _row(dip=0.0, rally=0.0)
    outcome = resolve_one_row(row, _hist([100] * 65), today=TODAY_AFTER_HORIZON)
    assert outcome.status == STATUS_OPEN


def test_missing_history_returns_open():
    """No history DataFrame → can't resolve → OPEN."""
    row = _row()
    outcome = resolve_one_row(row, None, today=TODAY_AFTER_HORIZON)
    assert outcome.status == STATUS_OPEN


def test_realized_max_drawdown_computed():
    """Drawdown from spot $100 to lowest close (say $72) → 28%."""
    closes = [100, 95, 88, 80, 72, 78, 85, 92, 95, 90, 85, 80, 75, 72]
    closes += [72] * (65 - len(closes))
    row = _row()
    outcome = resolve_one_row(row, _hist(closes), today=TODAY_AFTER_HORIZON)
    assert outcome.realized_max_drawdown == pytest.approx(0.28, abs=0.01)


def test_apply_outcome_round_trip_through_csv_strings():
    """apply_outcome_to_row serializes booleans to '1'/'0' so the row
    round-trips cleanly through csv.DictWriter."""
    closes = [100] * 5 + [90, 84, 80, 86, 100, 115, 122] + [122] * 60
    row = _row()
    outcome = resolve_one_row(row, _hist(closes), today=TODAY_AFTER_HORIZON)
    merged = apply_outcome_to_row(row, outcome)
    assert merged["outcome_status"] == STATUS_RESOLVED
    assert merged["dip_touched"] == "1"
    assert merged["rally_touched_after_dip"] == "1"
    assert merged["round_trip_completed"] == "1"
    assert merged["bag_hold_realized"] == "0"
    # Floats formatted as 4-decimal strings.
    assert "." in merged["realized_terminal_return"]


def test_apply_does_not_mutate_input():
    row = _row()
    before = dict(row)
    outcome = resolve_one_row(row, _hist([100] * 65), today=TODAY_AFTER_HORIZON)
    apply_outcome_to_row(row, outcome)
    assert row == before


def test_idempotent_resolution():
    """resolve_history called twice on the same rows shouldn't change
    already-RESOLVED rows or double-count newly-resolved ones."""
    closes = [100] * 65
    row = _row()
    rows = [row]
    df = _hist(closes)
    rows1, n1 = resolve_history(rows, df, today=TODAY_AFTER_HORIZON)
    rows2, n2 = resolve_history(rows1, df, today=TODAY_AFTER_HORIZON)
    assert n1 == 1
    assert n2 == 0  # already resolved
    assert rows2[0]["outcome_status"] == STATUS_RESOLVED


def test_open_row_doesnt_get_status_marker_persisted():
    """A row that's still in its window resolves to OPEN — but
    apply_outcome should NOT downgrade an existing RESOLVED row."""
    row = _row()
    row["outcome_status"] = STATUS_RESOLVED
    row["dip_touched"] = "1"
    # Apply an OPEN outcome (e.g. mid-window re-run) → row should stay
    # RESOLVED because it was already finalized.
    from src.calibration import _open_outcome
    merged = apply_outcome_to_row(row, _open_outcome())
    assert merged["outcome_status"] == STATUS_RESOLVED
    assert merged["dip_touched"] == "1"

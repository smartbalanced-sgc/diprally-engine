"""Tests for PR #76 — trading-day calendar helpers + the 4 call-site fixes
+ holiday-aware dashboard banner + AI-cache hygiene.

Background: PR #75 fixed one calendar-vs-trading-day bug (correlation gate
under-fetched bars). Audit (2026-05-25) found the same root-cause pattern
at 4 more sites:

  1. `engine._has_bearish_derating_catalyst` (sacred #18 parabola filter
     rescue) — used calendar `today + timedelta(days=horizon_days)`,
     creating a ~24-cal-day dead zone (28% of the actual 60-trading-day
     horizon). Bearish catalysts dated 60-84 cal days out were invisible.
  2. `engine._has_supporting_catalyst` (sacred #14 falling-knife rescue) —
     same bug.
  3. `engine.run_pipeline` peer-earnings filter — dropped peer events
     60-84 cal days out from the vol schedule.
  4. `math_utils.build_catalyst_vol_schedule` — indexed schedule by
     calendar offset; earnings at trading day 21 (~30 cal days) was
     written to schedule[30] = trading-day-30 (9 days too late),
     corrupting Brownian-bridge fidelity (sacred #9).
  5. `signals.signal_from_catalyst_proximity` — same as #1/#2 for the
     signal-layer drift contribution.
  6. `calibration.resolve_one_row` + `engine._build_per_day_status` +
     backtest resolver — triggered on calendar-day elapsed rather than
     trading-bar count, locking outcomes ~28% early and biasing W10
     calibration pessimistically.
  7. `ai_cache.today_str()` keyed on wall-clock date — running on a
     market holiday wrote a fresh cache file containing the prior
     trading day's data, contaminating the next real trading day.
  8. `orchestrator._spot_source_line` only checked weekday — said "Live
     FMP quote" on Memorial Day Monday. Misleading.

Fix: new `src.market_calendar` module wraps `pandas_market_calendars`
(XNYS). All call sites switch to `add_trading_days` /
`trading_days_between` / `last_trading_day`.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.market_calendar import (
    add_trading_days,
    early_close_time,
    holiday_name,
    is_trading_day,
    last_trading_day,
    trading_days_between,
)


# =============================================================================
# 1. Calendar primitives — known NYSE dates
# =============================================================================

def test_memorial_day_2026_is_closed():
    """2026-05-25 Memorial Day Monday — NYSE closed."""
    assert is_trading_day(date(2026, 5, 25)) is False
    assert holiday_name(date(2026, 5, 25)) == "Memorial Day"


def test_carter_funeral_2025_01_09_is_closed():
    """Carter funeral — ad-hoc closure. Library knows it (this is why we
    don't hardcode a holiday list)."""
    assert is_trading_day(date(2025, 1, 9)) is False
    # Ad-hoc closures lack named rules — label is "Special closure".
    assert holiday_name(date(2025, 1, 9)) == "Special closure"


def test_good_friday_2026_is_closed():
    """2026-04-03 Good Friday — NYSE closed (moves with Easter — would
    be very easy to mis-encode in a hardcoded list)."""
    assert is_trading_day(date(2026, 4, 3)) is False
    assert holiday_name(date(2026, 4, 3)) == "Good Friday"


def test_juneteenth_2026_is_closed():
    """2026-06-19 Juneteenth (Fri) — NYSE closed (added 2022, hardcoded
    lists could miss this)."""
    assert is_trading_day(date(2026, 6, 19)) is False


def test_regular_tuesday_is_open():
    assert is_trading_day(date(2026, 5, 26)) is True
    assert holiday_name(date(2026, 5, 26)) is None


def test_weekend_not_a_holiday():
    """Weekends close NYSE but aren't 'holidays'. holiday_name returns None."""
    assert is_trading_day(date(2026, 5, 23)) is False  # Sat
    assert holiday_name(date(2026, 5, 23)) is None


def test_christmas_eve_2024_is_half_day():
    """Day-before-Christmas often closes 1pm ET. Library knows."""
    assert is_trading_day(date(2024, 12, 24)) is True
    ec = early_close_time(date(2024, 12, 24))
    assert ec == time(13, 0)


def test_regular_day_has_no_early_close():
    assert early_close_time(date(2026, 5, 26)) is None


# =============================================================================
# 2. last_trading_day / add_trading_days / trading_days_between
# =============================================================================

def test_last_trading_day_on_holiday_returns_prior_friday():
    """Memorial Day Mon 2026-05-25 → last open = Fri 2026-05-22."""
    assert last_trading_day(date(2026, 5, 25)) == date(2026, 5, 22)


def test_last_trading_day_on_saturday_returns_friday():
    assert last_trading_day(date(2026, 5, 23)) == date(2026, 5, 22)


def test_last_trading_day_on_open_day_returns_itself():
    assert last_trading_day(date(2026, 5, 26)) == date(2026, 5, 26)


def test_add_trading_days_skips_memorial_day():
    """Fri 2026-05-22 + 1 trading day = Tue 2026-05-26 (skip weekend +
    Memorial Day Monday)."""
    assert add_trading_days(date(2026, 5, 22), 1) == date(2026, 5, 26)


def test_add_60_trading_days_lands_after_84_calendar_days():
    """60 trading days from Tue 2026-05-26. Pure 5/7 ratio predicts
    ~Aug 18 (cal day 84). With actual holiday calendar (July 4 + any
    other in window), lands slightly later. Sanity: at least 84 cal days."""
    end = add_trading_days(date(2026, 5, 26), 60)
    cal_gap = (end - date(2026, 5, 26)).days
    # 60 trading days = at minimum 60+24 weekend days; +1 for July 4 +
    # buffer for any other in-window holidays.
    assert cal_gap >= 84
    assert cal_gap <= 95
    assert is_trading_day(end)


def test_trading_days_between_skips_holiday():
    """Fri 2026-05-22 to Tue 2026-05-26: only 2 trading days (Fri + Tue).
    Mon 2026-05-25 is Memorial Day → skipped. Sat/Sun → skipped."""
    assert trading_days_between(date(2026, 5, 22), date(2026, 5, 26)) == 2


def test_add_trading_days_handles_negative():
    """Going backwards from Tue past Memorial Day Mon → lands on Fri."""
    assert add_trading_days(date(2026, 5, 26), -1) == date(2026, 5, 22)


def test_trading_days_between_negative_symmetry():
    a = date(2026, 5, 22)
    b = date(2026, 5, 26)
    assert trading_days_between(a, b) == -trading_days_between(b, a)


# =============================================================================
# 3. _has_bearish_derating_catalyst — the audit's flagship case
# =============================================================================

def _ai_pass2_with_catalyst(catalyst_date_str, direction="bearish"):
    """Build a minimal AIPassOutput with one catalyst at the given date."""
    from src.engine import AIPassOutput
    return AIPassOutput(
        pass_number=2, drift_estimate=0.0, drift_range=(-0.10, 0.10),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="neutral",
        catalysts=[{
            "name": "Test catalyst",
            "magnitude": "high",
            "direction_risk": direction,
            "date_or_window": catalyst_date_str,
        }],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.10, raw_sources_cited=4,
    )


def _patch_datetime_now(monkeypatch, module, fixed_dt):
    """Patch `module.datetime.now()` to return fixed_dt while leaving the
    `datetime(...)` constructor intact. Required because signal/engine
    code uses both `.now()` and bare `datetime(y, m, d)` constructor."""
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None): return fixed_dt
    monkeypatch.setattr(module, "datetime", _DT)


def test_bearish_catalyst_at_day_75_cal_IS_detected_for_60_trading_horizon(monkeypatch):
    """The flagship bug. Catalyst dated 75 calendar days out — well
    inside a 60-trading-day horizon (~84 cal). PRE-PR-#76 it was
    invisible. POST-PR-#76 it's detected → parabola filter (sacred #18)
    has its rescue path."""
    from src import engine
    fixed_today = date(2026, 5, 26)
    _patch_datetime_now(monkeypatch, engine,
                          datetime.combine(fixed_today, datetime.min.time()))
    from datetime import timedelta as _td
    cat_date = fixed_today + _td(days=75)
    ai = _ai_pass2_with_catalyst(cat_date.isoformat(), direction="bearish")
    assert engine._has_bearish_derating_catalyst(ai, horizon_days=60) is True


def test_bearish_catalyst_at_day_95_cal_NOT_detected_for_60_trading_horizon(monkeypatch):
    """Past the 60-trading-day horizon → correctly NOT detected."""
    from src import engine
    fixed_today = date(2026, 5, 26)
    _patch_datetime_now(monkeypatch, engine,
                          datetime.combine(fixed_today, datetime.min.time()))
    from datetime import timedelta as _td
    cat_date = fixed_today + _td(days=95)
    ai = _ai_pass2_with_catalyst(cat_date.isoformat(), direction="bearish")
    assert engine._has_bearish_derating_catalyst(ai, horizon_days=60) is False


def test_supporting_catalyst_at_day_75_cal_IS_detected(monkeypatch):
    """Mirror of the bearish case for sacred #14 (falling-knife rescue)."""
    from src import engine
    fixed_today = date(2026, 5, 26)
    _patch_datetime_now(monkeypatch, engine,
                          datetime.combine(fixed_today, datetime.min.time()))
    from datetime import timedelta as _td
    cat_date = fixed_today + _td(days=75)
    ai = _ai_pass2_with_catalyst(cat_date.isoformat(), direction="bullish")
    assert engine._has_supporting_catalyst(ai, horizon_days=60) is True


# =============================================================================
# 4. signal_from_catalyst_proximity — same window semantics
# =============================================================================

def test_signal_catalyst_proximity_includes_late_horizon_catalysts(monkeypatch):
    """Signal layer should also see catalysts 60-84 cal days out for a
    60-trading-day horizon."""
    from src import signals
    fixed_today = date(2026, 5, 26)
    _patch_datetime_now(monkeypatch, signals,
                          datetime.combine(fixed_today, datetime.min.time()))
    from datetime import timedelta as _td
    catalysts = [{
        "name": "Late earnings",
        "magnitude": "high",
        "direction_risk": "bullish",
        "date_or_window": (fixed_today + _td(days=75)).isoformat(),
    }]
    mu, conf, rationale = signals.signal_from_catalyst_proximity(catalysts, horizon_days=60)
    # Bullish + high magnitude → mu > 0. Pre-fix this would be 0 / "no
    # in-window catalysts".
    assert mu > 0


# =============================================================================
# 5. build_catalyst_vol_schedule — spike at correct trading-day index
# =============================================================================

def test_vol_schedule_spike_at_trading_day_index_not_calendar(monkeypatch):
    """Earnings 21 trading days from today (Tue 2026-05-26 → 2026-06-24,
    ~29 cal days). Pre-fix: spike at schedule[29] — wrong. Post-fix:
    spike at schedule[21]."""
    from src import math_utils
    fixed_today = date(2026, 5, 26)
    _patch_datetime_now(monkeypatch, math_utils,
                          datetime.combine(fixed_today, datetime.min.time()))
    # Pick an event 21 trading days from today.
    event = add_trading_days(fixed_today, 21)
    sched = math_utils.build_catalyst_vol_schedule(
        base_vol=0.30,
        horizon_days=60,
        self_earnings_date=datetime.combine(event, datetime.min.time()),
        peer_earnings_dates=[],
        macro_event_dates=[],
    )
    # Index 21 should be elevated (multiplier > 1 × base_vol = 0.30).
    assert sched[21] > 0.30
    # Calendar-day index (29) should NOT be the spike location.
    # (It might be elevated incidentally if 21+window covers 29, so we
    # check that 21 is the LOCAL MAXIMUM.)
    window = 3  # typical pre/post window
    nearby = sched[max(0, 21 - window):min(60, 21 + window + 1)]
    assert sched[21] == nearby.max()


# =============================================================================
# 6. Calibration resolver triggers on trading bars, not calendar days
# =============================================================================

def test_calibration_resolver_waits_for_60_trading_bars(monkeypatch):
    """Pre-fix: resolved when calendar elapsed >= 60 (~43 trading bars).
    Post-fix: only resolves when 60 actual trading bars exist in window."""
    from src import calibration
    import pandas as pd
    # Prediction made 60 calendar days ago (Mar 27 → May 26).
    pred_date = date(2026, 3, 27)
    today = date(2026, 5, 26)
    # Only 42 trading bars in that window (real count). Resolver must wait.
    biz_days = pd.bdate_range(start=pd.Timestamp(pred_date) + pd.Timedelta(days=1),
                                end=pd.Timestamp(today))
    # Build a history df with only those bars.
    history_df = pd.DataFrame({
        "Date": biz_days,
        "Close": [100.0 + i for i in range(len(biz_days))],
    })
    row = {
        "date": pred_date.isoformat(),
        "spot": "100.0",
        "recommended_dip": "95.0",
        "recommended_rally": "110.0",
        "horizon_days": "60",
    }
    outcome = calibration.resolve_one_row(row, history_df, today=today)
    # Window only has ~43 trading bars, horizon is 60 → STILL OPEN.
    assert outcome.status == calibration.STATUS_OPEN


def test_calibration_resolver_locks_when_60_trading_bars_present():
    """When history actually has ≥60 post-prediction bars → resolves."""
    from src import calibration
    import pandas as pd
    pred_date = date(2026, 1, 5)
    today = date(2026, 5, 26)  # ~100 trading bars later
    biz_days = pd.bdate_range(start=pd.Timestamp(pred_date) + pd.Timedelta(days=1),
                                end=pd.Timestamp(today))
    history_df = pd.DataFrame({
        "Date": biz_days,
        "Close": [100.0 + i for i in range(len(biz_days))],
    })
    row = {
        "date": pred_date.isoformat(),
        "spot": "100.0",
        "recommended_dip": "95.0",
        "recommended_rally": "110.0",
        "horizon_days": "60",
    }
    outcome = calibration.resolve_one_row(row, history_df, today=today)
    assert outcome.status == calibration.STATUS_RESOLVED


# =============================================================================
# 7. AI cache key uses last_trading_day
# =============================================================================

def test_ai_cache_today_str_uses_last_trading_day_on_holiday(monkeypatch):
    """Running on Memorial Day Mon 2026-05-25 → cache key is Fri 2026-05-22,
    not Mon. Prevents cache pollution with stale Friday quotes keyed under
    today's date."""
    from src import ai_cache, market_calendar
    fixed_today = datetime(2026, 5, 25, 12, 0)
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None): return fixed_today
    monkeypatch.setattr(ai_cache, "datetime", _DT)
    monkeypatch.setattr(market_calendar, "datetime", _DT)
    assert ai_cache.today_str() == "2026-05-22"


def test_ai_cache_today_str_normal_on_trading_day(monkeypatch):
    from src import ai_cache, market_calendar
    fixed_today = datetime(2026, 5, 26, 12, 0)
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None): return fixed_today
    monkeypatch.setattr(ai_cache, "datetime", _DT)
    monkeypatch.setattr(market_calendar, "datetime", _DT)
    assert ai_cache.today_str() == "2026-05-26"


# =============================================================================
# 8. Dashboard spot-source line — holiday-aware banner
# =============================================================================

def test_spot_source_line_on_memorial_day_says_closed(monkeypatch):
    """Mon 2026-05-25 = Memorial Day. Line must say NYSE closed +
    name the holiday + point at last open day."""
    from src import orchestrator
    fixed_now = datetime(2026, 5, 25, 12, 0)
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None): return fixed_now
    monkeypatch.setattr(orchestrator, "datetime", _DT)
    line = orchestrator._spot_source_line()
    assert "NYSE CLOSED" in line
    assert "Memorial Day" in line
    assert "2026-05-22" in line  # last trading day


def test_spot_source_line_on_weekend_says_weekend(monkeypatch):
    from src import orchestrator
    fixed_now = datetime(2026, 5, 23, 12, 0)  # Saturday
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None): return fixed_now
    monkeypatch.setattr(orchestrator, "datetime", _DT)
    line = orchestrator._spot_source_line()
    assert "weekend" in line.lower()
    assert "2026-05-22" in line


def test_spot_source_line_on_half_day_mentions_early_close(monkeypatch):
    """2024-12-24 Christmas Eve = 1pm ET close."""
    from src import orchestrator
    fixed_now = datetime(2024, 12, 24, 10, 0)
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None): return fixed_now
    monkeypatch.setattr(orchestrator, "datetime", _DT)
    line = orchestrator._spot_source_line()
    assert "half-day" in line
    assert "13:00" in line


def test_spot_source_line_on_regular_trading_day(monkeypatch):
    """Live FMP quote banner on a normal Tuesday."""
    from src import orchestrator
    fixed_now = datetime(2026, 5, 26, 14, 0)
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None): return fixed_now
    monkeypatch.setattr(orchestrator, "datetime", _DT)
    line = orchestrator._spot_source_line()
    # PR #85: wording now reflects /stable/quote endpoint + fallback policy.
    assert "live FMP /stable/quote" in line
    assert "NYSE CLOSED" not in line

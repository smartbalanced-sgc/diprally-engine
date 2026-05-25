"""NYSE trading-day calendar — single source of truth for trading-day arithmetic.

Rationale (PR #76): the engine simulates over a `horizon_days = 60`
**trading**-day window (MC step dt = 1/252). Multiple receivers were doing
`today + timedelta(days=horizon_days)` — that's CALENDAR days, leaving a
~24-day dead zone (28% of the actual horizon) at every site. PR #75 fixed
this at the data-fetch layer; PR #76 fixes the remaining 4 sites and adds
a holiday-aware run-banner.

We use `pandas_market_calendars` (XNYS) — Quantopian heritage, actively
maintained, knows half-days and ad-hoc closures (Carter funeral 2025-01-09,
9/11, Hurricane Sandy, etc). Hardcoded lists go stale; this does not.

Defense in depth: `orchestrator` calls `verify_market_state_via_fmp()` at
run start. If FMP's market-status endpoint disagrees with the library, the
orchestrator fails loud (don't trust either source unilaterally — surface
the conflict to the operator).

Public API:
  - is_trading_day(d)
  - last_trading_day(d)            today if open, else most recent open day
  - add_trading_days(d, n)         signed n; calendar-aware
  - trading_days_between(a, b)     count of trading days in [a, b]
  - holiday_name(d)                "Memorial Day" / "Good Friday" / None
  - early_close_time(d)            datetime.time of early close, or None

Sacred: callers should NEVER do `timedelta(days=N)` with an N that means
trading days. Always route through this module.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd
import pandas_market_calendars as mcal


_NYSE = mcal.get_calendar("XNYS")


def _to_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, pd.Timestamp):
        return d.date()
    raise TypeError(f"unsupported date type: {type(d)}")


@lru_cache(maxsize=4096)
def _valid_days_cached(start_iso: str, end_iso: str) -> tuple:
    """Cached wrapper — _NYSE.valid_days is not cheap on large ranges."""
    idx = _NYSE.valid_days(start_date=start_iso, end_date=end_iso)
    return tuple(t.date() for t in idx)


def is_trading_day(d) -> bool:
    """True iff NYSE has a regular OR half-day session on `d`.

    Half-days count as trading days — they have prices, just an early close.
    The half-day-specific quirk (FMP returning 1pm rather than 4pm close) is
    surfaced via `early_close_time(d)`.
    """
    d = _to_date(d)
    return d in _valid_days_cached(d.isoformat(), d.isoformat())


def last_trading_day(d=None) -> date:
    """Most recent trading day ≤ `d` (today if not provided).

    Holiday Monday → returns Friday. Saturday → returns Friday. Tuesday
    after Memorial-Day-Mon → returns Tuesday (today is open).
    """
    d = _to_date(d) if d is not None else datetime.now().date()
    if is_trading_day(d):
        return d
    # Walk back ≤ 10 days (covers any consecutive-closure cluster — longest
    # in NYSE history was 9/11's 4 trading days, plus weekends = ~6 cal days).
    start = d - timedelta(days=14)
    days = _valid_days_cached(start.isoformat(), d.isoformat())
    if not days:
        raise RuntimeError(
            f"no trading day found in {start}..{d} — calendar data gap?"
        )
    return days[-1]


def add_trading_days(d, n: int) -> date:
    """Return the trading day `n` sessions after `d` (negative `n` goes back).

    `d` may itself be a non-trading day; the result is always a trading day.
    Used for horizon-window math: `add_trading_days(today, 60)` gives the
    correct horizon-end the MC actually simulates to.
    """
    d = _to_date(d)
    if n == 0:
        return last_trading_day(d)
    if n > 0:
        # Generous calendar window: 60 trading days ≈ 84 cal days; add buffer.
        end = d + timedelta(days=int(n * 7 / 5) + 14)
        days = _valid_days_cached(d.isoformat(), end.isoformat())
        # Drop `d` itself if it's a trading day (we want the n-th AFTER).
        days = [x for x in days if x > d]
        if len(days) < n:
            # Calendar data gap — extend window once.
            end = d + timedelta(days=int(n * 7 / 5) + 60)
            days = _valid_days_cached(d.isoformat(), end.isoformat())
            days = [x for x in days if x > d]
            if len(days) < n:
                raise RuntimeError(
                    f"add_trading_days: need {n} sessions after {d}, "
                    f"only found {len(days)} within {end}"
                )
        return days[n - 1]
    # Negative n — go back.
    start = d - timedelta(days=int(abs(n) * 7 / 5) + 14)
    days = _valid_days_cached(start.isoformat(), d.isoformat())
    days = [x for x in days if x < d]
    if len(days) < abs(n):
        start = d - timedelta(days=int(abs(n) * 7 / 5) + 60)
        days = _valid_days_cached(start.isoformat(), d.isoformat())
        days = [x for x in days if x < d]
        if len(days) < abs(n):
            raise RuntimeError(
                f"add_trading_days: need {abs(n)} sessions before {d}, "
                f"only found {len(days)} within {start}"
            )
    return days[-abs(n)]


def trading_days_between(a, b) -> int:
    """Number of trading sessions in [a, b] INCLUSIVE. Negative if b < a.

    Note inclusive semantics: trading_days_between(d, d) = 1 if d is a
    trading day. For "how many sessions AFTER d" use trading_days_after.
    """
    a = _to_date(a)
    b = _to_date(b)
    if b < a:
        return -trading_days_between(b, a)
    days = _valid_days_cached(a.isoformat(), b.isoformat())
    return len(days)


def trading_days_after(d, target) -> int:
    """Number of trading sessions STRICTLY after `d`, up to and including
    `target`. trading_days_after(today, today) = 0. trading_days_after
    (today, next_trading_day) = 1. This is the right semantic for MC
    schedule indexing and "k sessions away from now" filters.

    Returns negative if target < d (count of trading days strictly before d
    down to and including target).
    """
    d = _to_date(d)
    target = _to_date(target)
    if target == d:
        return 0
    if target > d:
        # Trading days in (d, target] = trading days in [d, target]
        # minus 1 if d itself is a trading day.
        n = trading_days_between(d, target)
        if is_trading_day(d):
            n -= 1
        return n
    # target < d: trading days in [target, d) = inclusive count minus 1
    # if d is a trading day. Returned as negative.
    n = trading_days_between(target, d)
    if is_trading_day(d):
        n -= 1
    return -n


# Holiday-name lookup for the run banner. pandas_market_calendars exposes
# this via the schedule's `holidays()` regular-holiday rules; we cache a
# year-keyed dict for cheap lookups.
@lru_cache(maxsize=32)
def _holiday_map_for_year(year: int) -> dict:
    """Map of date → holiday name for the given year (NYSE-specific)."""
    try:
        # On exchange_calendars-backed NYSE, the named rules live on
        # `.regular_holidays`. Iterate rules → resolve each rule's dates
        # within the year → map date → friendly name.
        cal = _NYSE.regular_holidays
        out = {}
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        for rule in cal.rules:
            try:
                for ts in rule.dates(start, end):
                    out[ts.date()] = rule.name
            except Exception:
                continue
        # Ad-hoc closures (Carter funeral, 9/11, Sandy, etc.) live on
        # `.adhoc_holidays` as a flat list of Timestamps. They lack
        # individual names — label generically.
        try:
            for ts in getattr(_NYSE, "adhoc_holidays", []) or []:
                d = pd.Timestamp(ts).date()
                if start.date() <= d <= end.date() and d not in out:
                    out[d] = "Special closure"
        except Exception:
            pass
        return out
    except Exception:
        return {}


def holiday_name(d) -> Optional[str]:
    """Return the holiday name for `d` if it's a known NYSE holiday, else None.

    Weekends return None (they're not 'holidays', just non-trading). For an
    ad-hoc closure (e.g. Carter funeral) the library knows it's closed but
    may not name it — returns 'NYSE closure' as a fallback label.
    """
    d = _to_date(d)
    if d.weekday() >= 5:
        return None  # weekend, not a holiday
    name = _holiday_map_for_year(d.year).get(d)
    if name:
        return name
    # Non-trading weekday with no rule name = ad-hoc closure.
    if not is_trading_day(d):
        return "NYSE closure"
    return None


def early_close_time(d) -> Optional[time]:
    """Return the early-close time (ET, as a `time` object) for `d` if it's
    a half-day session, else None.

    Half-days at NYSE close 1pm ET (e.g. day after Thanksgiving, Christmas
    Eve, July 3 when adjacent to July 4). Engine cares because FMP's 'last
    trade' on these days isn't a 4pm close — downstream code must not
    assume 4pm semantics.
    """
    d = _to_date(d)
    if not is_trading_day(d):
        return None
    try:
        sched = _NYSE.schedule(start_date=d.isoformat(), end_date=d.isoformat())
        if sched.empty:
            return None
        close_utc = sched.iloc[0]["market_close"]
        # Convert UTC → ET. Regular close = 21:00 UTC during DST, 21:00 UTC
        # standard-time = 16:00 ET; half-day = 18:00 UTC = 13:00 ET.
        close_et = close_utc.tz_convert("America/New_York").time()
        regular = time(16, 0)
        if close_et < regular:
            return close_et
        return None
    except Exception:
        return None


def verify_market_state_via_fmp(api_key: Optional[str] = None) -> Optional[dict]:
    """Cross-check the library against FMP's runtime market-status endpoint.

    Returns dict with keys {'library_open', 'fmp_open', 'agree', 'fmp_raw'}
    or None if the call fails (network error, no api_key, etc).

    Caller (orchestrator) should fail loud on `agree == False` — that's
    either library staleness (ad-hoc closure not yet ingested) or an FMP
    bug. Don't override silently.
    """
    if not api_key:
        return None
    try:
        import requests
        url = f"https://financialmodelingprep.com/api/v3/is-the-market-open?apikey={api_key}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    fmp_open = bool(data.get("isTheStockMarketOpen", False))
    lib_open = is_trading_day(datetime.now().date())
    # FMP reports `isTheStockMarketOpen` for the CURRENT MOMENT — false
    # outside session hours even on a trading day. To compare apples to
    # apples, we use FMP's `stockMarketHours` if present, else fall back
    # to checking if the date itself is a trading day. Since FMP's payload
    # only distinguishes 'now', we treat agreement loosely: if FMP says
    # open NOW, library must say today is a trading day. If FMP says
    # closed NOW, the library might still legitimately say "today is a
    # trading day" (we're outside session hours). So the only hard
    # disagreement is FMP=open but library=closed → loud failure.
    agree = (not fmp_open) or (fmp_open and lib_open)
    return {
        "library_open": lib_open,
        "fmp_open_now": fmp_open,
        "agree": agree,
        "fmp_raw": data,
    }

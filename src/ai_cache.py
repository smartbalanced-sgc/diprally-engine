"""Same-day AI cache.

Sacred decision #11 (same-day CSV dedup) extends naturally to AI dedup:
intraday re-runs (debugging, checking after a spot move, rerunning after
a Pass 1 JSON parse failure) must not double-charge the $2/day broker.

Cache key: (ticker, date, spot). Cache hit when same ticker, same date,
and spot has moved < 1% since the cached spot. Otherwise re-run.

On disk: one JSON file per ticker-date at
    output/_ai_cache/{TICKER}_{YYYY-MM-DD}.json

A single file holds the full AI payload (Pass 1 raw, Pass 2 raw, stress
results, and their costs) so a cache hit replays the entire AI stack with
cost = $0.00.

Atomic writes: write to a .tmp file, fsync, rename. Crash-safe.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional


# Spot-move cache invalidation threshold sourced from config/diprally.yaml
# via src.config — sacred decision #17 (no hardcoded thresholds in src/).
from src.config import AI_CACHE_SPOT_MOVE_INVALIDATION_PCT as SPOT_MOVE_INVALIDATION_PCT  # noqa: E402


def _cache_dir() -> Path:
    """Cache lives next to output/. Mirrors engine.run_pipeline's path logic."""
    return Path(__file__).resolve().parent.parent / "output" / "_ai_cache"


def _cache_path(ticker: str, date_str: str) -> Path:
    return _cache_dir() / f"{ticker.upper()}_{date_str}.json"


def today_str() -> str:
    """Cache key date — uses the most recent TRADING day, not the wall-clock
    calendar day. PR #76 rationale: running on a market-holiday Monday with
    FMP returning Friday's quote would otherwise write a cache file dated
    Monday containing Friday's data — contaminating Tuesday's run if the
    cache invalidation skews. Keying on last_trading_day collapses Friday's
    original entry + weekend re-runs + Monday-holiday re-runs into the
    same key, with the spot-move guard handling actual price changes.
    """
    try:
        from src.market_calendar import last_trading_day
        return last_trading_day(datetime.now().date()).strftime("%Y-%m-%d")
    except Exception:
        # Defensive: if the calendar module is broken, fall back to
        # wall-clock date — better to over-write cache than crash.
        return datetime.now().strftime("%Y-%m-%d")


def get_cached(ticker: str, spot: float, date_str: Optional[str] = None) -> Optional[dict]:
    """Return the cached AI payload if it exists AND spot has moved < 1%
    since cache. Otherwise return None.

    Payload schema (when present):
      {
        "spot": float,
        "ticker": str,
        "date": str,
        "pass1_raw": dict | None,
        "pass1_cost": float,
        "pass1_sources": int,
        "pass2_raw": dict | None,
        "pass2_cost": float,
        "stress_results": list,
        "stress_cost": float,
        "models_used": {pass1: str, pass2: str, stress: str}
      }
    """
    date_str = date_str or today_str()
    path = _cache_path(ticker, date_str)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"   WARNING: ai_cache read failed for {ticker}/{date_str}: {e}")
        return None
    cached_spot = payload.get("spot")
    if cached_spot is None or cached_spot <= 0:
        return None
    spot_move = abs(spot - cached_spot) / cached_spot
    if spot_move >= SPOT_MOVE_INVALIDATION_PCT:
        print(f"   AI cache invalidated: spot moved {spot_move*100:.2f}% "
              f"(cached ${cached_spot:.2f}, now ${spot:.2f})")
        return None
    return payload


def save(ticker: str, spot: float, payload: dict,
         date_str: Optional[str] = None) -> Path:
    """Atomic write of payload to disk. Returns the cache path."""
    date_str = date_str or today_str()
    path = _cache_path(ticker, date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "spot": float(spot), "ticker": ticker.upper(),
               "date": date_str, "cached_at": datetime.now().isoformat()}
    # Atomic write: tmp file in same dir, fsync, rename
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path

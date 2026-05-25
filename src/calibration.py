"""W10 calibration harness — outcome resolution (PR #47).

Pure-function resolver that walks each ticker's prediction history and
fills in resolved-outcome columns when the horizon window has played
out. No new network calls — operates on the price history already
fetched by engine.run_pipeline (history_df with Date + Close columns).

The W10 harness ships first; the analysis layer (Brier scores,
per-signal calibration, auto-tuning the saturated caps from D-W2-17,
D-W2-18, D-W10-2) ships once N ≥ 30 trading days of resolved rows
have accumulated. Every day without the harness is permanent data
loss — predictions that complete their horizon window without their
outcomes being recorded can't be reconstructed later.

Sacred design contracts:
  - Resolution is ORDERED. Round-trip completes only when dip is
    touched FIRST, then rally is touched AFTER the dip touch. A
    rally-then-dip path is NOT a round-trip — it's a missed-entry
    scenario (sacred decision #6: trader didn't get filled at dip).
  - bag_hold means dip was touched but rally never came AND terminal
    close < dip price (trader bought the dip and is underwater at
    horizon). dip + rally-touched = round-trip, NOT bag-hold.
  - OPEN  → row is younger than horizon_days; nothing to resolve yet.
  - RESOLVED → row has been past horizon for ≥1 day; outcomes locked.
  - EXPIRED → reserved for future use (e.g. ticker delisted mid-window).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# Status constants (also used in CSV — keep stable).
STATUS_OPEN = "OPEN"
STATUS_RESOLVED = "RESOLVED"
STATUS_EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class ResolvedOutcome:
    """One row's resolution result. Used by the engine to overwrite
    the CSV row's outcome_* fields in place."""
    status: str                          # STATUS_OPEN / _RESOLVED / _EXPIRED
    dip_touched: Optional[bool]
    dip_touch_day: Optional[int]
    rally_touched_after_dip: Optional[bool]
    rally_touch_day: Optional[int]
    round_trip_completed: Optional[bool]
    bag_hold_realized: Optional[bool]
    realized_max_drawdown: Optional[float]
    realized_terminal_return: Optional[float]
    resolved_at: Optional[str]


def _open_outcome() -> ResolvedOutcome:
    """Row still inside its horizon window — no resolution yet."""
    return ResolvedOutcome(
        status=STATUS_OPEN,
        dip_touched=None, dip_touch_day=None,
        rally_touched_after_dip=None, rally_touch_day=None,
        round_trip_completed=None, bag_hold_realized=None,
        realized_max_drawdown=None, realized_terminal_return=None,
        resolved_at=None,
    )


def resolve_one_row(row: dict, history_df, today=None) -> ResolvedOutcome:
    """Compute the outcome for one prediction row given the post-prediction
    price history.

    Args:
        row: dict with at minimum 'date' (YYYY-MM-DD), 'spot',
             'recommended_dip', 'recommended_rally', 'horizon_days'.
        history_df: pandas DataFrame with 'Date' and 'Close' columns
                    covering at least [row.date+1 .. min(today, row.date+horizon)].
                    Already fetched by engine.run_pipeline.
        today: datetime.date for the resolution reference. Defaults to
               today's calendar date.

    Returns ResolvedOutcome. Returns OPEN status when the window hasn't
    closed yet OR when history_df lacks coverage of the window.
    """
    if today is None:
        today = datetime.now().date()

    try:
        row_date = datetime.strptime(str(row["date"])[:10], "%Y-%m-%d").date()
        spot = float(row["spot"])
        dip_target = float(row.get("recommended_dip") or 0.0)
        rally_target = float(row.get("recommended_rally") or 0.0)
        horizon_days = int(row.get("horizon_days") or 60)
    except (KeyError, ValueError, TypeError):
        return _open_outcome()  # malformed row — leave alone

    # No actionable prediction (math-only run with no qualifying pair).
    if dip_target <= 0 or rally_target <= 0:
        return _open_outcome()

    # PR #76: gate on TRADING days elapsed (was calendar). today must
    # be ≥ horizon_days trading days past row_date OR the history slice
    # below won't have enough bars; this also catches the case where a
    # synthetic / over-supplied history would otherwise let resolution
    # leak through before the window has actually closed in wall time.
    try:
        from src.market_calendar import trading_days_after
        td_elapsed = trading_days_after(row_date, today)
    except Exception:
        # Defensive fallback: calendar / 7 × 5 approximation if calendar
        # module unavailable. Coarse but never silently wrong.
        td_elapsed = (today - row_date).days * 5 // 7
    if td_elapsed < horizon_days:
        return _open_outcome()

    # Window has closed. Slice history to the post-prediction window.
    try:
        import pandas as pd
    except ImportError:
        return _open_outcome()

    if history_df is None or history_df.empty:
        return _open_outcome()
    if "Date" not in history_df.columns or "Close" not in history_df.columns:
        return _open_outcome()

    # Filter to bars strictly AFTER the prediction date (the prediction's
    # spot is the bar AT row_date; subsequent path is what matters for the
    # round-trip).
    cutoff = pd.Timestamp(row_date)
    mask = history_df["Date"] > cutoff
    window = history_df.loc[mask].copy().reset_index(drop=True)

    # PR #76: horizon_days is TRADING days. The previous code triggered
    # resolution after `horizon_days` CALENDAR days had elapsed (~43 of
    # 60 trading bars), locking outcomes ~28% early and biasing realized
    # dip/rally rates pessimistically. Now: only resolve when we actually
    # have `horizon_days` trading bars of post-prediction price history.
    if len(window) < horizon_days:
        return _open_outcome()

    # Use exactly horizon_days bars (consistent with MC's simulated window).
    window = window.head(horizon_days)

    closes = window["Close"].values
    n = len(closes)

    # Walk forward checking for dip then rally (sacred ordering).
    dip_touch_day = None
    rally_touch_day = None
    for i, c in enumerate(closes):
        if dip_touch_day is None and c <= dip_target:
            dip_touch_day = i
        elif dip_touch_day is not None and c >= rally_target:
            rally_touch_day = i
            break

    dip_touched = dip_touch_day is not None
    rally_touched_after_dip = rally_touch_day is not None
    round_trip_completed = dip_touched and rally_touched_after_dip

    # bag_hold: dip touched, rally NEVER reached after dip, terminal
    # close still below dip (trader is underwater at horizon).
    terminal_close = float(closes[-1]) if n > 0 else spot
    bag_hold_realized = (
        dip_touched
        and not rally_touched_after_dip
        and terminal_close < dip_target
    )

    realized_max_drawdown = float((spot - closes.min()) / spot) if n > 0 else 0.0
    realized_terminal_return = float(terminal_close / spot - 1.0)

    return ResolvedOutcome(
        status=STATUS_RESOLVED,
        dip_touched=dip_touched,
        dip_touch_day=dip_touch_day,
        rally_touched_after_dip=rally_touched_after_dip,
        rally_touch_day=rally_touch_day,
        round_trip_completed=round_trip_completed,
        bag_hold_realized=bag_hold_realized,
        realized_max_drawdown=realized_max_drawdown,
        realized_terminal_return=realized_terminal_return,
        resolved_at=today.strftime("%Y-%m-%d"),
    )


def apply_outcome_to_row(row: dict, outcome: ResolvedOutcome) -> dict:
    """Merge a ResolvedOutcome into a CSV row dict. Returns a NEW dict
    (does not mutate input). OPEN status leaves outcome fields blank;
    RESOLVED writes all fields with their string representations.

    Boolean fields are serialized as '1'/'0' to round-trip cleanly
    through csv.DictWriter.
    """
    out = dict(row)

    if outcome.status == STATUS_OPEN:
        # Idempotent for re-runs — don't overwrite an already-resolved row.
        existing = str(out.get("outcome_status") or "").strip()
        if existing in (STATUS_RESOLVED, STATUS_EXPIRED):
            return out
        out["outcome_status"] = STATUS_OPEN
        for col in ("dip_touched", "dip_touch_day", "rally_touched_after_dip",
                     "rally_touch_day", "round_trip_completed", "bag_hold_realized",
                     "realized_max_drawdown", "realized_terminal_return",
                     "resolved_at"):
            out.setdefault(col, "")
        return out

    out["outcome_status"] = outcome.status
    out["dip_touched"] = "1" if outcome.dip_touched else "0"
    out["dip_touch_day"] = (
        str(outcome.dip_touch_day) if outcome.dip_touch_day is not None else ""
    )
    out["rally_touched_after_dip"] = "1" if outcome.rally_touched_after_dip else "0"
    out["rally_touch_day"] = (
        str(outcome.rally_touch_day) if outcome.rally_touch_day is not None else ""
    )
    out["round_trip_completed"] = "1" if outcome.round_trip_completed else "0"
    out["bag_hold_realized"] = "1" if outcome.bag_hold_realized else "0"
    out["realized_max_drawdown"] = (
        f"{outcome.realized_max_drawdown:.4f}"
        if outcome.realized_max_drawdown is not None else ""
    )
    out["realized_terminal_return"] = (
        f"{outcome.realized_terminal_return:.4f}"
        if outcome.realized_terminal_return is not None else ""
    )
    out["resolved_at"] = outcome.resolved_at or ""
    return out


def resolve_history(rows: list, history_df, today=None) -> tuple[list, int]:
    """Resolve all rows in a history list against the given price history.
    Returns (updated_rows, n_newly_resolved). Idempotent — rows already
    flagged RESOLVED / EXPIRED pass through unchanged.
    """
    out_rows = []
    n_newly_resolved = 0
    for row in rows:
        existing_status = str(row.get("outcome_status") or "").strip()
        if existing_status in (STATUS_RESOLVED, STATUS_EXPIRED):
            out_rows.append(dict(row))
            continue
        outcome = resolve_one_row(row, history_df, today=today)
        merged = apply_outcome_to_row(row, outcome)
        if outcome.status == STATUS_RESOLVED:
            n_newly_resolved += 1
        out_rows.append(merged)
    return out_rows, n_newly_resolved

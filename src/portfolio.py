"""Portfolio-level correlation gate (W8 PR #49).

Sacred #6 compliant: this module does NOT size positions or allocate
capital. It refuses SUBSTITUTE recommendations — when two tickers in
the orchestrator's accepted set have 60d return-correlation above the
threshold (default 0.85), the trader doesn't have two independent
ideas, they have one idea expressed twice. The gate drops the lower-
EV ticker, surfacing only the highest-conviction representative of
each correlated cluster.

Institutional reasoning:
  - Risk decomposition: two tickers at ρ=0.92 share ~85% of return
    variance. A "diversified" book of 5 names at ρ=0.85+ behaves
    statistically like 1.5 names.
  - Idea generation: a daily scan that surfaces "BUY INTC + BUY AMD"
    is presenting 2 macro-correlated semis as separate ideas. The
    underlying thesis is one.
  - Trader workflow: the trader manually sizes externally (sacred #6);
    the engine's job is to surface the IDEAS, not the redundancies.

Algorithm (greedy):
  1. Sort accepted recommendations by EV bps descending.
  2. For each ticker in priority order, check 60d return-correlation
     against every already-accepted ticker.
  3. If any ρ exceeds threshold → DROP with reason citing the
     correlated ticker and ρ value.
  4. Otherwise → accept.

This is deterministic and rank-preserving: the highest-EV name in
any correlated cluster always survives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.config import PORTFOLIO_GATE


@dataclass(frozen=True)
class PortfolioRecommendation:
    """One ticker entering the gate. ev_bps is the sort key (priority).
    history_df is the ticker's daily price DataFrame — must contain
    'Date' and 'Close' columns. The gate computes daily returns
    internally and slices the last correlation_window_days."""
    ticker: str
    ev_bps: float
    history_df: object  # pandas DataFrame; left generic to avoid import


@dataclass
class GateResult:
    """Output of gate_by_correlation."""
    accepted: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)  # ticker → reason


def _daily_returns_last_n(history_df, n: int) -> Optional[np.ndarray]:
    """Compute the last N daily log returns from a price history.
    Returns None when there isn't enough data."""
    if history_df is None:
        return None
    if "Close" not in history_df.columns:
        return None
    closes = history_df["Close"].values
    if len(closes) < n + 1:
        return None
    closes_window = closes[-(n + 1):]
    rets = np.log(closes_window[1:] / closes_window[:-1])
    return rets


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equal-length arrays."""
    if a.shape != b.shape:
        # Truncate to shorter length — defensive, shouldn't happen
        # when both come from the same window size.
        n = min(len(a), len(b))
        a = a[-n:]
        b = b[-n:]
    if len(a) < 2:
        return 0.0
    # numpy returns a 2x2 matrix; we want the [0, 1] off-diagonal.
    c = np.corrcoef(a, b)
    if not np.isfinite(c[0, 1]):
        return 0.0
    return float(c[0, 1])


def gate_by_correlation(
    recommendations: list[PortfolioRecommendation],
    threshold: Optional[float] = None,
    window_days: Optional[int] = None,
    enabled: Optional[bool] = None,
) -> GateResult:
    """Apply the portfolio correlation gate. Defaults read from
    PORTFOLIO_GATE config. Pass explicit values to override (e.g. for
    backtests against pre-W8 behavior, pass enabled=False).

    Returns GateResult with accepted ticker list (priority order) and
    dropped dict (ticker → human-readable reason).
    """
    if enabled is None:
        enabled = PORTFOLIO_GATE.enabled
    if threshold is None:
        threshold = PORTFOLIO_GATE.correlation_threshold
    if window_days is None:
        window_days = PORTFOLIO_GATE.correlation_window_days

    result = GateResult()

    if not recommendations:
        return result

    if not enabled:
        # Pass-through — accept all in input order.
        result.accepted = [r.ticker for r in recommendations]
        return result

    # Sort by EV descending. Tie-break by ticker alphabetical for determinism.
    ranked = sorted(recommendations, key=lambda r: (-r.ev_bps, r.ticker))

    # Pre-compute return windows for everyone — None if insufficient data.
    return_windows = {
        r.ticker: _daily_returns_last_n(r.history_df, window_days)
        for r in ranked
    }

    for r in ranked:
        rets_r = return_windows[r.ticker]
        if rets_r is None:
            # Insufficient history → accept defensively (can't prove
            # correlation, give benefit of the doubt to the math layer
            # that already produced the recommendation).
            result.accepted.append(r.ticker)
            continue

        drop_reason = None
        for accepted_ticker in result.accepted:
            rets_a = return_windows[accepted_ticker]
            if rets_a is None:
                continue
            rho = _pearson(rets_r, rets_a)
            if rho >= threshold:
                drop_reason = (
                    f"correlated with already-accepted {accepted_ticker} "
                    f"(ρ={rho:.2f} ≥ {threshold:.2f} over last "
                    f"{window_days}d) — substitute idea, not a new one"
                )
                break

        if drop_reason is None:
            result.accepted.append(r.ticker)
        else:
            result.dropped[r.ticker] = drop_reason

    return result


def format_gate_result(gate: GateResult,
                       recommendations: list[PortfolioRecommendation]) -> str:
    """Human-readable gate summary for the orchestrator log + aggregate
    dashboard."""
    by_ticker = {r.ticker: r for r in recommendations}
    lines = []
    lines.append("=" * 78)
    lines.append(
        f"PORTFOLIO CORRELATION GATE — "
        f"{len(gate.accepted)} accepted / {len(gate.dropped)} dropped"
    )
    lines.append("=" * 78)
    if gate.accepted:
        lines.append("Accepted (priority order):")
        for t in gate.accepted:
            ev = by_ticker[t].ev_bps if t in by_ticker else 0.0
            lines.append(f"  {t:<8}  EV {ev:+.1f}bps")
    if gate.dropped:
        lines.append("")
        lines.append("Dropped (substitute ideas):")
        for t, reason in gate.dropped.items():
            lines.append(f"  {t:<8}  {reason}")
    return "\n".join(lines)

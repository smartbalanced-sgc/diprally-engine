"""AI accuracy backtest harness — does the AI's drift prediction earn its
$0.22/ticker keep, by category and in aggregate, on real historical data?

DIAGNOSIS BACKGROUND (2026-05-31, prompted by Jesse's MU substance audit):
The engine currently collapses 10+ AI-surfaced catalysts into ONE float
(drift_estimate) that enters the Bayesian blender at ~22% effective
weight. The catalogue's qualitative depth is invisible after that
compression. Before designing differentiated AI signal slots, we need
empirical answers to two questions:

  Q1. Is the AI's drift prediction DIRECTIONALLY correct more often
      than chance, on real historical events for the ticker?
  Q2. Is the AI's magnitude calibrated — does it overshoot, undershoot,
      or randomly scatter relative to realized 20d returns?

Mechanism: for each historical AS-OF date X in a configurable window:
  1. Filter the FMP bundle data (pt_revisions, grade changes, news) to
     entries dated ≤ X — what the AI would have seen on day X.
  2. Truncate price history to ≤ X and recompute spot/σ/RSI/mom_30d/YTD
     as they were on X.
  3. Build the facts bundle from the truncated/filtered inputs.
  4. Call AI Pass 1 + Pass 2 with web_search DISABLED (web_search would
     surface post-X news, contaminating the backtest — documented
     limitation: this measures bundle-only AI reasoning, not the marginal
     value of live web augmentation).
  5. Record AI's drift_estimate, catalysts, alignment signal.
  6. Compute realized 20-trading-day return from X+1 to X+20 (need at
     least 20 trading days of forward price history past X).
  7. Score: directional hit (sign match), magnitude error
     (annualized predicted vs realized), correlation across events.

Output: one CSV row per (ticker, as_of_date) event, plus console
summary with hit rate, RMSE, Spearman ρ, magnitude bias.

Cost discipline:
  - Web search disabled (~$0.04/Pass 1 instead of $0.08 with search).
  - AI calls cached on (ticker, as_of_date) — re-runs free.
  - --max-cost flag enforces budget; harness refuses to start when
    projected spend exceeds it.

Usage:
  python tools/diag/ai_accuracy_backtest.py --ticker MU \
      --as-of-dates 2026-04-04,2026-04-11,2026-04-18,2026-04-25,2026-05-02 \
      --output-csv output/ai_backtest_MU.csv \
      --max-cost 3.0

If --as-of-dates is omitted, the harness picks Fridays in the last 60
trading days that have at least 20 trading days of forward history. Use
--list-candidate-dates to dry-run the date selection.

End-to-end requires FMP_API_KEY and ANTHROPIC_API_KEY in env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd  # noqa: F401  (used by truncation)


_DEFAULT_LOOKBACK_TD = 60     # trading days to consider as candidate as-of dates
_DEFAULT_FORWARD_TD = 20      # ground-truth horizon (matches engine HORIZON)
_MIN_FORWARD_PRICES = 20      # need at least 20 forward trading days for scoring


# =============================================================================
# As-of filtering — what the AI would have seen on day X
# =============================================================================

def truncate_history_to(history_df, as_of_date: date):
    """Return df sliced to dates ≤ as_of_date. Caller is responsible for
    ensuring history_df has a 'date' column (string YYYY-MM-DD or datetime).
    Pure — no mutation."""
    if "date" not in history_df.columns:
        raise ValueError("history_df missing 'date' column")
    # Normalize to date objects for comparison
    dt_series = pd.to_datetime(history_df["date"]).dt.date
    mask = dt_series <= as_of_date
    return history_df[mask].reset_index(drop=True)


def forward_prices_from(history_df, as_of_date: date, n_trading_days: int):
    """Return forward-price slice for ground-truth: prices on the n trading
    days STRICTLY after as_of_date. Caller ensures history_df is sorted
    ascending by date. Returns at most n_trading_days rows; fewer iff
    history runs out before n full days."""
    dt_series = pd.to_datetime(history_df["date"]).dt.date
    mask = dt_series > as_of_date
    forward = history_df[mask].reset_index(drop=True)
    return forward.iloc[:n_trading_days]


def filter_bundle_lists_to_as_of(bundle: dict, as_of_date: date) -> dict:
    """Return a NEW bundle with pt_revisions / grade_changes / recent_news
    pruned to entries dated ≤ as_of_date. Date-relative fields
    (next_earnings_in_horizon, td_to_earnings) are NOT recomputed here —
    caller handles those via the snapshot.

    Pure — input bundle unchanged. Lists pruned in place on the copy.
    """
    import copy
    out = copy.deepcopy(bundle)
    out["as_of"] = as_of_date.isoformat()

    def _keep_if_dated_le(row, default_keep=False):
        d = row.get("date") or ""
        if not d:
            return default_keep
        try:
            row_dt = datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return default_keep
        return row_dt <= as_of_date

    for key in ("pt_revisions_90d", "grade_changes_90d", "recent_news_30d"):
        original = out.get(key)
        if isinstance(original, list):
            out[key] = [r for r in original if isinstance(r, dict)
                        and _keep_if_dated_le(r)]
    # Re-derive list-count fields that depend on filtered lists.
    if "pt_revisions_90d" in out:
        rev = out["pt_revisions_90d"]
        out["pt_revisions_90d_count"] = len(rev)
        if rev:
            raises = sum(1 for r in rev if r.get("action") == "raise")
            cuts = sum(1 for r in rev if r.get("action") == "cut")
            out["pt_revisions_90d_raise_cut_ratio"] = f"{raises} raises / {cuts} cuts"
        else:
            out.pop("pt_revisions_90d_raise_cut_ratio", None)
    if "grade_changes_90d" in out:
        gc = out["grade_changes_90d"]
        if gc:
            out["grade_actions_summary"] = {
                "upgrade": sum(1 for x in gc if (x.get("action") or "").lower() == "upgrade"),
                "downgrade": sum(1 for x in gc if (x.get("action") or "").lower() == "downgrade"),
                "maintain": sum(1 for x in gc if (x.get("action") or "").lower() == "maintain"),
                "initiate": sum(1 for x in gc if "init" in (x.get("action") or "").lower()),
            }
        else:
            out.pop("grade_actions_summary", None)
    if "recent_news_30d" in out:
        out["recent_news_30d_count"] = len(out["recent_news_30d"])

    return out


# =============================================================================
# As-of state — recompute spot/σ/RSI/mom from truncated prices
# =============================================================================

def compute_as_of_state(history_df, as_of_date: date) -> dict:
    """From a price history truncated to ≤ as_of_date, derive the state
    fields the AI prompt and facts bundle need:
        spot, sigma_blended (realized 30d annualized as proxy),
        rsi_14, mom_5d, mom_30d, ytd_return.
    This is a simplified state — the live engine's σ comes from a
    5-anchor triangulation (GARCH + 30/60/90 realized + IV), which we
    can't reconstruct retrospectively without options chains as-of-X.
    Backtest uses realized-30d as proxy, which is conservative (no
    GARCH spike-bias). Documented limitation."""
    from src.math_utils import compute_rsi_14, compute_realized_vol

    truncated = truncate_history_to(history_df, as_of_date)
    if len(truncated) < 35:
        raise ValueError(
            f"Insufficient history before {as_of_date}: only "
            f"{len(truncated)} trading days. Need ≥35 (30d realized vol "
            f"+ 5d RSI lookback)."
        )

    closes = truncated["close"].values.astype(float)
    spot = float(closes[-1])
    closes_series = pd.Series(closes)
    rsi = float(compute_rsi_14(closes_series))
    # compute_realized_vol takes log-returns Series; convert from prices.
    log_returns = pd.Series(np.log(closes_series / closes_series.shift(1)))
    realized = compute_realized_vol(log_returns, windows=[30])
    sigma_blended = realized.get(30)
    if sigma_blended is None or not np.isfinite(sigma_blended):
        sigma_blended = float(np.std(np.diff(np.log(closes[-30:]))) * np.sqrt(252))
    mom_5d = (closes[-1] / closes[-6] - 1.0) if len(closes) > 6 else 0.0
    mom_30d = (closes[-1] / closes[-31] - 1.0) if len(closes) > 31 else 0.0

    # YTD: anchor to first close of as_of_date.year
    year_anchor = None
    for i in range(len(truncated)):
        dt = pd.to_datetime(truncated.iloc[i]["date"]).date()
        if dt.year == as_of_date.year:
            year_anchor = float(truncated.iloc[i]["close"])
            break
    ytd_return = (
        (spot / year_anchor - 1.0) if year_anchor and year_anchor > 0 else 0.0
    )

    return {
        "spot": spot,
        "sigma_blended": sigma_blended,
        "rsi": rsi,
        "mom_5d": mom_5d,
        "mom_30d": mom_30d,
        "ytd_return": ytd_return,
        "as_of": as_of_date.isoformat(),
    }


# =============================================================================
# Scoring — predicted drift vs realized 20-trading-day return
# =============================================================================

def score_prediction(*, predicted_drift_annual: float,
                       spot_at_as_of: float, forward_df) -> dict:
    """Compare AI's drift forecast to realized 20-trading-day price action.

    predicted_drift_annual: AI's drift_estimate (already annualized).
    spot_at_as_of: closing price ON as_of_date.
    forward_df: 1..N rows of subsequent trading days (close column).

    Returns dict with:
      n_forward          - actual forward trading days available
      terminal_return    - (last_close / spot) - 1
      realized_drift_ann - terminal_return × (252 / n_forward)
      direction_match    - sign(predicted) == sign(realized) (1/0); None when
                            either is exactly zero (ambiguous)
      magnitude_error    - |predicted - realized|
      overshoot_ratio    - predicted / realized when realized != 0; None when
                            realized = 0 (denominator)
      max_drawdown_pct   - worst (low - spot)/spot across forward window
      max_runup_pct      - best (high - spot)/spot across forward window
      hit_dip_5pct       - did forward minimum touch spot * 0.95? (1/0)
      hit_rally_5pct     - did forward maximum touch spot * 1.05? (1/0)
    """
    if forward_df is None or len(forward_df) < 1:
        return {
            "n_forward": 0,
            "terminal_return": None,
            "realized_drift_ann": None,
            "direction_match": None,
            "magnitude_error": None,
            "overshoot_ratio": None,
            "max_drawdown_pct": None,
            "max_runup_pct": None,
            "hit_dip_5pct": None,
            "hit_rally_5pct": None,
        }
    n = len(forward_df)
    closes = forward_df["close"].values.astype(float)
    # Use high/low when available; else fall back to close.
    highs = (
        forward_df["high"].values.astype(float)
        if "high" in forward_df.columns else closes
    )
    lows = (
        forward_df["low"].values.astype(float)
        if "low" in forward_df.columns else closes
    )

    terminal_return = float(closes[-1] / spot_at_as_of - 1.0)
    realized_drift_ann = terminal_return * (252.0 / n)
    direction_match: Optional[int] = None
    if abs(predicted_drift_annual) > 1e-9 and abs(realized_drift_ann) > 1e-9:
        direction_match = int(
            np.sign(predicted_drift_annual) == np.sign(realized_drift_ann)
        )
    magnitude_error = abs(predicted_drift_annual - realized_drift_ann)
    overshoot_ratio: Optional[float] = None
    if abs(realized_drift_ann) > 1e-6:
        overshoot_ratio = predicted_drift_annual / realized_drift_ann

    max_low = float(lows.min())
    max_high = float(highs.max())
    max_drawdown_pct = (max_low - spot_at_as_of) / spot_at_as_of
    max_runup_pct = (max_high - spot_at_as_of) / spot_at_as_of
    hit_dip_5 = int(max_low <= spot_at_as_of * 0.95)
    hit_rally_5 = int(max_high >= spot_at_as_of * 1.05)

    return {
        "n_forward": n,
        "terminal_return": terminal_return,
        "realized_drift_ann": realized_drift_ann,
        "direction_match": direction_match,
        "magnitude_error": magnitude_error,
        "overshoot_ratio": overshoot_ratio,
        "max_drawdown_pct": float(max_drawdown_pct),
        "max_runup_pct": float(max_runup_pct),
        "hit_dip_5pct": hit_dip_5,
        "hit_rally_5pct": hit_rally_5,
    }


# =============================================================================
# As-of candidate dates
# =============================================================================

def pick_default_as_of_dates(history_df, lookback_td: int = _DEFAULT_LOOKBACK_TD,
                              forward_td: int = _DEFAULT_FORWARD_TD,
                              spacing_td: int = 5) -> list[date]:
    """Pick weekly (every 5 trading days) candidate as-of dates in the
    last `lookback_td` trading days that have at least `forward_td`
    forward trading days available. Returns list[date] sorted ascending.
    """
    if "date" not in history_df.columns:
        raise ValueError("history_df missing 'date' column")
    dates = pd.to_datetime(history_df["date"]).dt.date.tolist()
    if len(dates) < lookback_td + forward_td + 35:
        # Not enough history at all
        return []
    candidates: list[date] = []
    # Last available trading day — we can't use this; need forward history.
    last_idx = len(dates) - 1
    # The most recent valid as-of has at least `forward_td` trading days after it.
    most_recent_idx = last_idx - forward_td
    # Earliest valid as-of is bounded by lookback_td going back from most_recent.
    earliest_idx = max(35, most_recent_idx - lookback_td)
    idx = most_recent_idx
    while idx >= earliest_idx:
        candidates.append(dates[idx])
        idx -= spacing_td
    candidates.sort()
    return candidates


# =============================================================================
# Backtest event runner — one (ticker, as_of) call
# =============================================================================

def run_backtest_event(
    *, ticker: str, as_of_date: date, history_df,
    base_bundle: dict, peer_tickers: list,
    horizon_days: int = _DEFAULT_FORWARD_TD,
    pass1_model: str = "claude-haiku-4-5-20251001",  # cheaper for backtest scale
    pass2_model: str = "claude-sonnet-4-6",
    web_search_max_uses: int = 0,  # DISABLED for backtest — see module docstring
    cache_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> dict:
    """Replay a single (ticker, as_of_date) AI prediction. Returns a flat
    dict of fields suitable for CSV rows + accuracy scoring.

    Cache: when cache_dir provided, AI results saved to
    {cache_dir}/{ticker}_{as_of}.json and replayed on subsequent runs.
    Same-day re-runs are free.
    """
    from src.ai_layer import (
        build_ai_pass1_prompt, build_ai_pass2_prompt,
        call_ai_pass, parse_ai_pass1, parse_ai_pass2,
        compute_ai_cost,
    )
    from src.facts_bundle import bundle_to_prompt_block

    as_of_str = as_of_date.isoformat()
    cache_path = (cache_dir / f"{ticker}_{as_of_str}.json") if cache_dir else None
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return cached

    state = compute_as_of_state(history_df, as_of_date)
    spot = state["spot"]

    # As-of-filtered facts bundle.
    as_of_bundle = filter_bundle_lists_to_as_of(base_bundle, as_of_date)
    as_of_bundle.update({
        "spot": round(spot, 4),
        "sigma_blended_annual_pct": round(state["sigma_blended"] * 100, 2),
        "rsi_14": round(state["rsi"], 1),
        "mom_5d_pct": round(state["mom_5d"] * 100, 2),
        "mom_30d_pct": round(state["mom_30d"] * 100, 2),
        "ytd_return_pct": round(state["ytd_return"] * 100, 2),
    })
    bundle_block = bundle_to_prompt_block(as_of_bundle)

    # Minimal AiSnapshot stand-in (just the fields the prompt builder
    # reads). Bundles up the as-of state without engine-side baggage.
    class _Snap:
        pass
    snap = _Snap()
    snap.ticker = ticker
    snap.spot = spot
    snap.sector = base_bundle.get("sector", "Unknown")
    snap.industry = base_bundle.get("industry", "Unknown")
    snap.rsi = state["rsi"]
    snap.mom_5d = state["mom_5d"]
    snap.mom_30d = state["mom_30d"]
    snap.ytd_return = state["ytd_return"]
    snap.market_cap = base_bundle.get("market_cap_usd") or 0
    snap.beta = base_bundle.get("beta") or 1.0
    snap.timestamp = datetime.combine(as_of_date, datetime.min.time())

    class _VolP:
        pass
    vp = _VolP()
    vp.blended_sigma = state["sigma_blended"]
    vp.garch_sigma = state["sigma_blended"]
    vp.realized_vol = {30: state["sigma_blended"]}

    self_earnings_dt: Optional[datetime] = None  # not reconstructed retroactively

    # The prompt builders take rich engine objects. For the backtest we
    # call them with the minimal stand-ins above + an empty base-signal
    # summary string. AI Pass 1's bundle_block IS the substantive input.
    base_signal_summary = "(backtest mode — math-layer signal blend not reconstructed)"

    if dry_run:
        return {
            "ticker": ticker, "as_of_date": as_of_str,
            "spot_at_as_of": spot,
            "predicted_drift": None,
            "ai_cost": 0.0,
            "dry_run": True,
        }

    pass1_prompt = build_ai_pass1_prompt(
        ticker=ticker, snapshot=snap, vol_profile=vp,
        base_signal_summary=base_signal_summary,
        horizon_days=horizon_days,
        peer_tickers=peer_tickers,
        self_earnings_date=self_earnings_dt,
        facts_bundle_json=bundle_block,
    )
    p1_raw, p1_sources, p1_cost = call_ai_pass(
        prompt=pass1_prompt, max_tokens=3000, pass_label="Backtest Pass 1",
        model=pass1_model, web_search_max_uses=web_search_max_uses,
    )
    if not p1_raw:
        return {
            "ticker": ticker, "as_of_date": as_of_str,
            "spot_at_as_of": spot,
            "predicted_drift": None,
            "ai_cost": p1_cost or 0.0,
            "error": "Pass 1 returned no parseable JSON",
        }
    pass1 = parse_ai_pass1(p1_raw, p1_sources, p1_cost, today=as_of_date)

    pass2 = None
    p2_cost = 0.0
    # Pass 2 requires the math layer's marginal probability summary. In
    # backtest we don't reconstruct the full MC — we feed a placeholder
    # so Pass 2 can still critique Pass 1's catalysts. Pass 2 drift is
    # what we record; math-layer math agreement isn't backtested here.
    mc_marginal = {"p_up": "n/a", "p_down": "n/a", "bracket_pct_str": "10%"}
    sigma_tri = {"blended": state["sigma_blended"], "divergence": 0.0}
    pass2_prompt = build_ai_pass2_prompt(
        ticker=ticker, snapshot=snap, pass1=pass1,
        mc_marginal_summary=mc_marginal,
        sigma_triangulation_summary=sigma_tri,
        prior_posterior_drift=None,
        facts_bundle_json=bundle_block,
    )
    p2_raw, _, p2_cost = call_ai_pass(
        prompt=pass2_prompt, max_tokens=3000, pass_label="Backtest Pass 2",
        model=pass2_model, web_search_max_uses=0,
    )
    if p2_raw:
        pass2 = parse_ai_pass2(p2_raw, pass1, p2_cost, today=as_of_date)

    # Score against realized forward prices.
    forward = forward_prices_from(history_df, as_of_date, horizon_days)
    if len(forward) < _MIN_FORWARD_PRICES:
        forward_scored = score_prediction(
            predicted_drift_annual=0.0, spot_at_as_of=spot, forward_df=forward,
        )
        forward_scored["truncated_forward"] = True
    else:
        forward_scored = score_prediction(
            predicted_drift_annual=(pass2.drift_estimate if pass2 else pass1.drift_estimate),
            spot_at_as_of=spot, forward_df=forward,
        )
        forward_scored["truncated_forward"] = False

    row = {
        "ticker": ticker,
        "as_of_date": as_of_str,
        "spot_at_as_of": spot,
        "sigma_blended_at_as_of": state["sigma_blended"],
        "rsi_at_as_of": state["rsi"],
        "mom_30d_at_as_of": state["mom_30d"],
        "ai_drift_pass1": pass1.drift_estimate,
        "ai_drift_pass2": (pass2.drift_estimate if pass2 else None),
        "ai_predicted_drift": (pass2.drift_estimate if pass2 else pass1.drift_estimate),
        "ai_narrative_score": (pass2.narrative_score if pass2 else pass1.narrative_score),
        "ai_vol_regime": (pass2.vol_regime if pass2 else pass1.vol_regime),
        "ai_catalysts_count": len(pass2.catalysts if pass2 else pass1.catalysts),
        "ai_alignment_signal": (
            getattr(pass2, "verdict_alignment_signal", "") if pass2 else ""
        ),
        "ai_cost": (p1_cost or 0.0) + (p2_cost or 0.0),
        **forward_scored,
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(row, indent=2, default=str))
    return row


# =============================================================================
# Aggregate scoring across N events
# =============================================================================

def aggregate_results(rows: list[dict]) -> dict:
    """Roll N event rows into accuracy summary.

    Reports:
      n_events / n_directional        - count + count with both predicted
                                         and realized non-zero
      directional_hit_rate            - fraction of directional events where
                                         signs matched (45-55% = random)
      mean_predicted_drift            - mean AI drift forecast (annualized)
      mean_realized_drift             - mean realized drift (annualized)
      magnitude_bias                  - mean(predicted - realized);
                                         positive = AI overshoots up,
                                         negative = AI overshoots down
      rmse                            - sqrt of mean squared error
      pearson_corr                    - linear correlation predicted ↔ realized
      spearman_corr                   - rank correlation (robust to outliers)
      overshoot_median_ratio          - median predicted/realized (when
                                         realized non-zero); 1.0 = perfect
                                         calibration, >1 = overshoots,
                                         <1 = undershoots
    """
    if not rows:
        return {"n_events": 0}
    n_events = len(rows)
    valid = [r for r in rows if r.get("realized_drift_ann") is not None
             and r.get("ai_predicted_drift") is not None
             and not r.get("truncated_forward")]
    if not valid:
        return {"n_events": n_events, "n_scored": 0,
                "note": "no events with full 20d forward history"}

    predicted = np.array([r["ai_predicted_drift"] for r in valid], dtype=float)
    realized = np.array([r["realized_drift_ann"] for r in valid], dtype=float)

    directional_events = [
        r for r in valid if r.get("direction_match") is not None
    ]
    if directional_events:
        hit_rate = float(np.mean([r["direction_match"] for r in directional_events]))
    else:
        hit_rate = None

    magnitude_bias = float(np.mean(predicted - realized))
    rmse = float(np.sqrt(np.mean((predicted - realized) ** 2)))

    overshoot_ratios = [
        r["overshoot_ratio"] for r in valid
        if r.get("overshoot_ratio") is not None and np.isfinite(r["overshoot_ratio"])
    ]
    overshoot_median = float(np.median(overshoot_ratios)) if overshoot_ratios else None

    # Correlations — only meaningful with N≥5
    pearson = None
    spearman = None
    if len(predicted) >= 5:
        if np.std(predicted) > 1e-9 and np.std(realized) > 1e-9:
            pearson = float(np.corrcoef(predicted, realized)[0, 1])
            pr = pd.Series(predicted).rank()
            rr = pd.Series(realized).rank()
            spearman = float(np.corrcoef(pr.values, rr.values)[0, 1])

    return {
        "n_events": n_events,
        "n_scored": len(valid),
        "n_directional": len(directional_events),
        "directional_hit_rate": hit_rate,
        "mean_predicted_drift": float(np.mean(predicted)),
        "mean_realized_drift": float(np.mean(realized)),
        "magnitude_bias": magnitude_bias,
        "rmse": rmse,
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "overshoot_median_ratio": overshoot_median,
    }


def print_summary(agg: dict) -> None:
    """Pretty-print the aggregate accuracy block to stdout."""
    print()
    print("=" * 78)
    print("AI ACCURACY BACKTEST — AGGREGATE SUMMARY")
    print("=" * 78)
    if agg.get("n_events", 0) == 0:
        print("  No events scored.")
        return
    print(f"  Events scored: {agg.get('n_scored', 0)} / {agg['n_events']} attempted")
    if agg.get("n_scored", 0) == 0:
        print(f"  Note: {agg.get('note', '(none with full 20d forward history)')}")
        return
    hr = agg.get("directional_hit_rate")
    if hr is not None:
        n_dir = agg.get("n_directional", 0)
        print(f"  Directional hit rate: {hr*100:.1f}% over {n_dir} directional events")
        print(f"    (random baseline 50%; ≥58% on N≥20 is significant at p<0.10)")
    print(f"  Mean predicted drift (annualised): {agg.get('mean_predicted_drift', 0)*100:+.1f}%")
    print(f"  Mean realized drift (annualised):  {agg.get('mean_realized_drift', 0)*100:+.1f}%")
    print(f"  Magnitude bias (pred - real):       {agg.get('magnitude_bias', 0)*100:+.1f}pp")
    print(f"  RMSE:                                {agg.get('rmse', 0)*100:.1f}pp")
    if agg.get("pearson_corr") is not None:
        print(f"  Pearson ρ (linear):  {agg['pearson_corr']:+.3f}")
    if agg.get("spearman_corr") is not None:
        print(f"  Spearman ρ (rank):   {agg['spearman_corr']:+.3f}")
    osr = agg.get("overshoot_median_ratio")
    if osr is not None:
        verdict = (
            "OVERSHOOTS" if osr > 1.5 else
            "UNDERSHOOTS" if osr < 0.66 else
            "WELL-CALIBRATED"
        )
        print(f"  Overshoot median (pred/real):       {osr:+.2f}× → {verdict}")
    print("=" * 78)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="single ticker (e.g. MU)")
    parser.add_argument(
        "--as-of-dates", default="",
        help="comma-separated YYYY-MM-DD list; omit to auto-pick weekly "
             "dates in last 60 trading days with sufficient forward history",
    )
    parser.add_argument("--lookback-td", type=int, default=_DEFAULT_LOOKBACK_TD)
    parser.add_argument("--forward-td", type=int, default=_DEFAULT_FORWARD_TD)
    parser.add_argument("--spacing-td", type=int, default=5,
                        help="spacing between auto-picked as-of dates")
    parser.add_argument("--max-cost", type=float, default=5.0,
                        help="USD cap; refuses to start when projected spend exceeds")
    parser.add_argument(
        "--output-csv", default="output/ai_backtest_{ticker}.csv",
        help="CSV output path; {ticker} placeholder substituted",
    )
    parser.add_argument(
        "--cache-dir", default="output/ai_backtest_cache",
        help="cache dir; per-event AI results cached so re-runs are free",
    )
    parser.add_argument("--list-candidate-dates", action="store_true",
                        help="print picked as-of dates and exit (no AI cost)")
    parser.add_argument("--dry-run", action="store_true",
                        help="run pipeline but skip AI calls (cost = 0)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    api_key = os.getenv("FMP_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: FMP_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    # --- Fetch base data (today's view) — engine-style pre-fetch ---
    print(f"Loading today's full data for {ticker}...")
    from src.data_fetch import (
        fetch_history, fetch_pt_news, fetch_grades_history,
        fetch_recent_news, fetch_company_profile, fetch_analyst_targets,
        fetch_analyst_summary,
    )
    from src.facts_bundle import build_facts_bundle

    # Cache fetched data to avoid re-fetching across runs.
    history = fetch_history(ticker, api_key, lookback_days=400)
    profile = fetch_company_profile(ticker, api_key)
    analyst_targets = fetch_analyst_targets(ticker, api_key)
    analyst_summary = fetch_analyst_summary(ticker, api_key)
    pt_news = fetch_pt_news(ticker, api_key, lookback_days=120)
    grades = fetch_grades_history(ticker, api_key, lookback_days=120)
    recent_news = fetch_recent_news(ticker, api_key)

    # The "base bundle" — built with today's full data; we filter list
    # fields per as-of below.
    base_bundle = build_facts_bundle(
        ticker=ticker, spot=float(history.iloc[-1]["close"]),
        sigma_blended=0.5, sigma_class="UNKNOWN", rsi=50.0,
        mom_5d=0.0, mom_30d=0.0, ytd_return=0.0,
        horizon_days=args.forward_td, peer_tickers=[],
        profile=profile, analyst_targets=analyst_targets,
        analyst_summary=analyst_summary, pt_news=pt_news,
        grades_history=grades, recent_news=recent_news,
    )

    # --- Pick as-of dates ---
    if args.as_of_dates:
        as_of_dates = [
            datetime.strptime(s.strip(), "%Y-%m-%d").date()
            for s in args.as_of_dates.split(",") if s.strip()
        ]
    else:
        as_of_dates = pick_default_as_of_dates(
            history, lookback_td=args.lookback_td,
            forward_td=args.forward_td, spacing_td=args.spacing_td,
        )
    print(f"As-of dates to score: {len(as_of_dates)}")
    for d in as_of_dates:
        print(f"  {d.isoformat()}")
    if args.list_candidate_dates:
        return

    # Cost projection — Haiku Pass 1 + Sonnet Pass 2 ≈ $0.10 per event.
    PROJECTED_PER_EVENT = 0.10
    projected_total = len(as_of_dates) * PROJECTED_PER_EVENT
    print(f"Projected cost: {len(as_of_dates)} × ${PROJECTED_PER_EVENT} = ${projected_total:.2f}")
    if projected_total > args.max_cost:
        print(
            f"ERROR: projected cost ${projected_total:.2f} exceeds "
            f"--max-cost ${args.max_cost:.2f}. Lower --lookback-td or "
            f"raise --max-cost.", file=sys.stderr,
        )
        sys.exit(3)

    # --- Run events ---
    cache_dir = Path(args.cache_dir)
    rows = []
    total_cost = 0.0
    for i, d in enumerate(as_of_dates):
        print(f"\n[{i+1}/{len(as_of_dates)}] {ticker} @ {d.isoformat()}")
        try:
            row = run_backtest_event(
                ticker=ticker, as_of_date=d, history_df=history,
                base_bundle=base_bundle, peer_tickers=[],
                horizon_days=args.forward_td,
                cache_dir=cache_dir, dry_run=args.dry_run,
            )
            rows.append(row)
            total_cost += float(row.get("ai_cost") or 0.0)
            pred = row.get("ai_predicted_drift")
            real = row.get("realized_drift_ann")
            if pred is not None:
                print(f"  predicted {pred*100:+.1f}%/yr  realized "
                      f"{real*100:+.1f}%/yr" if real is not None
                      else f"  predicted {pred*100:+.1f}%/yr  realized n/a")
            time.sleep(0.5)  # rate-limit gentleness on AI provider
        except Exception as e:
            print(f"  ⚠ FAILED: {type(e).__name__}: {e}")
            rows.append({
                "ticker": ticker, "as_of_date": d.isoformat(),
                "error": f"{type(e).__name__}: {e}"[:300],
            })

    # --- Persist CSV ---
    out_path = Path(args.output_csv.replace("{ticker}", ticker))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    if rows:
        all_keys = sorted({k for r in rows for k in r.keys()})
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {len(rows)} rows → {out_path}")

    # --- Aggregate + print ---
    agg = aggregate_results(rows)
    print_summary(agg)
    print(f"\nTotal AI cost this run: ${total_cost:.2f}")


if __name__ == "__main__":
    main()

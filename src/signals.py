"""Drift signals + blend + Bayesian update + regime detection.

W0: 10 signals total. v1's 9 (less the AI synthesis variant we dropped as dead
code) plus v2's 2 AI-derived (catalyst_proximity, narrative) plus the unusual-
move Z and bull/bear factor arithmetic helpers.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import (
    ANALYST_EXTREME_DRIFT_THRESHOLD,
    BLEND_WEIGHTS,
    CATALYST_Z_THRESHOLD,
    CONFIDENCE_TO_SE,
    FACTOR_NET_THRESHOLD,
    FACTOR_TAIL_BIAS,
    FACTOR_WEIGHTS,
    NARRATIVE_DRIFT_ADJUSTMENT,
    SIGNAL_ANALYST,
    SIGNAL_CATALYST_PROXIMITY,
    SIGNAL_HISTORICAL,
    SIGNAL_MACRO_DRIFT_LEVELS,
    SIGNAL_PEER_RS,
    SIGNAL_REGIME_DETECTION,
    SIGNAL_FUNDAMENTALS,
    SIGNAL_REVISION_MOMENTUM,
    SIGNAL_SECTOR_DECOUPLING,
    SIGNAL_SECTOR_MOMENTUM_CAPS,
    SIGNAL_SHORT_INTEREST_BRACKETS,
)


# Confidence one-step downgrade ladder for outlier-suppression logic.
_CONF_DOWNGRADE = {"HIGH": "MEDIUM", "MEDIUM": "LOW", "LOW": "LOW"}


def _gate_extreme_drift(drift: float, conf: str, notes: str) -> tuple[str, str]:
    """Sanity gate for signal drifts. When |drift| exceeds the extreme-outlier
    threshold (analyst, etc.), step the confidence down one notch and append
    a verification flag to the notes. Returns (new_conf, new_notes).

    Surfaced by MOG-A FMP analyst signal returning -58.9% HIGH conf — possibly
    real, possibly stale/wrong-ticker data. Either way, no signal that extreme
    should drive a blend at full HIGH-conf weight without verification.
    """
    if abs(drift) > ANALYST_EXTREME_DRIFT_THRESHOLD:
        new_conf = _CONF_DOWNGRADE[conf]
        flag = (f" [EXTREME OUTLIER — |implied drift| > {ANALYST_EXTREME_DRIFT_THRESHOLD*100:.0f}%/yr; "
                f"conf downgraded {conf}→{new_conf}; manual verification recommended]")
        return new_conf, notes + flag
    return conf, notes


# =============================================================================
# Universal "no signal" sentinel
# =============================================================================

def _none_signal(reason):
    return {"drift": None, "confidence": "LOW",
            "source_quality": "NONE_FOUND", "sources_count": 0,
            "notes": reason}


# =============================================================================
# Base signals (v1)
# =============================================================================

def signal_from_analyst_targets(targets, S0, price_history_df=None,
                                  summary=None):
    """Analyst targets → drift signal. Prefers fresh price-target-summary.
    Thresholds in config signals.analyst (D-W2-6)."""
    cfg = SIGNAL_ANALYST
    if summary:
        target = None
        n_analysts = 0
        window = ""
        base_conf = "MEDIUM"

        if summary.get("last_month_count", 0) >= cfg.last_month_min_n_for_use and summary.get("last_month_avg"):
            target = float(summary["last_month_avg"])
            n_analysts = summary["last_month_count"]
            window = "last month"
            base_conf = "HIGH" if n_analysts >= cfg.last_month_high_conf_n else "MEDIUM"
        elif summary.get("last_quarter_count", 0) >= cfg.last_quarter_min_n_for_use and summary.get("last_quarter_avg"):
            target = float(summary["last_quarter_avg"])
            n_analysts = summary["last_quarter_count"]
            window = "last quarter"
            base_conf = "MEDIUM" if n_analysts >= cfg.last_quarter_medium_conf_n else "LOW"
        elif summary.get("last_year_avg"):
            target = float(summary["last_year_avg"])
            n_analysts = summary.get("last_year_count", 0)
            window = "last year"
            base_conf = "LOW"

        if target and target > 0 and S0 > 0:
            drift = (target / S0) - 1.0
            staleness_note = ""
            if window != "last month" and price_history_df is not None and len(price_history_df) >= 60:
                try:
                    p60 = float(price_history_df["Close"].iloc[-60])
                    move_60d = abs((S0 - p60) / p60)
                    if move_60d > cfg.staleness_move_60d:
                        base_conf = "LOW"
                        staleness_note = (f" (STALENESS: stock moved {move_60d*100:+.0f}% "
                                          f"in 60d, only {window} avg available)")
                except (ValueError, TypeError, IndexError):
                    pass
            base_notes = (f"{window} avg ${target:.0f} (n={n_analysts}), "
                          f"vs spot ${S0:.0f}, drift implied {drift*100:+.1f}%"
                          f"{staleness_note}")
            base_conf, base_notes = _gate_extreme_drift(drift, base_conf, base_notes)
            return {
                "drift": float(drift), "confidence": base_conf,
                "source_quality": "REPUTABLE", "sources_count": int(n_analysts),
                "notes": base_notes,
            }

    if not targets or not targets.get("target_mean") or S0 <= 0:
        return _none_signal("no analyst targets available")
    try:
        target = float(targets["target_mean"])
        if target <= 0:
            return _none_signal("invalid target price")
        drift = (target / S0) - 1.0
        high = float(targets.get("target_high") or target)
        low = float(targets.get("target_low") or target)
        spread = (high - low) / target if target > 0 else 1.0
        if spread < cfg.spread_high_conf:
            conf = "HIGH"
        elif spread < cfg.spread_medium_conf:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        staleness_note = " (stale-mixed consensus fallback)"
        if price_history_df is not None and len(price_history_df) >= 60:
            try:
                p60 = float(price_history_df["Close"].iloc[-60])
                move_60d = abs((S0 - p60) / p60)
                if move_60d > cfg.staleness_move_60d:
                    conf = "LOW"
                    staleness_note = (f" (STALE: stock moved {move_60d*100:+.0f}% "
                                      f"in 60d, consensus lags; no fresh summary either)")
            except (ValueError, TypeError, IndexError):
                pass

        notes = (f"consensus mean ${target:.0f}, range ${low:.0f}-${high:.0f}"
                 f"{staleness_note}")
        conf, notes = _gate_extreme_drift(drift, conf, notes)
        return {
            "drift": float(drift), "confidence": conf,
            "source_quality": "REPUTABLE", "sources_count": 5,
            "notes": notes,
        }
    except (ValueError, TypeError):
        return _none_signal("analyst target parse error")


def signal_from_sector(sector_perf, swing_regime=None):
    """Sector momentum → drift, regime-aware cap."""
    if not sector_perf or sector_perf.get("cum_return_pct") is None:
        return _none_signal("sector data unavailable")
    days = max(1, sector_perf.get("n_days", 30))
    cum = sector_perf["cum_return_pct"] / 100.0
    drift = (1 + cum) ** (252 / days) - 1.0

    # Regime-conditional caps from config (D-W2-6).
    caps = SIGNAL_SECTOR_MOMENTUM_CAPS
    regime_name = swing_regime.get("regime") if swing_regime else None
    if regime_name == "POST_PARABOLA":
        cap_high, cap_low = caps.post_parabola
        conf = "LOW"
        regime_note = f" [POST_PARABOLA regime: sector cap reduced to +{cap_high*100:.0f}%, conf LOW]"
    elif regime_name in ("MOMENTUM_BULL", "MOMENTUM_BEAR"):
        cap_high, cap_low = caps.momentum
        conf = "MEDIUM"
        regime_note = f" [{regime_name}: cap +{cap_high*100:.0f}%]"
    else:
        cap_high, cap_low = caps.default
        conf = "MEDIUM"
        regime_note = ""

    drift = max(cap_low, min(cap_high, drift))
    return {
        "drift": float(drift), "confidence": conf,
        "source_quality": "PRIMARY", "sources_count": 1,
        "notes": (f"{sector_perf.get('sector','?')} {cum*100:+.1f}% "
                  f"last {days}d (annualised {drift*100:+.0f}%){regime_note}"),
    }


def signal_from_macro(macro):
    """VIX + SPY → drift tilt. Levels in config signals.macro_drift_levels (D-W2-6)."""
    if not macro:
        return _none_signal("macro data unavailable")
    regime = macro.get("regime", "neutral")
    levels = SIGNAL_MACRO_DRIFT_LEVELS
    drift = levels.get(regime, levels.get("neutral", 0.05))
    return {
        "drift": float(drift), "confidence": "MEDIUM",
        "source_quality": "PRIMARY", "sources_count": 2,
        "notes": (f"VIX {macro['vix']:.1f}, SPY {macro['spy_trend']*100:+.1f}% "
                  f"vs MA50 -> {regime}"),
    }


def signal_from_historical(mu_capped, mu_raw, sigma):
    """Historical mean-return drift, cap-gated to LOW confidence when binding.
    Thresholds in config signals.historical (D-W2-6)."""
    if mu_capped is None:
        return _none_signal("historical drift unavailable")
    if abs(mu_raw) > SIGNAL_HISTORICAL.cap_binding_abs_drift:
        conf = "LOW"
        gate_note = " (CAP BINDING — extrapolation risk; gated LOW)"
    elif abs(mu_capped) > SIGNAL_HISTORICAL.medium_gating_abs_drift:
        conf = "MEDIUM"
        gate_note = " (large drift; gated MEDIUM)"
    else:
        conf = "HIGH"
        gate_note = ""
    return {
        "drift": float(mu_capped), "confidence": conf,
        "source_quality": "PRIMARY", "sources_count": 1,
        "notes": (f"GARCH-fit on 730d log returns, raw {mu_raw*100:+.0f}%/yr "
                  f"capped at {mu_capped*100:+.0f}%/yr{gate_note}"),
    }


def signal_from_short_interest(short_data):
    """Short interest as drift tilt. Brackets in config signals.short_interest_brackets
    (D-W2-6) — ordered list, first match wins."""
    if not short_data or short_data.get("short_percent_of_float") is None:
        return _none_signal("no short interest data")
    spf = short_data["short_percent_of_float"]
    drift = 0.0
    conf = "LOW"
    note = ""
    for bracket in SIGNAL_SHORT_INTEREST_BRACKETS:
        if spf < bracket.threshold_lt:
            drift = bracket.drift
            conf = bracket.confidence
            note = f"SI {spf*100:.1f}% of float — {bracket.note}"
            break
    return {
        "drift": float(drift), "confidence": conf,
        "source_quality": "PRIMARY", "sources_count": 1,
        "notes": note + f" [via {short_data.get('source', '?')}]",
    }


def signal_from_fundamentals(fundamentals):
    """W6 PR #34 — TTM FCF yield + leverage + margin trend → drift.

    Input `fundamentals` is the dict returned by
    data_fetch.fetch_fundamentals(). When all three sub-components are
    missing (pre-revenue / no FMP data), returns _none_signal.

    The three sub-component drifts are arithmetic-averaged (not summed)
    so a single strong sub-component doesn't dominate — the combined
    signal is a balanced read on financial quality. Cap on combined
    drift prevents extreme tails (drift_cap_abs from YAML).

    Confidence ladder:
      3 components available → HIGH
      2 components           → MEDIUM
      1 component            → LOW
      0                      → _none_signal
    """
    if not fundamentals:
        return _none_signal("no fundamentals fetched")
    n_available = int(fundamentals.get("n_components_available") or 0)
    if n_available == 0:
        return _none_signal("no fundamentals sub-components computable")

    cfg = SIGNAL_FUNDAMENTALS
    drifts = []
    notes = []

    # 1. FCF yield.
    fcfy = fundamentals.get("fcf_yield")
    if fcfy is not None:
        if fcfy >= cfg.fcf_yield_strong_bull:
            d = cfg.bullish_drift_pp
            tag = "FCF strong+"
        elif fcfy >= cfg.fcf_yield_mild_bull:
            d = cfg.mild_drift_pp
            tag = "FCF mild+"
        elif fcfy >= cfg.fcf_yield_neutral_low:
            d = 0.0
            tag = "FCF neutral"
        elif fcfy >= cfg.fcf_yield_strong_bear:
            d = -cfg.mild_drift_pp
            tag = "FCF mild-"
        else:
            d = -cfg.bullish_drift_pp
            tag = "FCF strong-"
        drifts.append(d)
        notes.append(f"{tag} ({fcfy*100:+.1f}%)")

    # 2. Net debt / EBITDA leverage.
    leverage = fundamentals.get("net_debt_to_ebitda")
    if leverage is not None:
        if leverage <= cfg.leverage_strong_bull:
            d = cfg.mild_drift_pp
            tag = "leverage low+"
        elif leverage <= cfg.leverage_neutral_high:
            d = 0.0
            tag = "leverage neutral"
        elif leverage <= cfg.leverage_mild_bear:
            d = -cfg.mild_drift_pp
            tag = "leverage mid-"
        else:
            d = -cfg.bullish_drift_pp
            tag = "leverage high-"
        drifts.append(d)
        notes.append(f"{tag} (ND/EBITDA={leverage:.1f}×)")

    # 3. Operating margin trend (last 4Q avg vs prior 4Q avg).
    margin_trend = fundamentals.get("op_margin_trend")
    if margin_trend is not None:
        if margin_trend >= cfg.margin_trend_bull:
            d = cfg.margin_trend_drift_pp
            tag = "op-margin improving"
        elif margin_trend <= cfg.margin_trend_bear:
            d = -cfg.margin_trend_drift_pp
            tag = "op-margin deteriorating"
        else:
            d = 0.0
            tag = "op-margin stable"
        drifts.append(d)
        notes.append(f"{tag} ({margin_trend*100:+.1f}pp)")

    combined = sum(drifts) / len(drifts) if drifts else 0.0
    capped = max(-cfg.drift_cap_abs, min(cfg.drift_cap_abs, combined))

    if n_available >= 3:
        confidence = "HIGH"
    elif n_available == 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    return {
        "drift": float(capped),
        "confidence": confidence,
        "source_quality": "PRIMARY",
        "sources_count": n_available,
        "notes": "; ".join(notes),
    }


def signal_from_revision_momentum(grades, today=None):
    """W6 PR #35 — analyst upgrade/downgrade momentum.

    Inputs:
      grades: list of dicts with 'publishedDate' (YYYY-MM-DD...) and
              'action' ("upgrade" / "downgrade" / "maintain" / "init" /
              "reiterated") from data_fetch.fetch_grades_history.
      today : datetime.date for the reference "now" — defaults to today.
              Tests pass an explicit date for determinism.

    Algorithm: each grade-change action contributes +1 / -1 / 0 weighted
    by time-decay bucket. Recent actions (last 30d) get full weight;

    PR #81 (audit #12): time-decay buckets here are CALENDAR days
    (`(today - gdate).days`), NOT trading days. This is intentional —
    analyst events are calendar-time-natural (a Tuesday upgrade and a
    Saturday upgrade are equally fresh to a Monday-morning reader; the
    NYSE schedule doesn't determine when sell-side desks publish).
    Do not "fix" this to trading days via `market_calendar` helpers;
    the audit explicitly evaluated this site and graded it informational.
    older actions decay. Sum is multiplied by drift_per_unit_pp and
    capped at drift_cap_abs.

    Returns the standard signal dict; _none_signal when no grades in
    the lookback window (small caps with no coverage).
    """
    if not grades:
        return _none_signal("no analyst coverage / no grade changes")

    cfg = SIGNAL_REVISION_MOMENTUM
    if today is None:
        today = datetime.now().date()

    weighted_score = 0.0
    in_window_count = 0
    n_up = 0
    n_down = 0

    for g in grades:
        if not isinstance(g, dict):
            continue
        date_str = g.get("publishedDate") or g.get("date") or ""
        try:
            # FMP returns "YYYY-MM-DD HH:MM:SS" or ISO — first 10 chars suffice.
            gdate = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        age_days = (today - gdate).days
        if age_days < 0 or age_days > cfg.lookback_days:
            continue
        if age_days <= 30:
            w = cfg.recent_weight
        elif age_days <= 60:
            w = cfg.medium_weight
        else:
            w = cfg.older_weight
        action = str(g.get("action", "")).lower()
        if action == "upgrade":
            sign = 1
            n_up += 1
        elif action == "downgrade":
            sign = -1
            n_down += 1
        else:
            # "maintain" / "init" / "reiterated" → no direction
            continue
        weighted_score += sign * w
        in_window_count += 1

    if in_window_count == 0:
        return _none_signal("no directional grade changes in lookback window")

    drift = weighted_score * cfg.drift_per_unit_pp
    capped = max(-cfg.drift_cap_abs, min(cfg.drift_cap_abs, drift))

    if in_window_count >= cfg.conf_high_count:
        confidence = "HIGH"
    elif in_window_count >= cfg.conf_medium_count:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    note = (
        f"{n_up} upgrades / {n_down} downgrades over last {cfg.lookback_days}d "
        f"(weighted score {weighted_score:+.1f})"
    )
    return {
        "drift": float(capped),
        "confidence": confidence,
        "source_quality": "PRIMARY",
        "sources_count": in_window_count,
        "notes": note,
    }


def signal_from_peer_rs(price_df, peer_dfs, lookback_days=60, ticker="ticker"):
    """Relative strength vs peer median return over lookback_days."""
    if price_df is None or len(price_df) < lookback_days + 1:
        return _none_signal("insufficient price history for peer RS")
    if not peer_dfs:
        return _none_signal("no peer data available")

    def n_day_return(df, n):
        if len(df) < n + 1:
            return None
        try:
            return float(df["Close"].iloc[-1] / df["Close"].iloc[-n - 1] - 1.0)
        except (IndexError, ValueError):
            return None

    own_ret = n_day_return(price_df, lookback_days)
    if own_ret is None:
        return _none_signal(f"could not compute {ticker} return")

    peer_rets = []
    for p, df in peer_dfs.items():
        r = n_day_return(df, lookback_days)
        if r is not None:
            peer_rets.append((p, r))
    if not peer_rets:
        return _none_signal("no peer returns computable")

    peer_median = float(np.median([r for _, r in peer_rets]))
    rs = own_ret - peer_median
    drift = rs * 252 / lookback_days
    cap = SIGNAL_PEER_RS.drift_cap_abs
    drift = max(-cap, min(cap, drift))

    if len(peer_rets) >= 2:
        peer_dispersion = float(np.std([r for _, r in peer_rets]))
        if peer_dispersion < SIGNAL_PEER_RS.dispersion_high_conf:
            conf = "HIGH"
        elif peer_dispersion < SIGNAL_PEER_RS.dispersion_medium_conf:
            conf = "MEDIUM"
        else:
            conf = "LOW"
    else:
        conf = "LOW"

    peer_list = ", ".join([f"{p} {r*100:+.0f}%" for p, r in peer_rets])
    return {
        "drift": float(drift), "confidence": conf,
        "source_quality": "PRIMARY", "sources_count": len(peer_rets),
        "notes": (f"{ticker} {own_ret*100:+.0f}% vs peers [{peer_list}] over {lookback_days}d "
                  f"-> RS {rs*100:+.0f}%, annualised tilt {drift*100:+.0f}%"),
    }


def signal_from_sector_decoupling(price_df, sector_perf, lookback_days=30, ticker="ticker"):
    """Decoupling: is ticker moving WITH or AGAINST its sector recently?

    PR #79 (audit #5): the two returns must be measured over the SAME
    window. Pre-fix:
      - own_ret = 30-trading-bar return (`iloc[-31]`).
      - sector_ret = cumulative return over `len(rows)` trading bars
        returned by `fetch_sector_perf` (could be 28-30 depending on
        FMP coverage gaps + holiday density in the calendar window
        the fetcher requested).
      - decoup = own_ret - sector_ret subtracted mismatched-period
        returns; annualisation `decoup * 252 / lookback_days` used
        the SIGNAL's nominal `lookback_days` (30), not the SECTOR's
        actual period (`sector_perf['n_days']`).
    Fix: use `sector_perf['n_days']` as the comparison window. Slice
    own_ret over those same N bars; annualise with `252 / n_days`. If
    the sector returned far fewer bars than requested, mark LOW conf
    rather than carry a stealthily-broken comparison forward.
    """
    if not sector_perf or sector_perf.get("cum_return_pct") is None:
        return _none_signal("no sector data for decoupling")

    # Use the SECTOR's actual period for both legs. Fall back to the
    # signal's nominal lookback if `n_days` is missing (legacy fetcher
    # without that field).
    n_days = int(sector_perf.get("n_days") or lookback_days)
    if n_days < 5:
        return _none_signal(
            f"sector window too short ({n_days} bars) — signal not reliable"
        )

    if price_df is None or len(price_df) < n_days + 1:
        return _none_signal(
            f"insufficient price history for {n_days}-bar decoupling"
        )

    try:
        own_ret = float(price_df["Close"].iloc[-1] /
                         price_df["Close"].iloc[-n_days - 1] - 1.0)
    except (IndexError, ValueError):
        return _none_signal(f"{ticker} return calc failed")

    sector_ret = sector_perf["cum_return_pct"] / 100.0
    decoup = own_ret - sector_ret
    drift = decoup * 252 / n_days
    cap = SIGNAL_SECTOR_DECOUPLING.drift_cap_abs
    drift = max(-cap, min(cap, drift))

    # Period-mismatch downgrade: if the SECTOR window we ended up using
    # differs materially from the signal's nominal lookback, the signal
    # still works (both legs measure the same period) but the operator
    # should know the comparison is shorter/longer than expected.
    period_drift_pp = abs(n_days - lookback_days)
    period_note = (
        f" [window {n_days}d, nominal {lookback_days}d]"
        if period_drift_pp >= 5 else ""
    )

    if abs(decoup) < SIGNAL_SECTOR_DECOUPLING.magnitude_low_conf:
        conf = "LOW"
        note_extra = "(low decoupling, signal noisy)"
    elif abs(decoup) < SIGNAL_SECTOR_DECOUPLING.magnitude_medium_conf:
        conf = "MEDIUM"
        note_extra = ""
    else:
        conf = "HIGH"
        note_extra = "(meaningful decoupling)"

    return {
        "drift": float(drift), "confidence": conf,
        "source_quality": "PRIMARY", "sources_count": 1,
        "notes": (f"{ticker} {own_ret*100:+.0f}% vs sector "
                  f"{sector_ret*100:+.0f}% over {n_days}d -> decouple "
                  f"{decoup*100:+.0f}% {note_extra}{period_note}"),
    }


# =============================================================================
# Regime detection + vol regime advisory (v1)
# =============================================================================

def detect_swing_regime(rsi, mom_5d, mom_30d_pct, sigma, ytd_return_pct=None):
    """Classify regime for signal interpretation."""
    regime = "UNCERTAIN"
    detail = ""

    # Triggers in config signals.regime_detection (D-W2-6).
    rd = SIGNAL_REGIME_DETECTION
    is_high_vol = sigma > rd.sigma_high_threshold
    has_parabola = ytd_return_pct is not None and ytd_return_pct > rd.ytd_parabola_pct
    rsi_overbought = rsi is not None and rsi > rd.rsi_overbought
    rsi_oversold = rsi is not None and rsi < rd.rsi_oversold
    mom5_pos = mom_5d > rd.mom_5d_threshold
    mom5_neg = mom_5d < -rd.mom_5d_threshold
    mom30_pos = mom_30d_pct is not None and mom_30d_pct > rd.mom_30d_pct_threshold
    mom30_neg = mom_30d_pct is not None and mom_30d_pct < -rd.mom_30d_pct_threshold

    if has_parabola:
        regime = "POST_PARABOLA"
        detail = (f"YTD +{ytd_return_pct:.0f}% — parabolic rally; "
                  f"mean-reversion risk over horizon > weeks")
    elif rsi_overbought and mom5_neg:
        regime = "MEAN_REVERSION"
        detail = f"RSI {rsi:.0f} (overbought) + 5d momentum {mom_5d*100:+.1f}% diverging"
    elif rsi_oversold and mom5_pos:
        regime = "MEAN_REVERSION"
        detail = f"RSI {rsi:.0f} (oversold) + 5d momentum {mom_5d*100:+.1f}% diverging upward"
    elif mom30_pos and mom5_pos and not rsi_overbought:
        regime = "MOMENTUM_BULL"
        detail = f"30d momentum +{mom_30d_pct:.1f}%, 5d {mom_5d*100:+.1f}%, RSI {rsi:.0f}"
    elif mom30_neg and mom5_neg:
        regime = "MOMENTUM_BEAR"
        detail = f"30d momentum {mom_30d_pct:+.1f}%, 5d {mom_5d*100:+.1f}%, RSI {rsi:.0f}"
    elif not is_high_vol and abs(mom_5d) < rd.mom_5d_threshold:
        regime = "RANGE"
        detail = f"low vol ({sigma*100:.0f}%) + flat momentum"
    else:
        detail = (f"RSI {rsi:.0f}, 5d {mom_5d*100:+.1f}%, "
                  f"30d {mom_30d_pct:+.1f}% — no clear regime"
                  if mom_30d_pct is not None
                  else f"RSI {rsi:.0f}, 5d {mom_5d*100:+.1f}% — no clear regime")

    return {"regime": regime, "detail": detail,
            "is_high_vol": is_high_vol, "has_parabola": has_parabola}


def vol_regime_advisory(sigma):
    """Translate vol level into decision-quality advisory."""
    sigma_pct = sigma * 100
    if sigma_pct >= 80:
        return {
            "level": "EXTREME",
            "advisory": (
                f"Sigma {sigma_pct:.0f}% — drift estimate is 2nd-order. At this vol, "
                "the outcome is dominated by dispersion, not direction. Focus on "
                "TAIL-RISK metrics (panic-floor touch probability, max drawdown "
                "distribution) rather than the blended drift point."),
            "drift_decisive": False,
        }
    elif sigma_pct >= 50:
        return {
            "level": "HIGH",
            "advisory": (
                f"Sigma {sigma_pct:.0f}% — high vol regime. Drift matters but "
                "the cushion above break-even must be earned BOTH from drift "
                "advantage AND from acceptable tail risk. Watch panic-floor probability."),
            "drift_decisive": False,
        }
    elif sigma_pct >= 25:
        return {
            "level": "NORMAL",
            "advisory": (
                f"Sigma {sigma_pct:.0f}% — normal vol regime. Blended drift is the "
                "primary input for the hold/cut decision."),
            "drift_decisive": True,
        }
    else:
        return {
            "level": "LOW",
            "advisory": (
                f"Sigma {sigma_pct:.0f}% — low vol regime. Drift dominates outcome. "
                "Cushion math is highly reliable; small drift changes flip verdicts."),
            "drift_decisive": True,
        }


# =============================================================================
# Blend + Bayesian update (v1)
# =============================================================================

# Signals whose drift comes from the AI layer. When ANY of these is dropped
# (source_quality == NONE_FOUND), the blend loses forward-looking synthesis
# capacity that the math signals can't substitute. The std must inflate to
# reflect that — see PHANTOM_SIGNAL_SE below.
_AI_DERIVED_SIGNAL_NAMES = ("ai", "catalyst_proximity", "narrative")

# Standard error attributed to a missing AI-derived signal for purposes of
# within_var inflation. 0.20 = LOW-confidence SE. The phantom signal's
# weight (original weight from the table, e.g. 0.25 for `ai`) and this SE
# combine in quadrature with the active signals' contributions. Result:
# ~+5.7pp added to blend std when all three AI signals are missing — the
# correct expression of "we are less certain when forward-looking synthesis
# is unavailable" rather than the previous behavior where dropping AI made
# the std artificially TIGHTER (because AI's between-signal disagreement
# also disappeared from between_var). Sourced from config/diprally.yaml
# via src.config — sacred decision #17.
from src.config import PHANTOM_SIGNAL_SE_CONFIG as PHANTOM_SIGNAL_SE  # noqa: E402


# PR #59 — D-W2-18: per-signal saturation caps for the multi-saturation
# detector. When |drift| ≥ MULTI_SATURATION.saturation_threshold × cap,
# the signal counts as "at cap". This dict knows each saturating
# signal's cap_abs; signals not listed never count as saturated.
def _signal_saturation_caps() -> dict:
    """Build the {signal_name → cap_abs} dict at module-load time
    from each signal's individual config. New signals with caps
    should be added here so the multi-saturation detector sees them."""
    from src.config import (
        SIGNAL_CATALYST_PROXIMITY,
        SIGNAL_FUNDAMENTALS,
        SIGNAL_PEER_RS,
        SIGNAL_REVISION_MOMENTUM,
        SIGNAL_SECTOR_DECOUPLING,
    )
    caps = {
        "peer_rs": SIGNAL_PEER_RS.drift_cap_abs,
        "sector_decoupling": SIGNAL_SECTOR_DECOUPLING.drift_cap_abs,
        "catalyst_proximity": SIGNAL_CATALYST_PROXIMITY.drift_cap_abs,
        "fundamentals": SIGNAL_FUNDAMENTALS.drift_cap_abs,
        "revision_momentum": SIGNAL_REVISION_MOMENTUM.drift_cap_abs,
    }
    return caps


_SIGNAL_SATURATION_CAPS = _signal_saturation_caps()


def _detect_multi_saturation(signals, effective_weights) -> dict:
    """PR #59 — count signals saturating at cap, same direction.
    Returns {n_saturated_pos, n_saturated_neg, max_count}.
    Only counts signals that:
      1. Have a known cap (in _SIGNAL_SATURATION_CAPS)
      2. Have effective_weight > 0 (passed quality gates)
      3. |drift| ≥ saturation_threshold × cap
    """
    from src.config import MULTI_SATURATION
    threshold = MULTI_SATURATION.saturation_threshold
    n_pos = 0
    n_neg = 0
    for name, info in signals.items():
        if name not in _SIGNAL_SATURATION_CAPS:
            continue
        if effective_weights.get(name, 0.0) <= 0:
            continue
        drift = info.get("drift")
        if drift is None:
            continue
        cap = _SIGNAL_SATURATION_CAPS[name]
        if abs(drift) >= threshold * cap:
            if drift > 0:
                n_pos += 1
            else:
                n_neg += 1
    return {
        "n_saturated_pos": n_pos,
        "n_saturated_neg": n_neg,
        "max_count": max(n_pos, n_neg),
    }


def blend_with_uncertainty(signals, weights_dict=None):
    """Blend with confidence intervals via signal-weighted variance.

    Quality gates:
      - NONE_FOUND → drop from the active set.
      - SPECULATIVE + single source → halve weight.
      - LOW confidence → halve weight.

    Phantom-signal std accounting:
      AI-derived signals (ai, catalyst_proximity, narrative) that drop with
      NONE_FOUND status are not silently absorbed by the proportional
      renormalization of the remaining signals. Their original weights
      contribute to within_var with a LOW-conf SE, inflating the posterior
      std. This corrects the W0 bug where the blend reported TIGHTER
      confidence with LESS information — the most-disagreeing signal (AI
      typically diverges from math consensus) disappearing from between_var
      compounded the error. Now: dropping AI widens the band ~5.7pp in std,
      Bayesian smoothing then weighs the prior more heavily.
    """
    if weights_dict is None:
        weights_dict = BLEND_WEIGHTS

    effective = {}
    phantom_weights = {}  # AI-derived signals that got dropped → contribute to within_var
    for name, info in signals.items():
        original_w = weights_dict.get(name, 0.0)
        if info.get("drift") is None:
            effective[name] = 0.0
            if name in _AI_DERIVED_SIGNAL_NAMES and original_w > 0:
                phantom_weights[name] = original_w
            continue
        w = original_w
        sq = info.get("source_quality", "REPUTABLE")
        sc = int(info.get("sources_count", 0))
        conf = info.get("confidence", "MEDIUM")
        if sq == "NONE_FOUND":
            w = 0.0
            if name in _AI_DERIVED_SIGNAL_NAMES and original_w > 0:
                phantom_weights[name] = original_w
        elif sq == "SPECULATIVE" and sc < 2:
            w *= 0.5
        if conf == "LOW":
            w *= 0.5
        effective[name] = w

    total = sum(effective.values())
    if total <= 0:
        return {"blended": None, "std": None,
                "lo68": None, "hi68": None, "lo95": None, "hi95": None,
                "weights": effective, "fallback": True,
                "dispersion_pp": 0.0, "n_active": 0,
                "phantom_signals": list(phantom_weights.keys()),
                "phantom_std_inflation": 0.0}

    norm = {n: w/total for n, w in effective.items()}

    blended = sum(signals[n]["drift"] * norm[n]
                  for n in signals if norm[n] > 0)

    within_var = 0.0
    between_var = 0.0
    for n in signals:
        if norm[n] <= 0:
            continue
        se = CONFIDENCE_TO_SE.get(signals[n].get("confidence", "MEDIUM"), 0.10)
        within_var += (norm[n] ** 2) * (se ** 2)
        between_var += norm[n] * (signals[n]["drift"] - blended) ** 2

    # Phantom-signal within_var inflation. Each missing AI-derived signal
    # contributes (original_weight)² × PHANTOM_SIGNAL_SE² as if it were a
    # LOW-conf signal sitting at the blended drift (zero between-contribution,
    # nonzero within-contribution).
    phantom_within = sum(
        (w ** 2) * (PHANTOM_SIGNAL_SE ** 2)
        for w in phantom_weights.values()
    )
    total_var = within_var + between_var + phantom_within
    pre_saturation_std = total_var ** 0.5

    # PR #59 — D-W2-18 multi-saturation std inflation. When N≥min_count
    # capped signals all hit cap same-direction, the blend is reading
    # what's really ONE correlated signal as N independent ones →
    # over-confident posterior. Inflate std proportionally.
    from src.config import MULTI_SATURATION
    sat = _detect_multi_saturation(signals, effective)
    if sat["max_count"] >= MULTI_SATURATION.min_count:
        std = pre_saturation_std * MULTI_SATURATION.inflation_multiplier
    else:
        std = pre_saturation_std

    phantom_std_inflation = (total_var ** 0.5) - ((within_var + between_var) ** 0.5)
    multi_saturation_std_inflation = std - pre_saturation_std

    active_drifts = [signals[n]["drift"] for n in signals
                     if norm[n] > 0 and signals[n]["drift"] is not None]
    dispersion = (max(active_drifts) - min(active_drifts)) * 100 if active_drifts else 0

    return {
        "blended": float(blended),
        "std": float(std),
        "lo68": float(blended - std),
        "hi68": float(blended + std),
        "lo95": float(blended - 2 * std),
        "hi95": float(blended + 2 * std),
        "weights": effective,
        "fallback": False,
        "dispersion_pp": float(dispersion),
        "n_active": sum(1 for w in effective.values() if w > 0),
        "phantom_signals": list(phantom_weights.keys()),
        "phantom_std_inflation": float(phantom_std_inflation),
        # PR #59 — D-W2-18 multi-saturation diagnostics.
        "n_saturated_pos": sat["n_saturated_pos"],
        "n_saturated_neg": sat["n_saturated_neg"],
        "multi_saturation_std_inflation": float(multi_saturation_std_inflation),
    }


def bayesian_update(prior_blend, today_blend, prior_age_days=1):
    """Bayesian update of blended drift; yesterday's posterior = today's prior."""
    if today_blend.get("blended") is None or today_blend.get("std") is None:
        return None
    if not prior_blend or prior_blend.get("blended") is None:
        return {"posterior_mu": today_blend["blended"],
                "posterior_std": today_blend["std"],
                "prior_weight": 0.0, "obs_weight": 1.0,
                "note": "no prior available — using today's blend"}

    from src.config import (
        BAYESIAN_DEFAULT_PRIOR_STD,
        BAYESIAN_PRIOR_AGE_INFLATION_PER_DAY,
    )
    prior_mu = prior_blend["blended"]
    prior_std = prior_blend.get("std", BAYESIAN_DEFAULT_PRIOR_STD)
    obs_mu = today_blend["blended"]
    obs_std = today_blend["std"]

    inflation = 1.0 + BAYESIAN_PRIOR_AGE_INFLATION_PER_DAY * max(0, prior_age_days - 1)
    prior_var = (prior_std * inflation) ** 2
    obs_var = obs_std ** 2

    posterior_var = 1.0 / (1.0/prior_var + 1.0/obs_var)
    posterior_mu = posterior_var * (prior_mu/prior_var + obs_mu/obs_var)
    posterior_std = posterior_var ** 0.5

    prior_weight = posterior_var / prior_var
    obs_weight = posterior_var / obs_var

    return {"posterior_mu": float(posterior_mu),
            "posterior_std": float(posterior_std),
            "prior_weight": float(prior_weight),
            "obs_weight": float(obs_weight),
            "note": (f"Bayesian: prior_mu={prior_mu*100:+.1f}% std={prior_std*100:.1f}%, "
                     f"obs_mu={obs_mu*100:+.1f}% std={obs_std*100:.1f}%, "
                     f"weights {prior_weight*100:.0f}/{obs_weight*100:.0f}")}


# =============================================================================
# v2 AI-derived signals: catalyst proximity, narrative, factor arithmetic,
# beta-adjusted unusual-move Z.
# =============================================================================

def signal_from_catalyst_proximity(catalysts, horizon_days):
    """AI-identified catalysts within horizon → direction-weighted drift signal.

    Returns (mu_annual, confidence, rationale).
    """
    if not catalysts:
        return 0.0, "LOW", "no catalysts identified"

    # Thresholds in config signals.catalyst_proximity (D-W2-6).
    cp = SIGNAL_CATALYST_PROXIMITY
    today = datetime.now().date()
    # PR #76: horizon_days is TRADING days; convert via market calendar
    # (was `today + timedelta(days=horizon_days)`, which under-counted
    # the actual horizon by ~28%).
    from src.market_calendar import add_trading_days
    horizon_end = add_trading_days(today, horizon_days)
    mag_map = cp.magnitude_drift_map
    mag_default = mag_map.get("med", 0.05)
    dir_sign = {"bullish": 1.0, "bearish": -1.0, "two-sided": 0.0}

    total = 0.0
    in_window_count = 0
    for c in catalysts:
        if not isinstance(c, dict):
            continue
        date_str = c.get("date_or_window", "")
        cdate = _parse_catalyst_date(date_str)
        if cdate is None:
            continue
        if today <= cdate <= horizon_end:
            mag = mag_map.get(str(c.get("magnitude", "med")).lower(), mag_default)
            sign = dir_sign.get(str(c.get("direction_risk", "two-sided")).lower(), 0.0)
            total += mag * sign
            in_window_count += 1

    cap = cp.drift_cap_abs
    total = max(-cap, min(cap, total))
    if in_window_count == 0:
        return 0.0, "LOW", "no catalysts in horizon"

    if in_window_count >= cp.in_window_count_high_conf:
        conf = "HIGH"
    elif in_window_count >= cp.in_window_count_medium_conf:
        conf = "MEDIUM"
    else:
        conf = "LOW"
    return total, conf, f"{in_window_count} catalysts in horizon, net {total:+.1%}"


def signal_from_structural_narrative(narrative_score, evidence_count):
    """AI narrative_score → drift adjustment. Requires evidence for 'strong'."""
    adj = NARRATIVE_DRIFT_ADJUSTMENT.get(narrative_score, 0.0)
    if narrative_score == "strong" and evidence_count < 2:
        return 0.0, "LOW", "strong narrative claimed but insufficient evidence — defaulting neutral"
    conf = "MEDIUM" if narrative_score != "neutral" else "LOW"
    return adj, conf, f"narrative={narrative_score} ({evidence_count} evidence sources)"


def _factor_weight(f):
    """Defensive: AI might return list of dicts OR list of strings."""
    if isinstance(f, dict):
        return str(f.get("weight", "low")).lower()
    return "low"


def apply_bull_bear_arithmetic(bull_factors, bear_factors):
    """Sum weighted bull/bear factors; return drift tail bias + rationale."""
    bull_high = sum(1 for f in bull_factors if _factor_weight(f) == "high")
    bear_high = sum(1 for f in bear_factors if _factor_weight(f) == "high")
    net = bull_high * FACTOR_WEIGHTS["high"] - bear_high * FACTOR_WEIGHTS["high"]
    if net > FACTOR_NET_THRESHOLD:
        return FACTOR_TAIL_BIAS, f"HIGH-bull dominance (net +{net}) → +{FACTOR_TAIL_BIAS:.0%} rally bias"
    if net < -FACTOR_NET_THRESHOLD:
        return -FACTOR_TAIL_BIAS, f"HIGH-bear dominance (net {net}) → -{FACTOR_TAIL_BIAS:.0%} dip bias"
    return 0.0, f"factors balanced (net {net:+d}) → no tail bias"


def compute_unusual_move_z(history_df, beta=1.0, lookback=60):
    """Beta-adjusted residual Z-score for today's return.

    |Z| >= CATALYST_Z_THRESHOLD signals an unusual move that may have a hidden
    catalyst. Used for situational awareness, not yet a drift signal.
    """
    if history_df is None or "Close" not in history_df.columns or len(history_df) < lookback + 1:
        return None
    try:
        closes = history_df["Close"].astype(float).values
        returns_ = np.diff(np.log(closes))
        if len(returns_) < lookback:
            return None
        today_return = float(returns_[-1])
        historical_vol = float(np.std(returns_[-lookback:]))
        if historical_vol <= 0:
            return None
        beta_safe = max(0.5, float(beta or 1.0))
        raw_z = abs(today_return) / historical_vol
        adjusted_z = raw_z / beta_safe
        return {
            "z_score": round(adjusted_z, 2),
            "return_pct": round(today_return * 100, 2),
            "beta": round(beta_safe, 2),
            "triggered": adjusted_z >= CATALYST_Z_THRESHOLD,
        }
    except Exception:
        return None


# Catalyst date parser — needed by signal_from_catalyst_proximity. Re-exported
# from engine.py as parse_catalyst_date for use elsewhere.
def _parse_catalyst_date(date_str):
    """Robust catalyst date parser. Handles Y/M/Q/H/range/relative formats.

    PR #51 extended set:
      - "ongoing" / "rolling" / "<year>-rolling"  → today + 1 day
        (semantically "active across horizon" → treat as in-horizon)
      - "<year>-H1" / "H1 <year>" / "H2 <year>"   → first day of half
      - "Q<n>" / "Q<n> <year>" without dash       → first day of quarter
        (year defaults to current year, then auto-advances to next year
         if the resulting date is already past today)
    """
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip().lower()
    today = datetime.now().date()

    # PR #51: "ongoing" / "rolling" semantics — catalyst is active across
    # the horizon window. Treat as in-horizon by returning a near-future
    # date (today+1 ensures the in-window check captures it). Pass 1/
    # Pass 2 use these for overhangs (insider selling, secondary risk,
    # debt refinancing window, export-control regime risk, etc.).
    if s in ("ongoing", "rolling", "continuous", "current",
              "tbd-rolling", "n/a"):
        return today + timedelta(days=1)

    # PR #51: "<year>-rolling" or "<year> rolling" — year-tagged rolling.
    import re
    m = re.match(r"(\d{4})[-\s]?rolling$", s)
    if m:
        return today + timedelta(days=1)

    # Existing: "/" separated range → parse each, return EARLIEST.
    if "/" in s:
        parts = [p.strip() for p in s.split("/") if p.strip()]
        candidates = [_parse_catalyst_date(p) for p in parts]
        valid = [c for c in candidates if c is not None]
        return min(valid) if valid else None

    # Existing: relative "next NN days"
    m = re.match(r"next\s+(\d+)\s*d", s)
    if m:
        offset = int(m.group(1)) // 2
        return today + timedelta(days=offset)

    # PR #51: "<year>-H1" / "<year>-H2" half-year format.
    m = re.match(r"(\d{4})[-\s]?h([12])$", s)
    if m:
        year = int(m.group(1))
        h = int(m.group(2))
        month = 1 if h == 1 else 7
        try:
            return datetime(year, month, 1).date()
        except ValueError:
            return None

    # PR #51: "H1 <year>" / "H2 <year>" reversed order.
    m = re.match(r"h([12])\s+(\d{4})$", s)
    if m:
        h = int(m.group(1))
        year = int(m.group(2))
        month = 1 if h == 1 else 7
        try:
            return datetime(year, month, 1).date()
        except ValueError:
            return None

    # Existing: "<year>-Q<n>"
    m = re.match(r"(\d{4})[-\s]?q([1-4])", s)
    if m:
        year = int(m.group(1))
        q = int(m.group(2))
        month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        try:
            return datetime(year, month, 1).date()
        except ValueError:
            return None

    # PR #51: "Q<n> <year>" reversed order.
    m = re.match(r"q([1-4])\s+(\d{4})$", s)
    if m:
        q = int(m.group(1))
        year = int(m.group(2))
        month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        try:
            return datetime(year, month, 1).date()
        except ValueError:
            return None

    # PR #51: "Q<n>" without year — assume current year; if past, roll forward.
    m = re.match(r"q([1-4])$", s)
    if m:
        q = int(m.group(1))
        month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        candidate = datetime(today.year, month, 1).date()
        if candidate < today:
            candidate = datetime(today.year + 1, month, 1).date()
        return candidate

    # Existing: "<year>-MM" or "<year>-MM-DD"
    m = re.match(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", s)
    if m:
        try:
            year = int(m.group(1))
            month = int(m.group(2))
            day = int(m.group(3)) if m.group(3) else 1
            return datetime(year, month, day).date()
        except ValueError:
            return None

    # Existing: bare year
    m = re.match(r"^(\d{4})$", s)
    if m:
        try:
            return datetime(int(m.group(1)), 1, 1).date()
        except ValueError:
            return None

    return None


# Public alias for backwards-compatible reference
parse_catalyst_date = _parse_catalyst_date

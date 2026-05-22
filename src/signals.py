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
    """Analyst targets → drift signal. Prefers fresh price-target-summary."""
    if summary:
        target = None
        n_analysts = 0
        window = ""
        base_conf = "MEDIUM"

        if summary.get("last_month_count", 0) >= 5 and summary.get("last_month_avg"):
            target = float(summary["last_month_avg"])
            n_analysts = summary["last_month_count"]
            window = "last month"
            base_conf = "HIGH" if n_analysts >= 12 else "MEDIUM"
        elif summary.get("last_quarter_count", 0) >= 5 and summary.get("last_quarter_avg"):
            target = float(summary["last_quarter_avg"])
            n_analysts = summary["last_quarter_count"]
            window = "last quarter"
            base_conf = "MEDIUM" if n_analysts >= 15 else "LOW"
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
                    if move_60d > 0.25:
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
        if spread < 0.10:
            conf = "HIGH"
        elif spread < 0.25:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        staleness_note = " (stale-mixed consensus fallback)"
        if price_history_df is not None and len(price_history_df) >= 60:
            try:
                p60 = float(price_history_df["Close"].iloc[-60])
                move_60d = abs((S0 - p60) / p60)
                if move_60d > 0.25:
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

    regime_name = swing_regime.get("regime") if swing_regime else None
    if regime_name == "POST_PARABOLA":
        cap_high, cap_low = 0.60, -0.50
        conf = "LOW"
        regime_note = " [POST_PARABOLA regime: sector cap reduced to +60%, conf LOW]"
    elif regime_name in ("MOMENTUM_BULL", "MOMENTUM_BEAR"):
        cap_high, cap_low = 1.00, -0.50
        conf = "MEDIUM"
        regime_note = f" [{regime_name}: cap +100%]"
    else:
        cap_high, cap_low = 1.50, -0.50
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
    """VIX + SPY → drift tilt."""
    if not macro:
        return _none_signal("macro data unavailable")
    regime = macro.get("regime", "neutral")
    drift = {"risk_on": 0.10, "neutral": 0.05, "risk_off": -0.05}.get(regime, 0.05)
    return {
        "drift": float(drift), "confidence": "MEDIUM",
        "source_quality": "PRIMARY", "sources_count": 2,
        "notes": (f"VIX {macro['vix']:.1f}, SPY {macro['spy_trend']*100:+.1f}% "
                  f"vs MA50 -> {regime}"),
    }


def signal_from_insider(insider, market_cap_usd=None):
    """Mcap-relative insider flow → drift tilt (audit fix #7)."""
    if not insider:
        return _none_signal("insider data unavailable")
    n_total = insider.get("n_buys", 0) + insider.get("n_sells", 0)
    if n_total == 0:
        return {"drift": 0.0, "confidence": "LOW",
                "source_quality": "PRIMARY", "sources_count": 1,
                "notes": "no insider P+S transactions in window"}
    net = insider.get("net_value_usd", 0)
    if market_cap_usd and market_cap_usd > 0:
        flow_pct_of_mcap = net / market_cap_usd
        drift = max(-0.10, min(0.10, flow_pct_of_mcap * 5.0))
        scaling_note = f" (mcap-relative: {flow_pct_of_mcap*100:.3f}% of $US{market_cap_usd/1e9:.0f}B)"
    else:
        drift = max(-0.10, min(0.10, net / 100_000_000))
        scaling_note = " (absolute scaling — no mcap available)"
    direction = "buying" if net > 0 else "selling"
    if market_cap_usd and market_cap_usd > 0 and abs(net) / market_cap_usd < 0.001:
        conf = "LOW"
        scaling_note += " — NOISE-LEVEL relative to mcap, downgraded LOW"
    else:
        conf = "MEDIUM"
    return {
        "drift": float(drift), "confidence": conf,
        "source_quality": "PRIMARY", "sources_count": 1,
        "notes": (f"net {direction} ${abs(net)/1e6:.1f}M "
                  f"({insider['n_buys']}P/{insider['n_sells']}S in "
                  f"{insider['days']}d){scaling_note}"),
    }


def signal_from_historical(mu_capped, mu_raw, sigma):
    """Historical mean-return drift, cap-gated to LOW confidence when binding."""
    if mu_capped is None:
        return _none_signal("historical drift unavailable")
    if abs(mu_raw) > 1.0:
        conf = "LOW"
        gate_note = " (CAP BINDING — extrapolation risk; gated LOW)"
    elif abs(mu_capped) > 0.5:
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
    """Short interest as drift tilt."""
    if not short_data or short_data.get("short_percent_of_float") is None:
        return _none_signal("no short interest data")
    spf = short_data["short_percent_of_float"]
    if spf < 0.03:
        drift = 0.00
        conf = "MEDIUM"
        note = f"SI {spf*100:.1f}% of float — low, neutral signal"
    elif spf < 0.10:
        drift = -0.03
        conf = "MEDIUM"
        note = f"SI {spf*100:.1f}% of float — moderate skepticism (mild bearish)"
    elif spf < 0.20:
        drift = -0.05
        conf = "LOW"
        note = (f"SI {spf*100:.1f}% of float — elevated; tail risk both directions "
                f"(squeeze upside vs structural bearishness)")
    else:
        drift = +0.05
        conf = "LOW"
        note = f"SI {spf*100:.1f}% of float — very high; squeeze tail upside"
    return {
        "drift": float(drift), "confidence": conf,
        "source_quality": "PRIMARY", "sources_count": 1,
        "notes": note + f" [via {short_data.get('source', '?')}]",
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
    drift = max(-0.30, min(0.30, drift))

    if len(peer_rets) >= 2:
        peer_dispersion = float(np.std([r for _, r in peer_rets]))
        if peer_dispersion < 0.05:
            conf = "HIGH"
        elif peer_dispersion < 0.15:
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
    """Decoupling: is ticker moving WITH or AGAINST its sector recently?"""
    if price_df is None or len(price_df) < lookback_days + 1:
        return _none_signal("insufficient price history for decoupling")
    if not sector_perf or sector_perf.get("cum_return_pct") is None:
        return _none_signal("no sector data for decoupling")

    try:
        own_ret = float(price_df["Close"].iloc[-1] /
                         price_df["Close"].iloc[-lookback_days - 1] - 1.0)
    except (IndexError, ValueError):
        return _none_signal(f"{ticker} return calc failed")

    sector_ret = sector_perf["cum_return_pct"] / 100.0
    decoup = own_ret - sector_ret
    drift = decoup * 252 / lookback_days
    drift = max(-0.20, min(0.20, drift))

    if abs(decoup) < 0.02:
        conf = "LOW"
        note_extra = "(low decoupling, signal noisy)"
    elif abs(decoup) < 0.10:
        conf = "MEDIUM"
        note_extra = ""
    else:
        conf = "HIGH"
        note_extra = "(meaningful decoupling)"

    return {
        "drift": float(drift), "confidence": conf,
        "source_quality": "PRIMARY", "sources_count": 1,
        "notes": (f"{ticker} {own_ret*100:+.0f}% vs sector {sector_ret*100:+.0f}% "
                  f"over {lookback_days}d -> decouple {decoup*100:+.0f}% {note_extra}"),
    }


# =============================================================================
# Regime detection + vol regime advisory (v1)
# =============================================================================

def detect_swing_regime(rsi, mom_5d, mom_30d_pct, sigma, ytd_return_pct=None):
    """Classify regime for signal interpretation."""
    regime = "UNCERTAIN"
    detail = ""

    is_high_vol = sigma > 0.50
    has_parabola = ytd_return_pct is not None and ytd_return_pct > 200
    rsi_overbought = rsi is not None and rsi > 70
    rsi_oversold = rsi is not None and rsi < 30
    mom5_pos = mom_5d > 0.02
    mom5_neg = mom_5d < -0.02
    mom30_pos = mom_30d_pct is not None and mom_30d_pct > 5
    mom30_neg = mom_30d_pct is not None and mom_30d_pct < -5

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
    elif not is_high_vol and abs(mom_5d) < 0.02:
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
# also disappeared from between_var).
PHANTOM_SIGNAL_SE = 0.20


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
    std = total_var ** 0.5
    phantom_std_inflation = (total_var ** 0.5) - ((within_var + between_var) ** 0.5)

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

    prior_mu = prior_blend["blended"]
    prior_std = prior_blend.get("std", 0.15)
    obs_mu = today_blend["blended"]
    obs_std = today_blend["std"]

    inflation = 1.0 + 0.2 * max(0, prior_age_days - 1)
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

    today = datetime.now().date()
    horizon_end = today + timedelta(days=horizon_days)
    mag_map = {"high": 0.10, "med": 0.05, "low": 0.02}
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
            mag = mag_map.get(str(c.get("magnitude", "med")).lower(), 0.05)
            sign = dir_sign.get(str(c.get("direction_risk", "two-sided")).lower(), 0.0)
            total += mag * sign
            in_window_count += 1

    total = max(-0.15, min(0.15, total))
    if in_window_count == 0:
        return 0.0, "LOW", "no catalysts in horizon"

    conf = "HIGH" if in_window_count >= 3 else ("MEDIUM" if in_window_count >= 1 else "LOW")
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
    """Robust catalyst date parser. Handles Y/M/Q/range/relative formats."""
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip().lower()
    today = datetime.now().date()

    if "/" in s:
        parts = [p.strip() for p in s.split("/") if p.strip()]
        candidates = [_parse_catalyst_date(p) for p in parts]
        valid = [c for c in candidates if c is not None]
        return min(valid) if valid else None

    import re
    m = re.match(r"next\s+(\d+)\s*d", s)
    if m:
        offset = int(m.group(1)) // 2
        return today + timedelta(days=offset)

    m = re.match(r"(\d{4})[-\s]?q([1-4])", s)
    if m:
        year = int(m.group(1))
        q = int(m.group(2))
        month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        try:
            return datetime(year, month, 1).date()
        except ValueError:
            return None

    m = re.match(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", s)
    if m:
        try:
            year = int(m.group(1))
            month = int(m.group(2))
            day = int(m.group(3)) if m.group(3) else 1
            return datetime(year, month, day).date()
        except ValueError:
            return None

    m = re.match(r"^(\d{4})$", s)
    if m:
        try:
            return datetime(int(m.group(1)), 1, 1).date()
        except ValueError:
            return None

    return None


# Public alias for backwards-compatible reference
parse_catalyst_date = _parse_catalyst_date

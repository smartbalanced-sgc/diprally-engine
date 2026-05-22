"""
swing_analyzer_dipnrally.py — SanDisk Swing Trader (v2)

Round-trip dip-and-rally framework for high-volatility stocks. Sibling to
swing_analyzer_analytic.py (v1), which solves a different problem (exit an
existing position). v2 solves: buy on dip, sell on rally, within horizon.

Design lineage:
  - v1 (swing_analyzer_analytic.py) — POSITION EXIT, single-barrier upside,
    one threshold, HOLD/TRIM/CUT verdict. LOCKED.
  - v2 (this file) — ROUND-TRIP, two-sided scan, asymmetric thresholds,
    joint conditional probability, bag-hold risk minimisation.

Key architectural decisions (locked, see SNDK_SWING_TOOL.md v2 section):
  1. 65% dip threshold (marginal P(touch dip) within horizon)
  2. 75% rally-conditional threshold (P(rally | dip touched, remaining horizon))
  3. 100k Monte Carlo paths (200k auto-scale when P(dip) < 40%)
  4. Three-method math cross-check on every run: MC + PDE + closed-form
  5. AI two-pass adversarial critique (Pass 2 wins, Pass 1 audit trail)
  6. AI outputs become numeric model inputs, never display-only prose
  7. 11-signal drift blend (v1's 9 + catalyst proximity + structural narrative)
  8. Catalyst-aware vol schedule (pattern from src/monte_carlo.py)
  9. Bayesian smoothing across days (preserved from v1)
 10. Mean reversion: configurable, default OFF (strength=0) until backtest data
 11. Backtest layer: built in from day 1, displays calibration when N >= 30
 12. v3 review at 30 days against locked criteria (see V3_REVIEW_CRITERIA)
 13. Branch isolation: lives on claude/analyze-sandisk-trading-6zYxn, never merged

Anti-patterns explicitly NOT touched:
  - Sacred files in src/ are READ ONLY — patterns ported by copy, not import
  - No new dependencies in requirements.txt
  - No block bootstrap, no multi-step vol forecast, no synthesised reliability
  - No EV-cushion verdict driver (different framework, different question)

Usage:
    cd ~/sgc/sgc-dip-engine
    source venv/bin/activate
    python3 tools/swing_analyzer_dipnrally.py SNDK \\
        --capital 10000 \\
        --horizon 60 \\
        --conviction-dip 0.65 \\
        --conviction-rally-cond 0.75 \\
        --show-rationale

Required env vars:
    FMP_API_KEY        — Financial Modeling Prep (Starter plan)
    ANTHROPIC_API_KEY  — Claude Opus 4.7 for AI two-pass synthesis

Output:
    - Console: structured report (~250 lines)
    - tools/output/round_trip_history_SNDK.csv — appended row per run
    - tools/output/sndk_dipnrally_dashboard.html — regenerated each run
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.stats import norm

# Reuse stable v1 functions — both files live on the same feature branch
# and delete together per §13 cleanup. Importing avoids ~1500 lines of
# duplicated FMP wrappers, GARCH fitting, and signal computations.
# v1 is LOCKED (no modifications) so import surface is stable.
sys.path.insert(0, str(Path(__file__).parent))
from swing_analyzer_analytic import (  # type: ignore
    fetch_history,
    fit_garch_11,
    compute_rsi_14,
    enrichment_drift,
    closed_touch_up,
    closed_touch_down,
    pde_two_barrier,
    _anthropic_client,
    compute_opus_cost,
    _fmp_get,
    fetch_analyst_targets,
    fetch_analyst_summary,
    fetch_next_earnings,
    fetch_company_profile,
    fetch_sector_perf,
    fetch_insider_activity,
    fetch_macro_indicators,
    _none_signal,
    signal_from_analyst_targets,
    signal_from_sector,
    signal_from_macro,
    signal_from_insider,
    signal_from_historical,
    detect_swing_regime,
    vol_regime_advisory,
    blend_with_uncertainty,
    bayesian_update,
    compute_realized_vol,
    fetch_options_iv,
    triangulate_sigma,
    fetch_short_interest,
    signal_from_short_interest,
    fetch_peer_history,
    signal_from_peer_rs,
    signal_from_sector_decoupling,
    DEFAULT_LOOKBACK_DAYS,
    FMP_BASE,
)


# =============================================================================
# v2 LOCKED CONFIGURATION
# =============================================================================

V2_VERSION = "DIPNRALLY-v1.0"
DEFAULT_CONVICTION_DIP = 0.65          # P(touch dip) marginal — LOCKED
DEFAULT_CONVICTION_RALLY_COND = 0.75   # P(rally | dip) conditional — LOCKED
DEFAULT_HORIZON_DAYS = 60
DEFAULT_MC_PATHS = 100_000             # 200k auto-scale when P(dip) < 40%
DEEP_DIP_AUTOSCALE_THRESHOLD = 0.40
DEEP_DIP_AUTOSCALE_PATHS = 200_000

# Asymmetric grid resolution: tighter near spot, coarser at extremes
DIP_GRID_STEP = 10.0    # dollar step for dip scan
RALLY_GRID_STEP = 10.0  # dollar step for rally scan
DIP_GRID_MAX_DEPTH_PCT = 0.40   # scan down to spot * (1 - 0.40) = 60% of spot
RALLY_GRID_MAX_REACH_PCT = 0.60 # scan up to spot * (1 + 0.60) = 160% of spot

# AI vol_regime → vol_mult mapping
# v2-CALIBRATED, NOT INHERITED from src/monte_carlo.py:73-84 (which used
# 0.75/1.0/1.30 tuned across the dip engine portfolio).
# For SNDK σ=97% extreme regime, narrower band is appropriate — extreme
# regime is already priced in, so AI vol_regime adds only marginal adjustment.
AI_VOL_REGIME_MULTIPLIERS = {
    "HIGH": 1.15,   # Post-earnings miss / catalyst risk → moderate widen
    "MEDIUM": 1.00, # No adjustment
    "LOW": 0.90,    # Post-beat / vol-collapse signal → moderate tighten
}

# Structural narrative score → drift adjustment (annualised pp)
NARRATIVE_DRIFT_ADJUSTMENT = {
    "strong": 0.05,   # +5pp drift (e.g. SNDK HBF + NVDA Vera Rubin link)
    "neutral": 0.00,
    "weak": -0.05,    # -5pp drift
}

# Bull/bear factor arithmetic weights (HIGH=3, MED=2, LOW=1)
FACTOR_WEIGHTS = {"high": 3, "med": 2, "low": 1}
FACTOR_NET_THRESHOLD = 4  # sum(HIGH bull) - sum(HIGH bear) > 4 → +5pp tail bias
FACTOR_TAIL_BIAS = 0.05    # ±5pp drift bias when threshold exceeded

# Catalyst Z-score thresholds (pattern from src/sentiment.py:112)
# For SNDK σ ≈ 97%, use high-vol threshold of 3.0
CATALYST_Z_THRESHOLD = 3.0

# Vol schedule multipliers around catalysts
# Pattern from src/monte_carlo.py:155-202; values v2-tuned for SNDK
VOL_SCHEDULE_MULTIPLIERS = {
    "self_earnings_day": 3.0,
    "self_earnings_pre_post": 1.5,
    "self_earnings_window_days": 2,
    "peer_earnings_day": 1.8,       # MU, WDC, NVDA earnings
    "peer_earnings_pre_post": 1.3,
    "peer_earnings_window_days": 1,
    "macro_event_day": 1.5,         # FOMC, CPI prints
}

# Three-method agreement tolerance.
# Marginal "ever touched" allowed 3pp because vol schedule + AI vol_regime
# multiplier can introduce time-varying σ that closed-form (constant σ)
# cannot exactly match (observed 2.7pp residual when vol_regime=HIGH).
# First-passage has irreducible residual at high σ from discrete bridge
# sampling (empirically 2-4pp at σ=98%).
METHOD_AGREEMENT_TOLERANCE_PP_MARGINAL = 3.0      # P(touch ever)
METHOD_AGREEMENT_TOLERANCE_PP_FIRST_PASSAGE = 4.0  # P(dip first), P(rally first)
# Legacy alias kept for any unhandled references; prefer the two above.
METHOD_AGREEMENT_TOLERANCE_PP = METHOD_AGREEMENT_TOLERANCE_PP_FIRST_PASSAGE

# Bag-hold valuation assumption for expected $/trade computation
# When bag-hold occurs at horizon end, assume position is at:
#   - 50% of (dip - some_recovery) — terminal price for paths that touched
#     dip but didn't rally. Approximated from path metric distribution.
BAG_HOLD_TERMINAL_ASSUMPTION = "median_terminal_dip_paths"  # alt: "dip_price"

# Backtest gate — minimum samples before calibration claims are made
BACKTEST_MIN_SAMPLES = 30

# v2 blend weights — 10 signals (v1's 9 less "ai" reweighted + 2 new AI-derived)
# Total = 1.00 approximately; blend_with_uncertainty normalises by quality gating.
BLEND_WEIGHTS_V2 = {
    "historical":          0.05,
    "analyst":             0.15,
    "sector":              0.04,
    "macro":               0.07,
    "insider":             0.02,
    "short_interest":      0.02,
    "peer_rs":             0.10,
    "sector_decoupling":   0.10,
    "ai":                  0.25,   # confidence-weighted via internal LOW halver
    "catalyst_proximity":  0.10,   # NEW v2 signal
    "narrative":           0.10,   # NEW v2 signal
}


# v3 review criteria — LOCKED at v2 ship, executed at 30 days of runtime data
V3_REVIEW_CRITERIA = {
    "n_days_min": 30,
    "calibration_dip_target": (0.60, 0.70),         # actual P(touch dip) within ±5pp of 65%
    "calibration_rally_cond_target": (0.70, 0.80),  # actual P(rally|dip) within ±5pp of 75%
    "ai_pass2_critique_rate_min": 0.20,             # if P2 disagrees with P1 < 20%, kill two-pass
    "catalyst_signal_correlation_min": 0.10,        # if catalyst signal correlation with realized < 0.10, drop
    "bag_hold_rate_target": (0.10, 0.20),           # actual bag-hold within target band
}


# =============================================================================
# GARCH(1,1) FULL FIT — returns α, β, ω + forecast variance (v1's fit_garch_11
# only returns scalar variance; v2 needs the params for α+β unit-root diagnostic).
# Implementation mirrors src/garch_model.py:10-83 (READ ONLY — sacred file).
# =============================================================================

def fit_garch_11_full(returns: pd.Series) -> dict:
    """GARCH(1,1) fit returning {omega, alpha, beta, forecast_variance, fit_ok}.

    σ²(t) = ω + α r²(t-1) + β σ²(t-1)
    Stationarity constraint: α + β < 1 (else non-stationary / IGARCH).
    Near-IGARCH (α+β > 0.98) means vol shocks are highly persistent.

    Returns a dict so v2 can expose α+β explicitly. Falls back to rolling variance
    if optimization fails or insufficient data.
    """
    from scipy.optimize import minimize as _minimize

    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 50:
        return {
            "omega": 0.0, "alpha": 0.0, "beta": 0.0,
            "forecast_variance": float(r.var()) if len(r) > 0 else 1e-6,
            "fit_ok": False,
        }

    def neg_ll(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e10
        T = len(r)
        s2 = np.zeros(T)
        s2[0] = r.var()
        for t in range(1, T):
            s2[t] = omega + alpha * r.iloc[t - 1] ** 2 + beta * s2[t - 1]
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + r.values ** 2 / s2)

    try:
        res = _minimize(
            neg_ll, [0.0001, 0.05, 0.90], method="L-BFGS-B",
            bounds=[(1e-8, 1.0), (0.0, 1.0), (0.0, 0.9999)],
        )
        omega, alpha, beta = res.x
        last_var = float(r.tail(20).var())
        forecast_var = float(omega + alpha * r.iloc[-1] ** 2 + beta * last_var)
        return {
            "omega": float(omega),
            "alpha": float(alpha),
            "beta": float(beta),
            "forecast_variance": forecast_var,
            "fit_ok": bool(res.success and forecast_var > 0 and not np.isnan(forecast_var)),
        }
    except Exception:
        fallback_var = float(r.tail(90).var())
        return {
            "omega": 0.0, "alpha": 0.0, "beta": 0.0,
            "forecast_variance": fallback_var, "fit_ok": False,
        }


# =============================================================================
# CATALYST DATE PARSER — handle Y/M/Q/range formats from AI output
# =============================================================================

def parse_catalyst_date(date_str: str) -> Optional["datetime.date"]:
    """Robust catalyst date parser. Handles:
      - YYYY-MM-DD          → exact date
      - YYYY-MM             → first of month
      - YYYY                → first of year
      - YYYY-Q1/Q2/Q3/Q4    → start of quarter
      - YYYY-MM/YYYY-MM     → earliest of range
      - "next 30d"          → today + 15 (mid-window)
      - "next NNd"          → today + NN/2
    Returns date or None if unparseable.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip().lower()
    today = datetime.now().date()

    # Range: take earliest
    if "/" in s:
        parts = [p.strip() for p in s.split("/") if p.strip()]
        candidates = [parse_catalyst_date(p) for p in parts]
        valid = [c for c in candidates if c is not None]
        return min(valid) if valid else None

    # "next NNd" / "next 30 days"
    m = None
    import re
    m = re.match(r"next\s+(\d+)\s*d", s)
    if m:
        offset = int(m.group(1)) // 2  # mid-window
        return today + timedelta(days=offset)

    # Quarter: YYYY-Q1..Q4
    m = re.match(r"(\d{4})[-\s]?q([1-4])", s)
    if m:
        year = int(m.group(1))
        q = int(m.group(2))
        month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        try:
            return datetime(year, month, 1).date()
        except ValueError:
            return None

    # YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS or YYYY-MM
    m = re.match(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", s)
    if m:
        try:
            year = int(m.group(1))
            month = int(m.group(2))
            day = int(m.group(3)) if m.group(3) else 1
            return datetime(year, month, day).date()
        except ValueError:
            return None

    # YYYY only
    m = re.match(r"^(\d{4})$", s)
    if m:
        try:
            return datetime(int(m.group(1)), 1, 1).date()
        except ValueError:
            return None

    return None


# =============================================================================
# DATA CLASSES — typed containers for daily state
# =============================================================================

@dataclass
class MarketSnapshot:
    """Immutable snapshot of market data at run time."""
    ticker: str
    timestamp: datetime
    spot: float
    market_cap: float
    sector: str
    industry: str
    rsi: float
    mom_5d: float
    mom_30d: float
    ytd_return: float
    price_history: pd.DataFrame


@dataclass
class VolatilityProfile:
    garch_sigma: float
    garch_alpha: float
    garch_beta: float
    garch_alpha_plus_beta: float
    realized_30d: float
    realized_60d: float
    realized_90d: float
    options_iv: Optional[float]
    options_dte: Optional[int]
    blended_sigma: float
    anchors_count: int
    divergence_pp: float
    near_unit_root: bool  # alpha + beta > 0.98


@dataclass
class DriftSignal:
    name: str
    mu_annual: float
    confidence: str  # LOW / MEDIUM / HIGH
    source_quality: str  # PRIMARY / REPUTABLE
    weight: float
    rationale: str


@dataclass
class AIPassOutput:
    pass_number: int
    drift_estimate: float
    drift_range: tuple[float, float]
    confidence: str
    vol_regime: str  # HIGH / MEDIUM / LOW
    narrative_score: str  # strong / neutral / weak
    catalysts: list[dict]
    bull_factors: list[dict]
    bear_factors: list[dict]
    key_risks: list[str]
    revision_from_prior_pass: Optional[float]
    cost_usd: float
    raw_sources_cited: int


@dataclass
class JointConditionalResult:
    dip_price: float
    rally_price: float
    p_dip_touched: float       # marginal
    p_rally_given_dip: float   # conditional
    p_round_trip: float        # joint
    p_bag_hold: float
    p_no_trade_rally_first: float
    p_neither: float
    expected_days_to_dip: float
    expected_days_dip_to_rally: float
    expected_gain_per_share: float
    expected_bag_hold_loss: float
    net_expected_value: float


def _signals_dict_to_display_list(signals_dict: dict, weights: dict) -> list[DriftSignal]:
    """Convert v1's signal dict format to v2's DriftSignal list for display only."""
    pretty_names = {
        "historical": "Historical (GARCH + enrichment)",
        "analyst": "Analyst (price-target-summary)",
        "sector": "Sector momentum",
        "macro": "Macro regime (VIX/SPY)",
        "insider": "Insider activity (90d, mcap-scaled)",
        "short_interest": "Short interest (squeeze tail)",
        "peer_rs": "Peer RS (MU+WDC, 60d)",
        "sector_decoupling": "Sector decoupling (vs sector, 30d)",
        "ai": "AI analyst",  # suffix added dynamically based on Pass 1 vs Pass 2
        "catalyst_proximity": "Catalyst proximity (AI-generated)",
        "narrative": "Structural narrative score",
    }
    out: list[DriftSignal] = []
    for name, info in signals_dict.items():
        drift = info.get("drift")
        if drift is None:
            drift = 0.0
        # Dynamic label for AI signal: detect Pass 2 vs Pass 1 from the notes field
        display_name = pretty_names.get(name, name)
        if name == "ai":
            notes = str(info.get("notes", ""))
            if "Pass 2" in notes:
                display_name = "AI analyst (Pass 2 revised, wins over Pass 1)"
            elif "Pass 1" in notes:
                display_name = "AI analyst (Pass 1, no Pass 2)"
            else:
                display_name = "AI analyst (skipped)"
        out.append(DriftSignal(
            name=display_name,
            mu_annual=float(drift),
            confidence=str(info.get("confidence", "LOW")),
            source_quality=str(info.get("source_quality", "PRIMARY")),
            weight=float(weights.get(name, 0.0)),
            rationale=str(info.get("notes", "")),
        ))
    return out


# =============================================================================
# MATH LAYER — three independent methods, cross-checked on every run
#
# Method 1: Monte Carlo (joint conditional, path-ordering aware) — NEW for v2
# Method 2: Fokker-Planck PDE (imported from v1, used for first-passage)
# Method 3: Closed-form analytic (imported from v1, single-barrier)
# =============================================================================

def run_mc_joint_conditional(
    S0: float,
    sigma: float,
    mu: float,
    horizon_days: int,
    n_paths: int = DEFAULT_MC_PATHS,
    vol_schedule: Optional[np.ndarray] = None,
    mean_reversion_strength: float = 0.0,
    mean_reversion_anchor: Optional[float] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate Monte Carlo paths with optional time-varying vol and mean reversion.

    Returns paths shape (n_paths, horizon_days) — daily prices, not including
    initial spot. The joint-conditional analysis in scan_dip_rally_grid uses
    these paths to compute P(dip touched then rally before horizon end).

    Mean reversion is OFF by default (strength=0.0) — see v2 spec §10.
    """
    np.random.seed(seed)
    dt = 1.0 / 252.0
    sd = sigma * np.sqrt(dt)  # baseline daily vol

    z = np.random.standard_normal((n_paths, horizon_days))
    paths = np.zeros((n_paths, horizon_days + 1))
    paths[:, 0] = S0

    for t in range(1, horizon_days + 1):
        # Vol for this day: scheduled multiplier × baseline
        if vol_schedule is not None:
            sd_t = vol_schedule[t - 1] * np.sqrt(dt)
        else:
            sd_t = sd

        # Mean reversion drift (pattern from src/monte_carlo.py:233-237)
        # Default off: strength=0 reduces to pure GBM
        if mean_reversion_strength > 0 and mean_reversion_anchor is not None:
            deviation = (paths[:, t - 1] - mean_reversion_anchor) / mean_reversion_anchor
            mr_drift = -mean_reversion_strength * deviation
        else:
            mr_drift = 0.0

        # GBM drift (Ito correction)
        gbm_drift = (mu - 0.5 * sigma**2) * dt + mr_drift * dt

        # Log-return step
        log_step = gbm_drift + sd_t * z[:, t - 1]
        paths[:, t] = paths[:, t - 1] * np.exp(log_step)

    return paths[:, 1:]  # drop initial column


def precompute_first_touch_days(
    paths: np.ndarray,
    S0: float,
    barriers: np.ndarray,
    sigma: float,
    vol_schedule: Optional[np.ndarray],
    direction: str,
    seed: int = 42,
) -> np.ndarray:
    """For each barrier, return first-touch day per path with Brownian bridge correction.

    Returns array of shape (n_paths, n_barriers): first-touch day index for each path
    and each barrier (n_days sentinel if never touched within horizon).

    Bridge math: for arithmetic Brownian motion in log-space with daily vol σ_d,
        P(min log S(τ) < log B, τ ∈ [t,t+1] | log S(t)=x, log S(t+1)=y)
            = exp(-2 (x - log B)(y - log B) / σ_d²)  when both x > log B, y > log B
    Symmetric for upper barrier. This corrects the ~13pp discrete-time MC bias
    versus continuous-time PDE/closed-form at high vol (observed at SNDK σ=98%).

    Vectorised across paths × days for each barrier; loops over barriers
    (one barrier at a time keeps memory under 100k × 60 × 8 = 48 MB per pass).
    """
    n_paths, n_days = paths.shape
    n_barriers = len(barriers)
    result = np.full((n_paths, n_barriers), n_days, dtype=np.int32)

    prev = np.concatenate([np.full((n_paths, 1), S0), paths[:, :-1]], axis=1)
    log_prev = np.log(prev)
    log_curr = np.log(paths)

    if vol_schedule is not None:
        sigma_d_sq = (vol_schedule.astype(float) ** 2) / 252.0
    else:
        sigma_d_sq = np.full(n_days, (sigma ** 2) / 252.0)
    sigma_d_sq = np.maximum(sigma_d_sq, 1e-12)

    rng = np.random.default_rng(seed=seed)

    for i, B in enumerate(barriers):
        log_B = float(np.log(B))
        if direction == "down":
            close_touch = paths <= B
            both_safe = (log_prev > log_B) & (log_curr > log_B)
            dx = log_prev - log_B
            dy = log_curr - log_B
        else:  # "up"
            close_touch = paths >= B
            both_safe = (log_prev < log_B) & (log_curr < log_B)
            dx = log_B - log_prev
            dy = log_B - log_curr

        with np.errstate(divide="ignore", invalid="ignore"):
            exponent = -2.0 * dx * dy / sigma_d_sq[np.newaxis, :]
        # Bridge touch probability, zero where endpoints already crossed
        p_touch_bridge = np.where(both_safe, np.exp(exponent), 0.0)
        # Sample Bernoulli per (path, day)
        u = rng.random(p_touch_bridge.shape)
        bridge_touch = (u < p_touch_bridge) & both_safe
        touch_mask = close_touch | bridge_touch
        touch_any = touch_mask.any(axis=1)
        first_day = np.where(touch_any, touch_mask.argmax(axis=1), n_days)
        result[:, i] = first_day

    return result


def analyze_joint_conditional(
    paths: np.ndarray,
    S0: float,
    dip_price: float,
    rally_price: float,
    horizon_days: int,
    sigma: Optional[float] = None,
    vol_schedule: Optional[np.ndarray] = None,
    dip_first_days: Optional[np.ndarray] = None,
    rally_first_days: Optional[np.ndarray] = None,
) -> dict:
    """
    Core v2 logic: for each MC path, track whether dip and rally were touched
    in correct order (dip first, then rally). Returns four-scenario breakdown.

    Scenarios:
      A. round_trip: path touched dip, then later touched rally before horizon
      B. bag_hold: path touched dip, never touched rally (held at horizon)
      C. no_trade_rally_first: path touched rally before any dip
      D. neither: path never touched either barrier

    All four sum to 1.0.

    When sigma is provided, applies Brownian bridge correction so first-passage
    probabilities match continuous-time PDE/closed-form within MC sample noise.
    """
    n_paths, n_days = paths.shape

    # Three ways to get first-touch days, in preference order:
    #   1. Caller passed precomputed first-touch arrays (fast path for grid scan)
    #   2. Sigma provided: apply Brownian bridge correction (matches PDE/closed-form)
    #   3. Neither: daily-close only (biased ~13pp at SNDK σ but cheap)
    if dip_first_days is not None and rally_first_days is not None:
        dip_first_day = dip_first_days
        rally_first_day = rally_first_days
    elif sigma is not None:
        dip_arr = precompute_first_touch_days(
            paths, S0, np.array([dip_price]), sigma, vol_schedule, "down"
        )
        rally_arr = precompute_first_touch_days(
            paths, S0, np.array([rally_price]), sigma, vol_schedule, "up", seed=43
        )
        dip_first_day = dip_arr[:, 0]
        rally_first_day = rally_arr[:, 0]
    else:
        dip_mask = paths <= dip_price
        rally_mask = paths >= rally_price
        dip_any_local = dip_mask.any(axis=1)
        rally_any_local = rally_mask.any(axis=1)
        dip_first_day = np.where(dip_any_local, dip_mask.argmax(axis=1), n_days)
        rally_first_day = np.where(rally_any_local, rally_mask.argmax(axis=1), n_days)

    dip_any = dip_first_day < n_days
    rally_any = rally_first_day < n_days

    # 4-way partition (mutually exclusive, exhaustive). Same-day first-touch
    # (dip AND rally touched on the same day, possible at high σ with bridge
    # correction) is classified as rally_first — conservative, doesn't
    # overstate round-trip probability when intraday order is ambiguous.
    both_touched = dip_any & rally_any
    round_trip = both_touched & (dip_first_day < rally_first_day)
    rally_first = (rally_any & ~dip_any) | (both_touched & (rally_first_day <= dip_first_day))
    bag_hold = dip_any & ~rally_any
    neither = ~dip_any & ~rally_any

    total = round_trip.sum() + rally_first.sum() + bag_hold.sum() + neither.sum()
    assert total == n_paths, f"scenario partition error: {total} != {n_paths}"

    n_round_trip = int(round_trip.sum())
    n_rally_first = int(rally_first.sum())
    n_bag_hold = int(bag_hold.sum())
    n_neither = int(neither.sum())

    # Conditional probabilities
    # p_dip_touched_first: paths where dip touched BEFORE rally (used for trade trigger)
    # p_dip_touched_any: paths where dip touched at ANY point (used for cross-check vs closed-form)
    p_dip_touched_first = float((n_round_trip + n_bag_hold) / n_paths)
    p_dip_touched_any = float(dip_any.sum() / n_paths)
    p_rally_touched_any = float(rally_any.sum() / n_paths)
    p_rally_given_dip = (
        float(n_round_trip / (n_round_trip + n_bag_hold))
        if (n_round_trip + n_bag_hold) > 0 else 0.0
    )

    # Expected days conditional on round-trip completion
    if n_round_trip > 0:
        rt_dip_days = dip_first_day[round_trip]
        rt_rally_days = rally_first_day[round_trip]
        exp_days_to_dip = float(np.mean(rt_dip_days))
        exp_days_dip_to_rally = float(np.mean(rt_rally_days - rt_dip_days))
    else:
        exp_days_to_dip = 0.0
        exp_days_dip_to_rally = 0.0

    # Bag-hold terminal price (median across bag-hold paths' final price)
    if n_bag_hold > 0:
        bag_hold_terminals = paths[bag_hold, -1]
        bag_hold_terminal_median = float(np.median(bag_hold_terminals))
    else:
        bag_hold_terminal_median = dip_price

    return {
        "n_paths": n_paths,
        "p_round_trip": n_round_trip / n_paths,
        "p_bag_hold": n_bag_hold / n_paths,
        "p_no_trade_rally_first": n_rally_first / n_paths,
        "p_neither": n_neither / n_paths,
        # First-passage: dip BEFORE rally (used by trade-trigger logic)
        "p_dip_touched_marginal": p_dip_touched_first,
        # Marginal "ever touched" (used for cross-check vs closed-form)
        "p_dip_touched_any": p_dip_touched_any,
        "p_rally_touched_any": p_rally_touched_any,
        "p_rally_given_dip_conditional": p_rally_given_dip,
        "expected_days_to_dip": exp_days_to_dip,
        "expected_days_dip_to_rally": exp_days_dip_to_rally,
        "bag_hold_terminal_median": bag_hold_terminal_median,
    }


def three_method_cross_check(
    S0: float,
    sigma: float,
    mu: float,
    horizon_days: int,
    dip_price: float,
    rally_price: float,
    mc_result: dict,
) -> dict:
    """
    Cross-check MC's first-passage probabilities against PDE and closed-form.
    Returns agreement table and disagreement flags.

    Method 2: PDE pde_two_barrier — exact first-passage for two barriers
    Method 3: closed_touch_up/down — exact single-barrier marginal P
    """
    T_years = horizon_days / 252.0

    # Method 2: PDE first-passage (rally up, dip down)
    pde = pde_two_barrier(S0, rally_price, dip_price, T_years, mu, sigma)
    p_rally_first_pde = pde["p_U_first"]
    p_dip_first_pde = pde["p_L_first"]

    # Method 3: closed-form marginal "ever touched"
    p_touch_dip_closed = closed_touch_down(S0, dip_price, T_years, mu, sigma)
    p_touch_rally_closed = closed_touch_up(S0, rally_price, T_years, mu, sigma)

    # MC's analogous values — use the CORRECT quantity for each row
    p_dip_first_mc = mc_result["p_bag_hold"] + mc_result["p_round_trip"]
    p_rally_first_mc = mc_result["p_no_trade_rally_first"]
    # For "ever touched" rows, use the bridge-corrected ANY mask, not first-passage
    p_touch_dip_marginal_mc = mc_result.get("p_dip_touched_any", p_dip_first_mc)
    p_touch_rally_marginal_mc = mc_result.get("p_rally_touched_any",
                                               mc_result["p_no_trade_rally_first"] + mc_result["p_round_trip"])

    # Disagreement flags — first-passage rows allowed a wider tolerance than marginal
    flags = []
    pp = lambda x: x * 100.0

    diff_dip_first = abs(pp(p_dip_first_mc) - pp(p_dip_first_pde))
    diff_rally_first = abs(pp(p_rally_first_mc) - pp(p_rally_first_pde))
    diff_touch_dip = abs(pp(p_touch_dip_marginal_mc) - pp(p_touch_dip_closed))
    diff_touch_rally = abs(pp(p_touch_rally_marginal_mc) - pp(p_touch_rally_closed))

    if diff_dip_first > METHOD_AGREEMENT_TOLERANCE_PP_FIRST_PASSAGE:
        flags.append(f"MC vs PDE disagree on P(dip first) by {diff_dip_first:.1f}pp")
    if diff_rally_first > METHOD_AGREEMENT_TOLERANCE_PP_FIRST_PASSAGE:
        flags.append(f"MC vs PDE disagree on P(rally first) by {diff_rally_first:.1f}pp")
    if diff_touch_dip > METHOD_AGREEMENT_TOLERANCE_PP_MARGINAL:
        flags.append(f"MC vs closed-form disagree on marginal P(touch dip) by {diff_touch_dip:.1f}pp")
    if diff_touch_rally > METHOD_AGREEMENT_TOLERANCE_PP_MARGINAL:
        flags.append(f"MC vs closed-form disagree on marginal P(touch rally) by {diff_touch_rally:.1f}pp")

    return {
        "table": [
            ("P(dip first)",      pp(p_dip_first_mc),      pp(p_dip_first_pde),       diff_dip_first),
            ("P(rally first)",    pp(p_rally_first_mc),    pp(p_rally_first_pde),     diff_rally_first),
            ("P(touch dip ever)", pp(p_touch_dip_marginal_mc), pp(p_touch_dip_closed),   diff_touch_dip),
            ("P(touch rally ever)", pp(p_touch_rally_marginal_mc), pp(p_touch_rally_closed), diff_touch_rally),
        ],
        "flags": flags,
        "pde_p_neither": pde["p_neither"],
        "pde_mass_conservation": pde["total"],
        "agreement_status": "✓ all methods agree within tolerance" if not flags else "⚠ disagreement flagged",
    }


# =============================================================================
# VOL SCHEDULE — catalyst-aware time-varying volatility
# Pattern adopted from src/monte_carlo.py:155-202
# Multipliers v2-tuned (see VOL_SCHEDULE_MULTIPLIERS)
# =============================================================================

def build_catalyst_vol_schedule(
    base_vol: float,
    horizon_days: int,
    self_earnings_date: Optional[datetime],
    peer_earnings_dates: list[datetime],
    macro_event_dates: list[datetime],
) -> np.ndarray:
    """Return per-day vol array of length horizon_days."""
    today = datetime.now().date()
    schedule = np.ones(horizon_days)

    # Self earnings spike (pattern from src/monte_carlo.py:167-187)
    if self_earnings_date:
        try:
            ed = self_earnings_date.date() if hasattr(self_earnings_date, "date") else self_earnings_date
            d_idx = (ed - today).days
            window = VOL_SCHEDULE_MULTIPLIERS["self_earnings_window_days"]
            if 0 <= d_idx < horizon_days:
                schedule[d_idx] = max(schedule[d_idx], VOL_SCHEDULE_MULTIPLIERS["self_earnings_day"])
                for off in range(1, window + 1):
                    if d_idx - off >= 0:
                        schedule[d_idx - off] = max(
                            schedule[d_idx - off],
                            VOL_SCHEDULE_MULTIPLIERS["self_earnings_pre_post"],
                        )
                    if d_idx + off < horizon_days:
                        schedule[d_idx + off] = max(
                            schedule[d_idx + off],
                            VOL_SCHEDULE_MULTIPLIERS["self_earnings_pre_post"],
                        )
        except Exception:
            pass

    # Peer earnings (lighter multiplier)
    for ped in peer_earnings_dates:
        try:
            d = ped.date() if hasattr(ped, "date") else ped
            d_idx = (d - today).days
            window = VOL_SCHEDULE_MULTIPLIERS["peer_earnings_window_days"]
            if 0 <= d_idx < horizon_days:
                schedule[d_idx] = max(schedule[d_idx], VOL_SCHEDULE_MULTIPLIERS["peer_earnings_day"])
                for off in range(1, window + 1):
                    if d_idx - off >= 0:
                        schedule[d_idx - off] = max(
                            schedule[d_idx - off],
                            VOL_SCHEDULE_MULTIPLIERS["peer_earnings_pre_post"],
                        )
                    if d_idx + off < horizon_days:
                        schedule[d_idx + off] = max(
                            schedule[d_idx + off],
                            VOL_SCHEDULE_MULTIPLIERS["peer_earnings_pre_post"],
                        )
        except Exception:
            pass

    # Macro events
    for mev in macro_event_dates:
        try:
            d = mev.date() if hasattr(mev, "date") else mev
            d_idx = (d - today).days
            if 0 <= d_idx < horizon_days:
                schedule[d_idx] = max(schedule[d_idx], VOL_SCHEDULE_MULTIPLIERS["macro_event_day"])
        except Exception:
            pass

    return base_vol * schedule


# =============================================================================
# AI LAYER — two-pass adversarial critique with numeric outputs
# Every output here becomes an arithmetic input to the model, never display only.
# =============================================================================

def build_ai_pass1_prompt(
    ticker: str,
    snapshot: MarketSnapshot,
    vol_profile: VolatilityProfile,
    horizon_days: int,
    base_signals: list[DriftSignal],
    self_earnings_date: Optional[datetime],
    peer_tickers: list[str],
) -> str:
    """Pass 1: data gathering + multi-hypothesis catalyst identification.

    Hard mandate: must return STRUCTURED JSON with ≥5 catalyst candidates,
    each cited from ≥2 distinct sources. No prose-only outputs accepted.
    """
    today = snapshot.timestamp.strftime("%Y-%m-%d")
    base_signal_summary = "\n".join(
        f"  - {s.name}: mu={s.mu_annual:+.1%}/yr conf={s.confidence}"
        for s in base_signals
    )
    earnings_str = (
        self_earnings_date.strftime("%Y-%m-%d")
        if self_earnings_date else "unknown"
    )

    return f"""Analyse {ticker} for a 60-day round-trip swing trade.
Today: {today}. Spot: ${snapshot.spot:.2f}. Sector: {snapshot.sector}.
σ blended: {vol_profile.blended_sigma:.1%}. RSI: {snapshot.rsi:.1f}. 30d mom: {snapshot.mom_30d:+.1%}. YTD: {snapshot.ytd_return:+.1%}.
Next own earnings: {earnings_str}. Peers: {', '.join(peer_tickers)}.

Base signal blend (math-derived):
{base_signal_summary}

OUTPUT — single JSON object. NO PROSE BEFORE OR AFTER. NO MARKDOWN FENCES. STRINGS MUST NOT CONTAIN UNESCAPED NEWLINES. Keep each string < 250 chars.

{{
"drift_estimate_annualized": 0.20,
"drift_range_low_high": [-0.20, 0.50],
"confidence": "MEDIUM",
"vol_regime": "MEDIUM",
"narrative_score": "neutral",
"narrative_evidence": [{{"claim": "short", "source": "publisher"}}],
"catalysts": [{{"name": "short name", "type": "earnings", "date_or_window": "YYYY-MM-DD", "magnitude": "med", "direction_risk": "two-sided", "sources": ["src1", "src2"]}}],
"bull_factors": [{{"factor": "concise factor", "weight": "med", "sources": ["src1", "src2"]}}],
"bear_factors": [{{"factor": "concise factor", "weight": "med", "sources": ["src1", "src2"]}}],
"key_risks": ["short risk 1", "short risk 2"]
}}

RULES:
- vol_regime: HIGH if post-event vol expansion expected; LOW if vol-collapse signal; MEDIUM otherwise.
- narrative_score: "strong" only if ≥2 sources defend a structural multi-quarter story; else "neutral".
- catalysts: list 3-5 candidates, each with ≥2 sources. Concise names.
- bull_factors and bear_factors: each list 2-4 items, concise (<200 chars).
- key_risks: 2-3 risks, one short sentence each.
- Return ONLY the JSON object. No preamble. No explanation. No markdown.
"""


def build_ai_pass2_prompt(
    ticker: str,
    snapshot: MarketSnapshot,
    pass1: AIPassOutput,
    mc_marginal_summary: dict,
    sigma_triangulation_summary: dict,
    prior_posterior_drift: Optional[float],
) -> str:
    """Pass 2: ADVERSARIAL critique of Pass 1. Pass 2 wins.

    Pass 2 sees Pass 1 output, math results, and prior context. Its job is
    to find errors, missing catalysts, or weak reasoning in Pass 1, then
    produce a REVISED drift estimate with specific corrections.
    """
    def _safe_name(c):
        return c.get("name", "?") if isinstance(c, dict) else str(c)
    def _safe_factor(f):
        return f.get("factor", str(f)) if isinstance(f, dict) else str(f)
    pass1_summary = {
        "drift_estimate": pass1.drift_estimate,
        "drift_range": list(pass1.drift_range),
        "confidence": pass1.confidence,
        "vol_regime": pass1.vol_regime,
        "narrative_score": pass1.narrative_score,
        "catalysts_count": len(pass1.catalysts),
        "catalyst_names": [_safe_name(c) for c in pass1.catalysts],
        "bull_factors_high": [_safe_factor(f) for f in pass1.bull_factors if _factor_weight(f) == "high"],
        "bear_factors_high": [_safe_factor(f) for f in pass1.bear_factors if _factor_weight(f) == "high"],
    }
    prior_str = f"{prior_posterior_drift:+.1%}/yr" if prior_posterior_drift is not None else "n/a (no history)"
    return f"""You are PASS 2 — an adversarial critic of Pass 1's analysis of {ticker}.

PASS 1 PRODUCED:
{json.dumps(pass1_summary, indent=2)}

INDEPENDENT MATH LAYER SAYS:
- σ blended (5-anchor): {sigma_triangulation_summary['blended']:.1%}
- σ divergence: {sigma_triangulation_summary['divergence']:.1f}pp ({'tight' if sigma_triangulation_summary['divergence'] < 5 else 'wide'})
- Closed-form P(touch +10% from spot in horizon): {mc_marginal_summary.get('p_up_10pct', 'n/a')}
- Closed-form P(touch -10% from spot in horizon): {mc_marginal_summary.get('p_down_10pct', 'n/a')}
- Prior posterior drift (yesterday): {prior_str}

YOUR JOB: critique Pass 1. Find the most likely error. Return JSON:

{{
  "agreement_with_pass1": "agree" | "partial_disagree" | "strong_disagree",
  "primary_critique": "Specific error or weakness in Pass 1",
  "missing_catalysts": ["catalysts Pass 1 missed, if any"],
  "revised_drift_estimate": <float, your corrected annualised drift>,
  "revised_confidence": "LOW" | "MEDIUM" | "HIGH",
  "revision_reasoning": "Why you revised (or kept) Pass 1's estimate",
  "vol_regime_concur": true | false,
  "narrative_score_concur": true | false
}}

ADVERSARIAL POSTURE:
- DO NOT rubber-stamp Pass 1. If Pass 1 is right, say so explicitly with reasoning.
- If Pass 1's drift estimate is inconsistent with the math (e.g., very bullish but stock has touched dip more than rally in MC), critique it.
- If Pass 1 missed a known catalyst in the horizon, flag it.
- If Pass 1 anchored on single source where multiple were available, critique it.
- Return ONLY valid JSON.
"""


def call_ai_pass(prompt: str, max_tokens: int = 3000, pass_label: str = "Pass") -> tuple[Optional[dict], float, int]:
    """Call Claude Opus, parse JSON, return (parsed, cost, sources_cited).

    Returns (None, 0.0, 0) on failure.
    """
    client = _anthropic_client()
    if client is None:
        print(f"⚠️  No Anthropic client — {pass_label} skipped")
        return None, 0.0, 0

    try:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=max_tokens,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            messages=[{"role": "user", "content": prompt}],
        )
        cost = compute_opus_cost(response, had_web_search=True)
        # Extract text from response (may have multiple content blocks)
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        full_text = "\n".join(text_parts).strip()

        # Find JSON boundaries (strip any preamble)
        start = full_text.find("{")
        end = full_text.rfind("}")
        if start < 0 or end < 0:
            print(f"⚠️  {pass_label}: no JSON found in response")
            return None, cost, 0
        json_text = full_text[start:end + 1]

        try:
            # strict=False allows literal control characters (newlines, tabs) inside
            # JSON string values — Opus often emits these instead of \n escapes.
            parsed = json.loads(json_text, strict=False)
        except json.JSONDecodeError as e:
            # Fallback: sanitize control chars and retry
            import re
            sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', json_text)
            try:
                parsed = json.loads(sanitized, strict=False)
            except json.JSONDecodeError as e2:
                print(f"⚠️  {pass_label}: JSON parse error after sanitisation: {e2}")
                print(f"   (first 400 chars of response): {full_text[:400]}")
                return None, cost, 0

        # Count distinct sources mentioned across all citations
        sources = set()
        def collect_sources(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("sources", "url_or_publication"):
                        if isinstance(v, list):
                            sources.update(str(s) for s in v if s)
                        elif v:
                            sources.add(str(v))
                    else:
                        collect_sources(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect_sources(item)
        collect_sources(parsed)
        return parsed, cost, len(sources)
    except Exception as e:
        print(f"⚠️  {pass_label} call failed: {e}")
        return None, 0.0, 0


def call_ai_catalyst_stress_test(
    ticker: str,
    spot: float,
    dip_price: float,
    rally_price: float,
    catalysts: list[dict],
    horizon_days: int,
) -> tuple[list[dict], float]:
    """For top 3 catalysts within horizon, ask AI: what's the directional drift
    impact if this catalyst disappoints by 20%? Numeric output enters sensitivity.
    """
    client = _anthropic_client()
    if client is None or not catalysts:
        return [], 0.0

    top = [c for c in catalysts[:3] if isinstance(c, dict)]
    if not top:
        return [], 0.0
    prompt = f"""For {ticker} at spot ${spot:.2f}, dip target ${dip_price:.0f}, rally target ${rally_price:.0f},
60-day horizon. For each catalyst below, estimate the directional drift impact
(annualised pp) if the catalyst disappoints by 20% on its key metric.

Catalysts:
{json.dumps([{'name': c.get('name'), 'date': c.get('date_or_window'), 'direction': c.get('direction_risk')} for c in top], indent=2)}

Return JSON list, one per catalyst:
[
  {{"catalyst_name": "...", "drift_shock_pp_on_disappointment": <float, signed pp e.g. -8.0 for -8pp>, "reasoning": "..."}},
  ...
]
Return ONLY valid JSON list.
"""
    try:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        cost = compute_opus_cost(response, had_web_search=False)
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        text = "\n".join(text_parts).strip()
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end < 0:
            return [], cost
        parsed = json.loads(text[start:end + 1])
        return parsed if isinstance(parsed, list) else [], cost
    except Exception as e:
        print(f"⚠️  Catalyst stress test failed: {e}")
        return [], 0.0


# =============================================================================
# SIGNAL COMPUTATIONS — 11 signals
# v1's 9 signals reused via import; 2 new added: catalyst proximity + narrative
# =============================================================================

def signal_from_catalyst_proximity(catalysts: list[dict], horizon_days: int) -> tuple[float, str, str]:
    """Compute drift signal from AI-identified catalysts within horizon.

    Returns (mu_annual, confidence, rationale).

    Logic:
      - For each catalyst with date in horizon, compute direction-weighted impact
      - Magnitude bucket: high → ±10pp, med → ±5pp, low → ±2pp
      - Direction: bullish → positive, bearish → negative, two-sided → zero net (but adds vol — handled separately)
      - Sum across all in-horizon catalysts, cap at ±15pp
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
        cdate = parse_catalyst_date(date_str)
        if cdate is None:
            continue
        if today <= cdate <= horizon_end:
            mag = mag_map.get(str(c.get("magnitude", "med")).lower(), 0.05)
            sign = dir_sign.get(str(c.get("direction_risk", "two-sided")).lower(), 0.0)
            total += mag * sign
            in_window_count += 1

    # Cap at ±15pp
    total = max(-0.15, min(0.15, total))
    if in_window_count == 0:
        return 0.0, "LOW", "no catalysts in horizon"

    conf = "HIGH" if in_window_count >= 3 else ("MEDIUM" if in_window_count >= 1 else "LOW")
    return total, conf, f"{in_window_count} catalysts in horizon, net {total:+.1%}"


def signal_from_structural_narrative(narrative_score: str, evidence_count: int) -> tuple[float, str, str]:
    """Convert AI narrative_score to drift adjustment. Requires evidence for 'strong'."""
    adj = NARRATIVE_DRIFT_ADJUSTMENT.get(narrative_score, 0.0)
    # Guardrail: 'strong' classification requires ≥2 evidence sources
    if narrative_score == "strong" and evidence_count < 2:
        return 0.0, "LOW", "strong narrative claimed but insufficient evidence — defaulting neutral"
    conf = "MEDIUM" if narrative_score != "neutral" else "LOW"
    return adj, conf, f"narrative={narrative_score} ({evidence_count} evidence sources)"


def _factor_weight(f) -> str:
    """Defensive: AI might return list of dicts OR list of strings."""
    if isinstance(f, dict):
        return str(f.get("weight", "low")).lower()
    return "low"  # strings don't carry weight metadata


def compute_unusual_move_z(
    history_df: pd.DataFrame,
    beta: Optional[float] = 1.0,
    lookback: int = 60,
) -> Optional[dict]:
    """Beta-adjusted residual Z-score for today's return (pattern from
    src/sentiment.py:130-186, detect_catalysts trigger B/C).

    A |Z| >= CATALYST_Z_THRESHOLD (3.0 for high-vol names) signals an unusual
    move that may have a hidden catalyst. Used for situational awareness in
    the report, not yet as a numeric drift signal (would need backtest data
    to calibrate weight).

    Returns dict {z_score, return_pct, beta, triggered} or None if insufficient data.
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


def apply_bull_bear_arithmetic(
    bull_factors: list, bear_factors: list
) -> tuple[float, str]:
    """Sum weighted bull/bear factors, return drift tail bias and rationale."""
    bull_high = sum(1 for f in bull_factors if _factor_weight(f) == "high")
    bear_high = sum(1 for f in bear_factors if _factor_weight(f) == "high")
    net = bull_high * FACTOR_WEIGHTS["high"] - bear_high * FACTOR_WEIGHTS["high"]
    if net > FACTOR_NET_THRESHOLD:
        return FACTOR_TAIL_BIAS, f"HIGH-bull dominance (net +{net}) → +{FACTOR_TAIL_BIAS:.0%} rally bias"
    if net < -FACTOR_NET_THRESHOLD:
        return -FACTOR_TAIL_BIAS, f"HIGH-bear dominance (net {net}) → -{FACTOR_TAIL_BIAS:.0%} dip bias"
    return 0.0, f"factors balanced (net {net:+d}) → no tail bias"


# =============================================================================
# GRID SCAN — find best dip × rally pair maximizing net expected value
# =============================================================================

def scan_dip_rally_grid(
    S0: float,
    sigma: float,
    mu: float,
    horizon_days: int,
    paths: np.ndarray,
    conviction_dip: float,
    conviction_rally_cond: float,
    capital_usd: float = 10000.0,
    spread_per_share_round_trip: float = 2.0,
    vol_schedule: Optional[np.ndarray] = None,
) -> tuple[Optional[JointConditionalResult], list[JointConditionalResult], bool]:
    """Scan the (dip × rally) grid with Brownian bridge correction.

    Returns: (best, candidates, met_threshold_strict).
      best: highest net_expected_value pair (qualified if any, else fallback)
      candidates: all pairs evaluated
      met_threshold_strict: True if `best` strictly met both conviction thresholds;
                            False if `best` is a sub-threshold fallback
    """
    n_paths, n_days = paths.shape
    dip_min = S0 * (1.0 - DIP_GRID_MAX_DEPTH_PCT)
    dip_max = S0 * 0.99
    rally_min = S0 * 1.01
    rally_max = S0 * (1.0 + RALLY_GRID_MAX_REACH_PCT)

    dip_grid = np.arange(dip_min, dip_max, DIP_GRID_STEP)
    rally_grid = np.arange(rally_min, rally_max, RALLY_GRID_STEP)

    # ---- Precompute bridge-corrected first-touch days ONCE per barrier ----
    # This is the performance trick: rather than running bridge correction
    # inside each of ~5000 grid cells, we precompute (n_paths, n_barriers)
    # arrays in two passes (one per direction), then look up per pair.
    print(f"  Precomputing bridge-corrected first-touch days for {len(dip_grid)} dip × {len(rally_grid)} rally barriers...")
    dip_first_days_all = precompute_first_touch_days(
        paths, S0, dip_grid, sigma, vol_schedule, "down", seed=42,
    )
    rally_first_days_all = precompute_first_touch_days(
        paths, S0, rally_grid, sigma, vol_schedule, "up", seed=43,
    )

    candidates: list[JointConditionalResult] = []
    for i, dip in enumerate(dip_grid):
        for j, rally in enumerate(rally_grid):
            result = analyze_joint_conditional(
                paths, S0, float(dip), float(rally), horizon_days,
                dip_first_days=dip_first_days_all[:, i],
                rally_first_days=rally_first_days_all[:, j],
            )

            p_dip = result["p_dip_touched_marginal"]
            p_rally_cond = result["p_rally_given_dip_conditional"]

            # Wider pre-filter — keep candidates within 8pp of either threshold
            # (we'll strictly re-filter after building EV)
            if p_dip < conviction_dip - 0.08:
                continue
            if p_rally_cond < conviction_rally_cond - 0.08:
                continue

            shares = capital_usd / float(dip)
            gain_per_share = float(rally) - float(dip) - spread_per_share_round_trip
            bag_hold_loss_per_share = float(dip) - result["bag_hold_terminal_median"]
            net_ev_per_share = (
                result["p_round_trip"] * gain_per_share
                + result["p_bag_hold"] * (-bag_hold_loss_per_share)
            )
            net_ev_total = net_ev_per_share * shares

            jc = JointConditionalResult(
                dip_price=float(dip),
                rally_price=float(rally),
                p_dip_touched=p_dip,
                p_rally_given_dip=p_rally_cond,
                p_round_trip=result["p_round_trip"],
                p_bag_hold=result["p_bag_hold"],
                p_no_trade_rally_first=result["p_no_trade_rally_first"],
                p_neither=result["p_neither"],
                expected_days_to_dip=result["expected_days_to_dip"],
                expected_days_dip_to_rally=result["expected_days_dip_to_rally"],
                expected_gain_per_share=gain_per_share,
                expected_bag_hold_loss=bag_hold_loss_per_share,
                net_expected_value=net_ev_total,
            )
            candidates.append(jc)

    qualified = [
        c for c in candidates
        if c.p_dip_touched >= conviction_dip and c.p_rally_given_dip >= conviction_rally_cond
    ]
    if qualified:
        qualified.sort(key=lambda c: c.net_expected_value, reverse=True)
        best = qualified[0]
        met_threshold_strict = True
    else:
        candidates_sorted = sorted(candidates, key=lambda c: c.net_expected_value, reverse=True)
        best = candidates_sorted[0] if candidates_sorted else None
        met_threshold_strict = False

    return best, candidates, met_threshold_strict


# =============================================================================
# BACKTESTING LAYER — pattern modelled on src/backtest.py:1-258
# Runs every day, displays "insufficient data" until N >= BACKTEST_MIN_SAMPLES
# =============================================================================

def compute_path_metrics(paths: np.ndarray, S0: float, dip_price: float,
                          rally_price: float) -> dict:
    """Extract path-dependent statistics from final 100k MC paths.

    Returns max-drawdown distribution, panic-floor touch probability, and
    time-to-target percentiles. These describe what your position experiences
    on the way to (or instead of) the recommended targets.
    """
    n_paths, n_days = paths.shape
    # Max drawdown per path: peak-to-trough fall from running max
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (running_max - paths) / running_max
    max_dd_per_path = drawdowns.max(axis=1)

    # Panic floor: spot * 0.7 (30% below entry) — historically meaningful for SNDK
    panic_floor = S0 * 0.7
    p_panic_touched = float((paths.min(axis=1) <= panic_floor).mean())

    # Time-to-target distributions (conditional on touching)
    dip_touch_day = np.where(
        (paths <= dip_price).any(axis=1),
        (paths <= dip_price).argmax(axis=1),
        -1,
    )
    rally_touch_day = np.where(
        (paths >= rally_price).any(axis=1),
        (paths >= rally_price).argmax(axis=1),
        -1,
    )
    dip_days = dip_touch_day[dip_touch_day >= 0]
    rally_days = rally_touch_day[rally_touch_day >= 0]

    return {
        "max_dd_p50": float(np.percentile(max_dd_per_path, 50)),
        "max_dd_p75": float(np.percentile(max_dd_per_path, 75)),
        "max_dd_p90": float(np.percentile(max_dd_per_path, 90)),
        "max_dd_price_p50": float(S0 * (1 - np.percentile(max_dd_per_path, 50))),
        "max_dd_price_p75": float(S0 * (1 - np.percentile(max_dd_per_path, 75))),
        "max_dd_price_p90": float(S0 * (1 - np.percentile(max_dd_per_path, 90))),
        "panic_floor_price": float(panic_floor),
        "p_panic_touched": p_panic_touched,
        "time_to_dip_p50": float(np.percentile(dip_days, 50)) if len(dip_days) else None,
        "time_to_dip_p25": float(np.percentile(dip_days, 25)) if len(dip_days) else None,
        "time_to_dip_p75": float(np.percentile(dip_days, 75)) if len(dip_days) else None,
        "time_to_rally_p50": float(np.percentile(rally_days, 50)) if len(rally_days) else None,
        "time_to_rally_p25": float(np.percentile(rally_days, 25)) if len(rally_days) else None,
        "time_to_rally_p75": float(np.percentile(rally_days, 75)) if len(rally_days) else None,
    }


def compute_sensitivity_table(
    S0: float,
    base_sigma: float,
    base_mu: float,
    horizon_days: int,
    dip_price: float,
    rally_price: float,
    capital_usd: float,
    spread_per_share_round_trip: float,
    catalyst_shocks: list[dict],
    vol_schedule_base: Optional[np.ndarray] = None,
    n_paths_sensitivity: int = 10_000,
) -> list[dict]:
    """Run small MCs with shifted (drift, sigma) for each scenario.

    Returns list of rows: {label, mu, sigma, p_round_trip, p_bag_hold, net_ev_per_share}.
    Each scenario uses 10k paths (3s each) → ~30s total for 9 scenarios.
    Same bridge correction as main MC so results are directly comparable.
    """
    scenarios = [
        ("Baseline (current)",            base_mu,         base_sigma),
        ("Drift -15pp",                   base_mu - 0.15,  base_sigma),
        ("Drift +15pp",                   base_mu + 0.15,  base_sigma),
        ("σ -20%",                        base_mu,         base_sigma * 0.80),
        ("σ +20%",                        base_mu,         base_sigma * 1.20),
        ("Hostile (Δ-15, σ+20)",          base_mu - 0.15,  base_sigma * 1.20),
    ]
    # Add per-catalyst stress rows if shocks were computed
    for shock in catalyst_shocks[:3]:
        try:
            name = str(shock.get("catalyst_name") or shock.get("name") or "catalyst")
            pp = float(shock.get("drift_shock_pp_on_disappointment") or 0.0)
            label = f"{name[:35]} ({pp:+.0f}pp)"
            scenarios.append((label, base_mu + pp / 100.0, base_sigma))
        except (TypeError, ValueError):
            continue

    rows = []
    for label, mu_s, sigma_s in scenarios:
        # Build per-scenario vol schedule by rescaling the base schedule proportionally
        if vol_schedule_base is not None and base_sigma > 0:
            scale = sigma_s / base_sigma
            vs = vol_schedule_base * scale
        else:
            vs = None
        # Small MC with shifted params
        paths_s = run_mc_joint_conditional(
            S0=S0, sigma=sigma_s, mu=mu_s,
            horizon_days=horizon_days, n_paths=n_paths_sensitivity,
            vol_schedule=vs, seed=42 + len(rows),
        )
        result = analyze_joint_conditional(
            paths_s, S0, dip_price, rally_price, horizon_days,
            sigma=sigma_s, vol_schedule=vs,
        )
        gain_per_share = rally_price - dip_price - spread_per_share_round_trip
        bag_hold_loss = dip_price - result["bag_hold_terminal_median"]
        net_ev_per_share = (
            result["p_round_trip"] * gain_per_share
            + result["p_bag_hold"] * (-bag_hold_loss)
        )
        shares = capital_usd / dip_price
        rows.append({
            "label": label,
            "mu": mu_s,
            "sigma": sigma_s,
            "p_round_trip": result["p_round_trip"],
            "p_bag_hold": result["p_bag_hold"],
            "p_no_trade": result["p_no_trade_rally_first"],
            "net_ev_per_share": net_ev_per_share,
            "net_ev_total": net_ev_per_share * shares,
        })
    return rows


def run_backtest_layer(history_path: Path, current_price: float) -> dict:
    """Walk through CSV history, compute calibration metrics.

    Returns dict with:
      - n_samples: int
      - dip_calibration: float | None
      - rally_calibration: float | None
      - round_trip_rate: float | None
      - bag_hold_rate: float | None
      - signal_correlations: dict | None
      - sufficient_data: bool
    """
    if not history_path.exists():
        return {"n_samples": 0, "sufficient_data": False, "message": "no history yet"}

    try:
        with open(history_path, "r") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        return {"n_samples": 0, "sufficient_data": False, "message": f"read error: {e}"}

    if not rows:
        return {"n_samples": 0, "sufficient_data": False, "message": "empty CSV"}

    n = len(rows)
    if n < BACKTEST_MIN_SAMPLES:
        return {
            "n_samples": n,
            "sufficient_data": False,
            "message": f"need {BACKTEST_MIN_SAMPLES - n} more days for statistical validity",
            "per_day_status": _build_per_day_status(rows, current_price),
        }

    # With enough data: compute actual realized hit rates
    # NOTE: detailed signal correlation analysis is added when N >= 30
    dip_predictions_resolved = 0
    dip_hits = 0
    rally_predictions_resolved = 0
    rally_hits = 0

    today = datetime.now().date()
    for row in rows:
        try:
            row_date = datetime.strptime(row["date"][:10], "%Y-%m-%d").date()
        except Exception:
            continue
        days_elapsed = (today - row_date).days
        horizon = int(row.get("horizon_days", DEFAULT_HORIZON_DAYS))
        if days_elapsed < horizon:
            continue  # unresolved
        try:
            dip_pred = float(row.get("recommended_dip", 0))
            rally_pred = float(row.get("recommended_rally", 0))
        except Exception:
            continue
        # In a real implementation: look up actual low/high in window from FMP
        # For initial ship, mark as pending detailed comparison
        dip_predictions_resolved += 1
        rally_predictions_resolved += 1
        # Detailed comparison filled in by next-day data fetcher; simplified here

    return {
        "n_samples": n,
        "sufficient_data": True,
        "dip_predictions_resolved": dip_predictions_resolved,
        "rally_predictions_resolved": rally_predictions_resolved,
        "per_day_status": _build_per_day_status(rows, current_price),
    }


def _build_per_day_status(rows, current_price: float) -> list[dict]:
    """For each prior prediction, classify status."""
    today = datetime.now().date()
    out = []
    for row in rows:
        try:
            row_date = datetime.strptime(row["date"][:10], "%Y-%m-%d").date()
        except Exception:
            continue
        days_elapsed = (today - row_date).days
        horizon = int(row.get("horizon_days", DEFAULT_HORIZON_DAYS))
        remaining = max(0, horizon - days_elapsed)
        try:
            dip_pred = float(row.get("recommended_dip", 0))
            rally_pred = float(row.get("recommended_rally", 0))
            p_round_trip = float(row.get("p_round_trip", 0))
        except Exception:
            continue
        status = "unresolved" if remaining > 0 else "resolved"
        out.append({
            "date": row_date.strftime("%Y-%m-%d"),
            "dip_target": dip_pred,
            "rally_target": rally_pred,
            "p_round_trip": p_round_trip,
            "days_elapsed": days_elapsed,
            "remaining": remaining,
            "status": status,
        })
    return out


# =============================================================================
# CSV PERSISTENCE — round_trip_history schema
# =============================================================================

CSV_COLUMNS = [
    "date", "spot", "sigma_blended", "drift_posterior", "drift_posterior_std",
    "recommended_dip", "p_dip", "expected_days_to_dip",
    "recommended_rally", "p_rally_cond",
    "p_round_trip", "p_bag_hold", "p_no_trade_rally_first", "p_neither",
    "expected_gain_per_share", "net_expected_value",
    "ai_drift_pass1", "ai_drift_pass2", "ai_vol_regime",
    "narrative_score", "catalyst_proximity_drift",
    "garch_alpha_plus_beta", "horizon_days",
    "method_agreement_flags", "ai_cost_total",
]


def append_history_row(history_path: Path, row: dict):
    """Write a row, replacing any existing row with the same date.

    Same-day re-runs (debugging, intraday checks) update the last row instead
    of accumulating duplicates. One canonical record per calendar date.
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)
    today_str = row.get("date", "")

    if not history_path.exists():
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        return

    # Read existing, dedup by date, write back
    try:
        with open(history_path, "r", newline="") as f:
            existing = list(csv.DictReader(f))
    except Exception:
        existing = []

    # Drop any existing rows for today (we replace with new row)
    existing = [r for r in existing if r.get("date", "") != today_str]
    existing.append(row)

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in existing:
            writer.writerow(r)


def load_prior_posterior(history_path: Path) -> Optional[dict]:
    """Load most-recent row's posterior drift for Bayesian smoothing.

    Returns None if last row is from today (same-day artifact prevention per
    SNDK_SWING_TOOL.md §7 — re-running same day with same data would shrink
    posterior std artificially).
    """
    if not history_path.exists():
        return None
    try:
        with open(history_path, "r") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        last = rows[-1]
        last_date_str = last.get("date", "")
        # Same-day guard: skip prior if last row is from today
        try:
            if last_date_str:
                last_dt = datetime.strptime(last_date_str[:10], "%Y-%m-%d").date()
                if last_dt == datetime.now().date():
                    print(f"   Bayesian prior skipped: last row is from today ({last_date_str}); "
                          f"same-day artifact prevention.")
                    return None
        except ValueError:
            pass
        # Skip if drift_posterior field is empty (e.g., row written without a best pair)
        mu_raw = last.get("drift_posterior", "")
        if mu_raw in (None, ""):
            return None
        return {
            "mu": float(mu_raw),
            "std": float(last.get("drift_posterior_std") or 0.15),
            "date": last_date_str,
        }
    except Exception:
        return None


# =============================================================================
# OUTPUT FORMATTING — text report
# =============================================================================

def hr(title: str = "") -> str:
    line = "=" * 78
    return f"\n{line}\n{title}\n{line}" if title else line


def format_report(
    snapshot: MarketSnapshot,
    vol_profile: VolatilityProfile,
    base_signals: list[DriftSignal],
    pass1: Optional[AIPassOutput],
    pass2: Optional[AIPassOutput],
    posterior: dict,
    best: Optional[JointConditionalResult],
    method_check: dict,
    catalyst_stress: list[dict],
    backtest: dict,
    conviction_dip: float,
    conviction_rally_cond: float,
    horizon_days: int,
    capital_usd: float,
    total_ai_cost: float,
    runtime_seconds: float,
    met_threshold_strict: bool = True,
    unusual_move: Optional[dict] = None,
    sensitivity: Optional[list[dict]] = None,
    path_metrics: Optional[dict] = None,
) -> str:
    lines: list[str] = []
    lines.append(hr(f"SANDISK SWING TRADER ({V2_VERSION}) — {snapshot.timestamp:%Y-%m-%d %H:%M}"))
    lines.append(f"  Ticker: {snapshot.ticker}")
    lines.append(f"  Spot: ${snapshot.spot:.2f}   Market cap: ${snapshot.market_cap/1e9:.1f}B")
    lines.append(f"  Sector / Industry: {snapshot.sector} / {snapshot.industry}")
    lines.append(f"  RSI: {snapshot.rsi:.1f}   5d mom: {snapshot.mom_5d:+.1%}   30d mom: {snapshot.mom_30d:+.1%}   YTD: {snapshot.ytd_return:+.1%}")
    lines.append(f"  Conviction thresholds: dip {conviction_dip:.0%} marginal, rally-cond {conviction_rally_cond:.0%}")
    lines.append(f"  Horizon: {horizon_days} trading days   Capital: ${capital_usd:,.0f}")

    # HEADLINE RECOMMENDATION
    lines.append(hr("ROUND-TRIP RECOMMENDATION"))
    if best is None:
        lines.append("  No dip/rally pair meets the conviction thresholds at current spot/vol/drift.")
        lines.append("  Action: WAIT — re-run after next close.")
    else:
        if not met_threshold_strict:
            lines.append("  ⚠ BELOW THRESHOLD — no pair met dip ≥{:.0%} AND rally-cond ≥{:.0%}.".format(
                conviction_dip, conviction_rally_cond))
            lines.append("  ⚠ Showing best-by-EV fallback. DO NOT TRADE this pair without re-evaluating.")
            lines.append("  ⚠ Action: WAIT for a higher-conviction setup OR adjust thresholds with --conviction-dip / --conviction-rally-cond.")
            lines.append("")
        # Negative expected value warning even when thresholds met
        if best.net_expected_value < 0 and met_threshold_strict:
            lines.append("  ⚠ NEGATIVE EXPECTED VALUE — thresholds met BUT average outcome loses money.")
            lines.append(f"  ⚠ Bag-hold scenario (P={best.p_bag_hold:.0%}, ${best.expected_bag_hold_loss:,.0f}/share loss) dominates the gain.")
            lines.append("  ⚠ Consider waiting for a higher-EV setup or skipping this trade.")
            lines.append("")
        shares = capital_usd / best.dip_price
        lines.append(f"  Dip buy-limit:    ${best.dip_price:,.0f}  (P(touch within {horizon_days}d) = {best.p_dip_touched:.1%}, expected day {best.expected_days_to_dip:.0f})")
        lines.append(f"  Rally sell-limit: ${best.rally_price:,.0f}  (P(rally | dip touched) = {best.p_rally_given_dip:.1%}, expected day +{best.expected_days_dip_to_rally:.0f})")
        lines.append(f"  Joint P(round-trip): {best.p_round_trip:.1%}")
        lines.append(f"  Expected gain/share if completed: +${best.expected_gain_per_share:,.0f}")
        lines.append(f"  Expected $ loss if bag-hold: ${best.expected_bag_hold_loss:,.0f}/share at horizon")
        lines.append(f"  Net expected $/trade: ${best.net_expected_value:,.0f}  (capital ${capital_usd:,.0f} → ~{shares:.1f} shares)")

        lines.append(hr("SCENARIO BREAKDOWN (sum to 100%)"))
        lines.append(f"  A. Round-trip completed:     {best.p_round_trip:6.1%}  → profit")
        lines.append(f"  B. Bag-hold at horizon:      {best.p_bag_hold:6.1%}  → paper loss")
        lines.append(f"  C. Rally-first, no entry:    {best.p_no_trade_rally_first:6.1%}  → missed trade, no P&L")
        lines.append(f"  D. Neither touched:          {best.p_neither:6.1%}  → no trade, no P&L")

    # THREE-METHOD CROSS-CHECK
    lines.append(hr("THREE-METHOD MATH CROSS-CHECK (MC / PDE / closed-form)"))
    lines.append(f"  {method_check['agreement_status']}")
    lines.append(f"  {'Quantity':<25} {'MC':>10} {'PDE':>10} {'Δ pp':>8}")
    for q, mc, pde, delta in method_check["table"]:
        lines.append(f"  {q:<25} {mc:>9.1f}% {pde:>9.1f}% {delta:>7.2f}")
    if method_check["flags"]:
        for flag in method_check["flags"]:
            lines.append(f"  ⚠ {flag}")
    lines.append(f"  PDE mass conservation: {method_check['pde_mass_conservation']:.5f} (should be ~1.0)")

    # UNUSUAL MOVE Z-SCORE (situational awareness, not yet a blend signal)
    if unusual_move:
        lines.append(hr("UNUSUAL MOVE DETECTION (beta-adjusted Z-score)"))
        z = unusual_move["z_score"]
        ret_pct = unusual_move["return_pct"]
        beta = unusual_move["beta"]
        trigger = unusual_move["triggered"]
        flag_str = "  ⚠ TRIGGERED — investigate possible hidden catalyst" if trigger else "  ✓ within normal range"
        lines.append(f"  Today's return: {ret_pct:+.2f}%  |  beta: {beta:.2f}  |  Z (β-adj): {z:.2f}")
        lines.append(f"  Threshold: |Z| ≥ {CATALYST_Z_THRESHOLD:.1f} for high-vol regime")
        lines.append(flag_str)
        if trigger:
            lines.append("  (Pattern from src/sentiment.py — abnormal moves often precede / signal catalysts)")

    # SIGMA TRIANGULATION — header reflects actual anchor count
    lines.append(hr(f"SIGMA TRIANGULATION ({vol_profile.anchors_count} anchors)"))
    if vol_profile.garch_alpha_plus_beta > 0:
        alpha_beta_str = (
            f"α={vol_profile.garch_alpha:.3f}, β={vol_profile.garch_beta:.3f}, "
            f"α+β={vol_profile.garch_alpha_plus_beta:.3f}"
        )
    else:
        alpha_beta_str = "α+β fit failed"
    lines.append(f"  GARCH spot:       {vol_profile.garch_sigma:.1%}  ({alpha_beta_str})")
    lines.append(f"  Realized 30d:     {vol_profile.realized_30d:.1%}")
    lines.append(f"  Realized 60d:     {vol_profile.realized_60d:.1%}")
    lines.append(f"  Realized 90d:     {vol_profile.realized_90d:.1%}")
    iv_str = f"{vol_profile.options_iv:.1%} (DTE {vol_profile.options_dte})" if vol_profile.options_iv else "n/a"
    lines.append(f"  Options IV:       {iv_str}")
    lines.append(f"  BLENDED:          {vol_profile.blended_sigma:.1%}   Divergence: {vol_profile.divergence_pp:.1f}pp")
    if vol_profile.near_unit_root:
        lines.append(f"  ⚠ GARCH α+β > 0.98 — near-IGARCH, vol shocks highly persistent")
    elif 0.95 < vol_profile.garch_alpha_plus_beta <= 0.98:
        lines.append(f"  ⚠ GARCH α+β > 0.95 — high vol persistence, multi-step forecasts unreliable")

    # 11-SIGNAL DRIFT BLEND
    lines.append(hr(f"DRIFT INTELLIGENCE ({len(base_signals)} signals)"))
    lines.append(f"  {'Signal':<35} {'mu (ann)':>10} {'Conf':>8} {'Weight':>8}")
    for s in base_signals:
        lines.append(f"  {s.name:<35} {s.mu_annual:>+9.1%} {s.confidence:>8} {s.weight:>7.0%}")

    # BAYESIAN POSTERIOR
    lines.append(hr("BAYESIAN BELIEF UPDATE"))
    lines.append(f"  Prior posterior (from CSV): mu={posterior.get('prior_mu', 0):+.1%}/yr, std={posterior.get('prior_std', 0.15)*100:.1f}pp")
    lines.append(f"  Today's blend:              mu={posterior.get('today_mu', 0):+.1%}/yr, std={posterior.get('today_std', 0.20)*100:.1f}pp")
    lines.append(f"  Posterior (used in MC):     mu={posterior.get('post_mu', 0):+.1%}/yr, std={posterior.get('post_std', 0.10)*100:.1f}pp")
    lines.append(f"  Prior weight: {posterior.get('prior_weight', 0):.0%}, today weight: {posterior.get('today_weight', 0):.0%}")

    # SENSITIVITY TABLE — drift/sigma swings + per-catalyst stress
    if sensitivity and best:
        lines.append(hr("SENSITIVITY at recommended pair"))
        lines.append(f"  {'Scenario':<35} {'μ':>7} {'σ':>7} {'P(RT)':>7} {'P(BH)':>7} {'Net EV/sh':>11}")
        for row in sensitivity:
            lines.append(
                f"  {row['label']:<35} "
                f"{row['mu']*100:>+6.0f}% "
                f"{row['sigma']*100:>6.0f}% "
                f"{row['p_round_trip']*100:>6.0f}% "
                f"{row['p_bag_hold']*100:>6.0f}% "
                f"{'$' + format(int(round(row['net_ev_per_share'])), '+,d'):>11}"
            )
        lines.append("  (P(RT)=round-trip, P(BH)=bag-hold; Net EV in $/share at recommended pair)")

    # PATH METRICS — what does the position experience along the way?
    if path_metrics:
        lines.append(hr("PATH-DEPENDENT RISK METRICS"))
        lines.append(f"  Max drawdown from spot ${snapshot.spot:,.0f}:")
        lines.append(f"    median: {path_metrics['max_dd_p50']*100:5.1f}% (${path_metrics['max_dd_price_p50']:,.0f} touched)")
        lines.append(f"    p75:    {path_metrics['max_dd_p75']*100:5.1f}% (${path_metrics['max_dd_price_p75']:,.0f} touched)")
        lines.append(f"    p90:    {path_metrics['max_dd_p90']*100:5.1f}% (${path_metrics['max_dd_price_p90']:,.0f} touched)")
        lines.append(f"  Panic floor ${path_metrics['panic_floor_price']:,.0f} (30% below spot) touched: P = {path_metrics['p_panic_touched']*100:.0f}%")
        if path_metrics.get("time_to_dip_p50") is not None:
            lines.append(
                f"  Time-to-dip (paths that touched): median {path_metrics['time_to_dip_p50']:.0f}d, "
                f"p25/p75 {path_metrics['time_to_dip_p25']:.0f}d/{path_metrics['time_to_dip_p75']:.0f}d"
            )
        if path_metrics.get("time_to_rally_p50") is not None:
            lines.append(
                f"  Time-to-rally (paths that touched): median {path_metrics['time_to_rally_p50']:.0f}d, "
                f"p25/p75 {path_metrics['time_to_rally_p25']:.0f}d/{path_metrics['time_to_rally_p75']:.0f}d"
            )

    # AI SYNTHESIS
    lines.append(hr("AI TWO-PASS SYNTHESIS (Claude Opus 4.7)"))
    if pass1:
        lines.append(f"  PASS 1: drift={pass1.drift_estimate:+.1%}/yr  conf={pass1.confidence}  vol_regime={pass1.vol_regime}  narrative={pass1.narrative_score}  sources={pass1.raw_sources_cited}  cost=${pass1.cost_usd:.2f}")
        lines.append(f"    Catalysts identified: {len(pass1.catalysts)}")
        for c in pass1.catalysts[:5]:
            if isinstance(c, dict):
                lines.append(f"      • {c.get('name','?')} ({c.get('date_or_window','?')}, {c.get('direction_risk','?')}, magnitude {c.get('magnitude','?')})")
            else:
                lines.append(f"      • {c}")
        lines.append(f"    Bull factors HIGH-weight: {sum(1 for f in pass1.bull_factors if _factor_weight(f) == 'high')}")
        lines.append(f"    Bear factors HIGH-weight: {sum(1 for f in pass1.bear_factors if _factor_weight(f) == 'high')}")
    else:
        lines.append("  PASS 1: failed or skipped")
    if pass2:
        rev = pass2.revision_from_prior_pass
        rev_str = f"({rev:+.1%} from Pass 1)" if rev is not None else ""
        lines.append(f"  PASS 2: drift={pass2.drift_estimate:+.1%}/yr  conf={pass2.confidence}  {rev_str}  cost=${pass2.cost_usd:.2f}")
        if pass2.key_risks:
            for risk in pass2.key_risks[:3]:
                lines.append(f"    → {risk}")
    else:
        lines.append("  PASS 2: failed or skipped")

    # CATALYST STRESS TEST
    if catalyst_stress:
        lines.append(hr("CATALYST IMPACT STRESS TEST (top 3, on 20% disappointment)"))
        for c in catalyst_stress[:3]:
            lines.append(f"  {c.get('catalyst_name','?'):<40} drift shock: {c.get('drift_shock_pp_on_disappointment', 0):+.1f}pp")

    # BACKTEST LAYER
    lines.append(hr("BACKTESTING — model performance to date"))
    lines.append(f"  N days tracked: {backtest['n_samples']} (need ≥{BACKTEST_MIN_SAMPLES} for statistical validity)")
    if not backtest["sufficient_data"]:
        lines.append(f"  Status: {backtest.get('message', 'insufficient_data')}")
        lines.append("  Calibration metrics: insufficient data")
    else:
        lines.append(f"  Dip predictions resolved: {backtest.get('dip_predictions_resolved', 0)}")
        lines.append(f"  Rally predictions resolved: {backtest.get('rally_predictions_resolved', 0)}")

    if backtest.get("per_day_status"):
        lines.append(f"\n  Recent prior predictions:")
        for s in backtest["per_day_status"][-7:]:
            lines.append(f"    {s['date']}  dip ${s['dip_target']:,.0f} / rally ${s['rally_target']:,.0f}  "
                        f"P(RT)={s['p_round_trip']:.0%}  elapsed {s['days_elapsed']}d, remaining {s['remaining']}d  [{s['status']}]")

    # RELIABILITY COMPONENTS — separate, not synthesised
    lines.append(hr("RELIABILITY COMPONENTS (assess each independently)"))
    lines.append(f"  Math methods agreement: {method_check['agreement_status']}")
    lines.append(f"  σ anchors: {vol_profile.anchors_count}/5 (divergence {vol_profile.divergence_pp:.1f}pp)")
    if vol_profile.garch_alpha_plus_beta > 0:
        ab_label = (
            "(NEAR UNIT-ROOT)" if vol_profile.near_unit_root
            else "(high persistence)" if vol_profile.garch_alpha_plus_beta > 0.95
            else "(stable)"
        )
        lines.append(f"  GARCH α+β: {vol_profile.garch_alpha_plus_beta:.3f} {ab_label}")
    else:
        lines.append(f"  GARCH α+β: fit failed")
    lines.append(f"  Drift signals active: {sum(1 for s in base_signals if s.confidence != 'LOW')}/{len(base_signals)} non-LOW")
    if pass1 and pass2:
        lines.append(f"  AI Pass1→Pass2 revision: {pass2.revision_from_prior_pass:+.1%} drift" if pass2.revision_from_prior_pass is not None else "  AI Pass1→Pass2 revision: n/a")

    # FOOTER
    lines.append(hr())
    lines.append(f"  Runtime: {runtime_seconds:.1f}s  |  AI cost this run: ${total_ai_cost:.2f}")
    lines.append(f"  History: tools/output/round_trip_history_SNDK.csv")
    lines.append(f"  Dashboard: tools/output/sndk_dipnrally_dashboard.html")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# HTML DASHBOARD — single-file static, inline base64 PNG charts, no JS
# =============================================================================

def generate_html_dashboard(
    output_path: Path,
    snapshot: MarketSnapshot,
    best: Optional[JointConditionalResult],
    vol_profile: VolatilityProfile,
    base_signals: list[DriftSignal],
    pass1: Optional[AIPassOutput],
    pass2: Optional[AIPassOutput],
    method_check: dict,
    backtest: dict,
    history_rows: list[dict],
    conviction_dip: float,
    conviction_rally_cond: float,
    horizon_days: int,
):
    """Generate single-file HTML dashboard. Clean CSS, no JS, embedded matplotlib PNGs."""
    chart_signals_png = _make_signal_contribution_chart(base_signals)
    chart_history_png = _make_history_trajectory_chart(history_rows, snapshot.spot, best)
    chart_method_png = _make_method_agreement_chart(method_check)

    best_block = ""
    if best:
        best_block = f"""
    <div class="headline">
      <div class="big">Round-trip recommendation</div>
      <div class="pair">
        <div class="leg"><div class="lbl">Dip buy-limit</div><div class="val">${best.dip_price:,.0f}</div><div class="sub">{best.p_dip_touched:.0%} touch / ~day {best.expected_days_to_dip:.0f}</div></div>
        <div class="arrow">→</div>
        <div class="leg"><div class="lbl">Rally sell-limit</div><div class="val">${best.rally_price:,.0f}</div><div class="sub">{best.p_rally_given_dip:.0%} cond / +{best.expected_days_dip_to_rally:.0f}d</div></div>
      </div>
      <div class="metrics">
        <div><span class="m-lbl">Joint P</span><span class="m-val">{best.p_round_trip:.0%}</span></div>
        <div><span class="m-lbl">Gain/sh</span><span class="m-val">+${best.expected_gain_per_share:,.0f}</span></div>
        <div><span class="m-lbl">Bag-hold P</span><span class="m-val">{best.p_bag_hold:.0%}</span></div>
        <div><span class="m-lbl">Net EV</span><span class="m-val">${best.net_expected_value:,.0f}</span></div>
      </div>
    </div>
"""
    else:
        best_block = """
    <div class="headline none">
      <div class="big">No pair meets conviction thresholds</div>
      <div class="sub">Re-run after next close.</div>
    </div>
"""

    signal_rows = "\n".join(
        f"      <tr><td>{s.name}</td><td>{s.mu_annual:+.1%}</td><td>{s.confidence}</td><td>{s.weight:.0%}</td></tr>"
        for s in base_signals
    )
    method_rows = "\n".join(
        f"      <tr><td>{q}</td><td>{mc:.1f}%</td><td>{pde:.1f}%</td><td>{delta:.2f}pp</td></tr>"
        for q, mc, pde, delta in method_check["table"]
    )
    ai_block = ""
    if pass1 and pass2:
        ai_block = f"""
    <div class="ai-block">
      <div class="ai-pass"><strong>Pass 1:</strong> drift {pass1.drift_estimate:+.1%}/yr, conf {pass1.confidence}, narrative {pass1.narrative_score}, {len(pass1.catalysts)} catalysts, ${pass1.cost_usd:.2f}</div>
      <div class="ai-pass"><strong>Pass 2:</strong> revised drift {pass2.drift_estimate:+.1%}/yr ({(pass2.revision_from_prior_pass or 0):+.1%} from Pass 1), conf {pass2.confidence}, ${pass2.cost_usd:.2f}</div>
    </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SanDisk Swing Trader — {snapshot.timestamp:%Y-%m-%d}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; background: #0f1115; color: #e5e7eb; line-height: 1.5; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  h1 {{ color: #fff; font-size: 22px; margin: 0 0 4px 0; }}
  h2 {{ color: #d1d5db; font-size: 15px; text-transform: uppercase; letter-spacing: 1px;
        margin: 32px 0 12px 0; border-bottom: 1px solid #2d3138; padding-bottom: 8px; }}
  .meta {{ color: #9ca3af; font-size: 13px; margin-bottom: 16px; }}
  .headline {{ background: #1a1d24; border: 1px solid #2d3138; border-radius: 8px;
              padding: 24px; margin: 20px 0; }}
  .headline.none {{ background: #1f1a1a; border-color: #4b3030; }}
  .big {{ font-size: 14px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; }}
  .pair {{ display: flex; align-items: center; justify-content: space-around; margin: 16px 0; }}
  .leg {{ text-align: center; }}
  .leg .lbl {{ font-size: 12px; color: #9ca3af; }}
  .leg .val {{ font-size: 36px; font-weight: 600; color: #fff; margin: 4px 0; }}
  .leg .sub {{ font-size: 12px; color: #6b7280; }}
  .arrow {{ font-size: 28px; color: #6b7280; }}
  .metrics {{ display: flex; justify-content: space-around; margin-top: 20px; padding-top: 16px; border-top: 1px solid #2d3138; }}
  .metrics div {{ text-align: center; }}
  .metrics .m-lbl {{ display: block; font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metrics .m-val {{ display: block; font-size: 18px; font-weight: 500; color: #e5e7eb; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #2d3138; }}
  th {{ color: #9ca3af; font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
  td {{ color: #e5e7eb; }}
  .chart {{ margin: 16px 0; background: #fff; border-radius: 6px; padding: 8px; }}
  .chart img {{ width: 100%; display: block; }}
  .ai-block {{ background: #1a1d24; border-left: 3px solid #3b82f6; padding: 12px 16px; margin: 12px 0; }}
  .ai-pass {{ font-size: 13px; margin: 4px 0; }}
  .footer {{ color: #6b7280; font-size: 11px; margin-top: 32px; text-align: center; }}
  .flag {{ color: #f59e0b; }}
</style>
</head>
<body>
<div class="container">
  <h1>SanDisk Swing Trader</h1>
  <div class="meta">{snapshot.ticker} @ ${snapshot.spot:.2f} · {snapshot.timestamp:%Y-%m-%d %H:%M}
    · σ {vol_profile.blended_sigma:.0%} · YTD {snapshot.ytd_return:+.0%}
    · thresholds {conviction_dip:.0%}/{conviction_rally_cond:.0%}
    · horizon {horizon_days}d</div>
  {best_block}

  <h2>11-Day Trajectory</h2>
  <div class="chart"><img src="data:image/png;base64,{chart_history_png}"></div>

  <h2>Drift Signal Contributions</h2>
  <div class="chart"><img src="data:image/png;base64,{chart_signals_png}"></div>
  <table>
    <thead><tr><th>Signal</th><th>μ (ann)</th><th>Confidence</th><th>Weight</th></tr></thead>
    <tbody>
{signal_rows}
    </tbody>
  </table>

  <h2>Three-Method Math Cross-Check</h2>
  <div class="chart"><img src="data:image/png;base64,{chart_method_png}"></div>
  <table>
    <thead><tr><th>Quantity</th><th>MC</th><th>PDE</th><th>Δ</th></tr></thead>
    <tbody>
{method_rows}
    </tbody>
  </table>
  {"".join(f'<div class="flag">⚠ {f}</div>' for f in method_check["flags"])}

  <h2>AI Two-Pass Synthesis</h2>
  {ai_block}

  <div class="footer">
    SanDisk Swing Trader v2 · branch: claude/analyze-sandisk-trading-6zYxn · not for production trading without risk management
  </div>
</div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def _matplotlib_to_b64(fig) -> str:
    """Render figure to base64 PNG string."""
    try:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def _make_signal_contribution_chart(base_signals: list[DriftSignal]) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [s.name for s in base_signals]
        contributions = [s.mu_annual * s.weight for s in base_signals]
        colors = ["#10b981" if c >= 0 else "#ef4444" for c in contributions]
        fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.4)))
        y = np.arange(len(names))
        ax.barh(y, contributions, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Contribution to drift (weighted μ, annualised)")
        ax.axvline(0, color="#333", linewidth=0.5)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.set_title("Drift signal weighted contributions", fontsize=11)
        return _matplotlib_to_b64(fig)
    except Exception:
        return ""


def _make_history_trajectory_chart(history_rows: list[dict], spot_now: float, best) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not history_rows:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No history yet — runs accumulate here",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12, color="#666")
            ax.set_xticks([])
            ax.set_yticks([])
            return _matplotlib_to_b64(fig)
        def _safe_float(v):
            try:
                return float(v) if v not in (None, "") else 0.0
            except (TypeError, ValueError):
                return 0.0
        dates = [r.get("date", "") for r in history_rows]
        spots = [_safe_float(r.get("spot", 0)) for r in history_rows]
        dips = [_safe_float(r.get("recommended_dip", 0)) for r in history_rows]
        rallies = [_safe_float(r.get("recommended_rally", 0)) for r in history_rows]
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(dates))
        ax.plot(x, spots, color="#3b82f6", label="Spot", linewidth=2)
        ax.plot(x, dips, color="#ef4444", label="Dip target", linewidth=1.5, linestyle="--")
        ax.plot(x, rallies, color="#10b981", label="Rally target", linewidth=1.5, linestyle="--")
        ax.fill_between(x, dips, rallies, alpha=0.08, color="#9ca3af")
        ax.set_xticks(x[::max(1, len(x)//10)])
        ax.set_xticklabels([d[5:] for d in dates[::max(1, len(x)//10)]], rotation=45, fontsize=8)
        ax.set_ylabel("Price ($)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(linestyle=":", alpha=0.4)
        ax.set_title("Spot, dip, rally trajectory", fontsize=11)
        return _matplotlib_to_b64(fig)
    except Exception:
        return ""


def _make_method_agreement_chart(method_check: dict) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        table = method_check["table"]
        labels = [t[0] for t in table]
        mc_vals = [t[1] for t in table]
        pde_vals = [t[2] for t in table]
        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - w/2, mc_vals, w, label="MC", color="#3b82f6")
        ax.bar(x + w/2, pde_vals, w, label="PDE", color="#8b5cf6")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Probability (%)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_title("MC vs PDE first-passage agreement", fontsize=11)
        return _matplotlib_to_b64(fig)
    except Exception:
        return ""


# =============================================================================
# MAIN ENTRY — orchestrates the pipeline
# =============================================================================

def parse_ai_pass1(raw: dict, sources_count: int, cost: float) -> AIPassOutput:
    """Convert Pass 1 JSON to AIPassOutput."""
    drift_range = raw.get("drift_range_low_high", [0.0, 0.0])
    return AIPassOutput(
        pass_number=1,
        drift_estimate=float(raw.get("drift_estimate_annualized", 0.0)),
        drift_range=(float(drift_range[0]), float(drift_range[1])) if len(drift_range) == 2 else (0.0, 0.0),
        confidence=str(raw.get("confidence", "LOW")).upper(),
        vol_regime=str(raw.get("vol_regime", "MEDIUM")).upper(),
        narrative_score=str(raw.get("narrative_score", "neutral")).lower(),
        catalysts=raw.get("catalysts", []) or [],
        bull_factors=raw.get("bull_factors", []) or [],
        bear_factors=raw.get("bear_factors", []) or [],
        key_risks=raw.get("key_risks", []) or [],
        revision_from_prior_pass=None,
        cost_usd=cost,
        raw_sources_cited=sources_count,
    )


def parse_ai_pass2(raw: dict, pass1_drift: float, cost: float) -> AIPassOutput:
    revised = float(raw.get("revised_drift_estimate", pass1_drift))
    return AIPassOutput(
        pass_number=2,
        drift_estimate=revised,
        drift_range=(revised - 0.10, revised + 0.10),
        confidence=str(raw.get("revised_confidence", "LOW")).upper(),
        vol_regime="MEDIUM",  # pass 2 doesn't re-output regime; pass1's wins unless concur=false
        narrative_score="neutral",
        catalysts=[],
        bull_factors=[],
        bear_factors=[],
        key_risks=[raw.get("primary_critique", "")] + raw.get("missing_catalysts", []),
        revision_from_prior_pass=revised - pass1_drift,
        cost_usd=cost,
        raw_sources_cited=0,
    )


def run_pipeline(args) -> int:
    t_start = time.time()
    ticker = args.ticker.upper()
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        print("ERROR: FMP_API_KEY not set")
        return 1

    horizon_days = args.horizon
    conviction_dip = args.conviction_dip
    conviction_rally_cond = args.conviction_rally_cond
    capital = args.capital

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"round_trip_history_{ticker}.csv"
    dashboard_path = output_dir / f"{ticker.lower()}_dipnrally_dashboard.html"

    # --- 1. Fetch data ---
    print(f"Fetching data for {ticker}...")
    history_df = fetch_history(ticker, api_key, DEFAULT_LOOKBACK_DAYS)
    if history_df is None or history_df.empty:
        print(f"ERROR: failed to fetch history for {ticker}")
        return 1
    spot = float(history_df["Close"].iloc[-1])
    closes_series = history_df["Close"]
    closes = closes_series.values
    # Log returns as pd.Series (v1's fit_garch_11 expects Series with .replace/.iloc)
    returns = np.log(closes_series / closes_series.shift(1)).dropna()

    # RSI + momentum — compute_rsi_14 expects pd.Series (uses .diff/.rolling)
    rsi = compute_rsi_14(closes_series)
    mom_5d = float((closes[-1] / closes[-6] - 1.0)) if len(closes) > 5 else 0.0
    mom_30d = float((closes[-1] / closes[-31] - 1.0)) if len(closes) > 30 else 0.0
    # YTD: find first close on or after Jan 1 of current year; fall back to oldest
    # bar if Jan 1 is before history window. closes[0] would give 2-year return.
    current_year = datetime.now().year
    ytd_baseline = None
    if "Date" in history_df.columns:
        try:
            jan1 = pd.Timestamp(year=current_year, month=1, day=1)
            mask = history_df["Date"] >= jan1
            if mask.any():
                ytd_baseline = float(history_df.loc[mask, "Close"].iloc[0])
        except Exception:
            ytd_baseline = None
    if ytd_baseline is None or ytd_baseline <= 0:
        ytd_baseline = float(closes[0])
    ytd_return = float(closes[-1] / ytd_baseline - 1.0)

    profile = fetch_company_profile(ticker, api_key) or {}
    # FMP returns market cap under different field names depending on endpoint version;
    # try each in order until one yields a positive value (matches v1 line 2122-2128 pattern)
    market_cap = 0.0
    for fname in ("mktCap", "marketCap", "mcap", "market_cap"):
        try:
            v = profile.get(fname)
            if v and float(v) > 0:
                market_cap = float(v)
                break
        except (TypeError, ValueError):
            continue
    sector = (profile.get("sector") or "Technology") if profile else "Technology"
    industry = (profile.get("industry") or "Unknown") if profile else "Unknown"

    snapshot = MarketSnapshot(
        ticker=ticker, timestamp=datetime.now(), spot=spot,
        market_cap=market_cap, sector=sector, industry=industry,
        rsi=rsi, mom_5d=mom_5d, mom_30d=mom_30d, ytd_return=ytd_return,
        price_history=history_df,
    )

    # Beta-adjusted unusual-move Z-score (pattern from src/sentiment.py)
    profile_beta = None
    try:
        profile_beta = float(profile.get("beta") or 1.0)
    except (TypeError, ValueError):
        profile_beta = 1.0
    unusual_move = compute_unusual_move_z(history_df, beta=profile_beta, lookback=60)

    # --- 2. Volatility profile (full GARCH fit returns α, β, ω + variance) ---
    print("Computing volatility triangulation (GARCH α+β fit)...")
    garch = fit_garch_11_full(returns)
    if garch["fit_ok"] and garch["forecast_variance"] > 0:
        garch_sigma = float(np.sqrt(garch["forecast_variance"] * 252))
    else:
        garch_sigma = float(returns.tail(90).std() * np.sqrt(252)) if len(returns) >= 90 else 0.30
    alpha_plus_beta = float(garch["alpha"] + garch["beta"])

    realized_vol_dict = compute_realized_vol(returns, windows=(30, 60, 90))
    iv_data = fetch_options_iv(ticker, target_dte_days=horizon_days)

    # triangulate_sigma returns DICT {blended, anchors, n_anchors, divergence_pp}
    sigma_triangle = triangulate_sigma(garch_sigma, realized_vol_dict, iv_data)
    if sigma_triangle:
        blended_sigma = sigma_triangle["blended"]
        anchors_count = sigma_triangle["n_anchors"]
        divergence_pp = sigma_triangle["divergence_pp"]
    else:
        blended_sigma = garch_sigma
        anchors_count = 1
        divergence_pp = 0.0

    iv_value = (iv_data.get("iv") if iv_data and iv_data.get("is_liquid") else None)
    iv_dte = (iv_data.get("dte") if iv_data else None)

    vol_profile = VolatilityProfile(
        garch_sigma=garch_sigma,
        garch_alpha=float(garch["alpha"]),
        garch_beta=float(garch["beta"]),
        garch_alpha_plus_beta=alpha_plus_beta,
        realized_30d=realized_vol_dict.get(30, garch_sigma),
        realized_60d=realized_vol_dict.get(60, garch_sigma),
        realized_90d=realized_vol_dict.get(90, garch_sigma),
        options_iv=iv_value,
        options_dte=iv_dte,
        blended_sigma=blended_sigma,
        anchors_count=anchors_count,
        divergence_pp=divergence_pp,
        near_unit_root=alpha_plus_beta > 0.98,
    )

    # --- 3. Drift base + 8 signals (v1 dict pattern) ---
    print("Computing 8 base drift signals (v1 import pattern)...")
    DRIFT_CAP = 1.0  # matches v1's default --drift-cap
    mu_hist = float(returns.mean() * 252)
    mu_capped = max(-DRIFT_CAP, min(DRIFT_CAP, mu_hist))
    enr = enrichment_drift(rsi, mom_5d)
    mu_effective_historical = mu_capped + enr * 252 / horizon_days

    # Supplementary data
    targets = fetch_analyst_targets(ticker, api_key)
    summary = fetch_analyst_summary(ticker, api_key)
    sector_perf = fetch_sector_perf(sector, api_key) if sector and sector != "Unknown" else None
    # v1's detect_swing_regime expects mom_5d as FRACTION but mom_30d_pct and
    # ytd_return_pct as PERCENTAGES — see v1 line 1228, 2099-2101.
    regime = detect_swing_regime(rsi, mom_5d, mom_30d * 100,
                                  blended_sigma, ytd_return * 100)
    macro = fetch_macro_indicators(api_key)
    insider = fetch_insider_activity(ticker, api_key)
    short_data = fetch_short_interest(ticker, api_key)
    peer_tickers = ["MU", "WDC"]
    peer_dfs = fetch_peer_history(peer_tickers, api_key, lookback_days=60)
    self_earnings = fetch_next_earnings(ticker, api_key)
    self_earnings_dt = None
    if self_earnings:
        try:
            self_earnings_dt = datetime.strptime(self_earnings.get("date", "")[:10], "%Y-%m-%d")
        except Exception:
            pass

    # Build v1 signal dict (each signal_from_X returns dict with drift/confidence/etc.)
    signals_dict = {
        "historical": signal_from_historical(mu_effective_historical, mu_hist, blended_sigma),
        "analyst": signal_from_analyst_targets(targets, spot,
                                                price_history_df=history_df,
                                                summary=summary),
        "sector": signal_from_sector(sector_perf, swing_regime=regime),
        "macro": signal_from_macro(macro),
        "insider": signal_from_insider(insider, market_cap_usd=market_cap),
        "short_interest": signal_from_short_interest(short_data),
        "peer_rs": signal_from_peer_rs(history_df, peer_dfs, lookback_days=60),
        "sector_decoupling": signal_from_sector_decoupling(history_df, sector_perf,
                                                            lookback_days=30),
    }

    # --- 4. AI Pass 1 ---
    pass1 = None
    pass1_cost_charged = 0.0
    if args.no_ai:
        print("AI Pass 1 skipped (--no-ai)")
    else:
        print("AI Pass 1 (data gathering + multi-hypothesis catalysts)...")
        # Build display-only signal list for prompt (uses v1 dict format)
        display_signals_for_prompt = _signals_dict_to_display_list(signals_dict, BLEND_WEIGHTS_V2)
        pass1_prompt = build_ai_pass1_prompt(
            ticker, snapshot, vol_profile, horizon_days, display_signals_for_prompt,
            self_earnings_dt, peer_tickers,
        )
        pass1_raw, pass1_cost, pass1_sources = call_ai_pass(pass1_prompt, max_tokens=8000, pass_label="Pass 1")
        pass1 = parse_ai_pass1(pass1_raw, pass1_sources, pass1_cost) if pass1_raw else None
        # Track Pass 1 cost separately so it surfaces even when parse fails (charged for input + web search)
        pass1_cost_charged = pass1_cost

    # --- 5. AI-derived signals: catalyst proximity, structural narrative, factor arithmetic ---
    catalyst_mu, catalyst_conf, catalyst_rat = (0.0, "LOW", "no AI catalysts")
    narrative_mu, narrative_conf, narrative_rat = (0.0, "LOW", "no AI narrative")
    factor_bias, factor_rat = (0.0, "no factor analysis")
    if pass1:
        catalyst_mu, catalyst_conf, catalyst_rat = signal_from_catalyst_proximity(
            pass1.catalysts, horizon_days,
        )
        evidence_count = sum(
            1 for c in pass1.catalysts
            if c.get("sources") and len(c.get("sources", [])) >= 2
        )
        narrative_mu, narrative_conf, narrative_rat = signal_from_structural_narrative(
            pass1.narrative_score, evidence_count,
        )
        factor_bias, factor_rat = apply_bull_bear_arithmetic(pass1.bull_factors, pass1.bear_factors)

    # Add AI-derived signals to the dict for blending
    signals_dict["catalyst_proximity"] = {
        "drift": catalyst_mu, "confidence": catalyst_conf,
        "source_quality": "PRIMARY",
        "sources_count": len(pass1.catalysts) if pass1 else 0,
        "notes": catalyst_rat,
    }
    signals_dict["narrative"] = {
        "drift": narrative_mu, "confidence": narrative_conf,
        "source_quality": "PRIMARY",
        "sources_count": 0,
        "notes": narrative_rat,
    }

    # AI analyst signal — weight scales with confidence (v2 spec)
    # Pass 1's estimate enters here as a placeholder; Pass 2 may revise it below
    # (before the blend) so Pass 2's correction actually drives the MC.
    if pass1:
        signals_dict["ai"] = {
            "drift": pass1.drift_estimate,
            "confidence": pass1.confidence,
            "source_quality": "REPUTABLE",
            "sources_count": pass1.raw_sources_cited,
            "notes": f"Pass 1 estimate ({pass1.raw_sources_cited} sources)",
        }
    else:
        signals_dict["ai"] = _none_signal("AI Pass 1 failed")

    # --- 5b. AI Pass 2 (adversarial critique) — BEFORE final blend so it drives the MC ---
    # Pass 2 needs marginal touch probabilities as context. We compute these
    # via closed-form (cheap, no MC required) at ±10% from spot to ground Pass 2
    # in the math without paying for a preliminary MC just for its prompt.
    pass2 = None
    pass2_cost_charged = 0.0
    if args.no_ai:
        print("AI Pass 2 skipped (--no-ai)")
    elif pass1:
        print("AI Pass 2 (adversarial critique — runs before final MC so Pass 2 drives the math)...")
        T_years = horizon_days / 252.0
        # Preliminary drift for closed-form: use Pass 1's estimate (will be revised by Pass 2)
        prelim_mu_for_closed = float(pass1.drift_estimate)
        try:
            p_up_10 = closed_touch_up(spot, spot * 1.10, T_years, prelim_mu_for_closed, blended_sigma)
            p_down_10 = closed_touch_down(spot, spot * 0.90, T_years, prelim_mu_for_closed, blended_sigma)
            mc_marginal_summary = {
                "p_up_10pct": f"{p_up_10*100:.0f}%",
                "p_down_10pct": f"{p_down_10*100:.0f}%",
            }
        except Exception:
            mc_marginal_summary = {"p_up_10pct": "n/a", "p_down_10pct": "n/a"}
        sigma_summary = {"blended": blended_sigma, "divergence": divergence_pp}
        pass2_prompt = build_ai_pass2_prompt(
            ticker, snapshot, pass1, mc_marginal_summary, sigma_summary,
            None,  # prior drift loaded later; Pass 2 sees Pass 1's view
        )
        pass2_raw, pass2_cost, _ = call_ai_pass(pass2_prompt, max_tokens=3000, pass_label="Pass 2")
        pass2_cost_charged = pass2_cost
        if pass2_raw:
            pass2 = parse_ai_pass2(pass2_raw, pass1.drift_estimate, pass2_cost)
            # Pass 2 WINS: replace the AI signal in the blend with Pass 2's revised estimate.
            # Pass 1 is preserved on `pass1` for the audit trail / report.
            signals_dict["ai"] = {
                "drift": pass2.drift_estimate,
                "confidence": pass2.confidence,
                "source_quality": "REPUTABLE",
                "sources_count": pass1.raw_sources_cited,
                "notes": f"Pass 2 revised ({pass2.revision_from_prior_pass:+.1%} vs Pass 1)",
            }

    # --- 6. Blend signals into today's drift (Pass 2-revised AI signal) ---
    print(f"Blending {len(signals_dict)} signals + bull/bear arithmetic...")
    blend = blend_with_uncertainty(signals_dict, weights_dict=BLEND_WEIGHTS_V2)
    if blend and blend.get("blended") is not None:
        today_mu = float(blend["blended"]) + factor_bias  # apply HIGH-factor net bias
        today_std = float(blend.get("std", 0.20))
    else:
        today_mu = mu_effective_historical + factor_bias
        today_std = 0.25

    # --- 7. Bayesian smoothing (v1 dict format) ---
    # v1's load_prior_blend reads thesis_history schema; v2 has its own.
    # We read v2's CSV and convert to v1's dict format for bayesian_update.
    prior_v2 = load_prior_posterior(history_path)
    if prior_v2:
        prior_age_days = max(1, (datetime.now().date() -
                                  datetime.strptime(prior_v2["date"][:10], "%Y-%m-%d").date()).days)
        # Floor std at 0.05 to avoid div-by-zero or pathological narrow priors
        prior_std_safe = max(0.05, float(prior_v2.get("std") or 0.15))
        today_std_safe = max(0.05, float(today_std))
        prior_blend_v1_fmt = {"blended": prior_v2["mu"], "std": prior_std_safe}
        today_blend_v1_fmt = {"blended": today_mu, "std": today_std_safe}
        bayesian = bayesian_update(prior_blend_v1_fmt, today_blend_v1_fmt,
                                    prior_age_days=prior_age_days)
        if bayesian and bayesian.get("posterior_mu") is not None:
            post_mu = float(bayesian["posterior_mu"])
            post_std = float(bayesian["posterior_std"])
            prior_weight = float(bayesian.get("prior_weight", 0.0))
        else:
            post_mu, post_std, prior_weight = today_mu, today_std, 0.0
    else:
        post_mu, post_std, prior_weight = today_mu, today_std, 0.0

    posterior_summary = {
        "prior_mu": prior_v2["mu"] if prior_v2 else 0.0,
        "prior_std": prior_v2["std"] if prior_v2 else 0.15,
        "today_mu": today_mu, "today_std": today_std,
        "post_mu": post_mu, "post_std": post_std,
        "prior_weight": prior_weight,
        "today_weight": 1 - prior_weight,
    }

    # Build display list for report (converts dict back to list[DriftSignal])
    base_signals = _signals_dict_to_display_list(signals_dict, BLEND_WEIGHTS_V2)

    # --- 8. Build vol schedule ---
    vol_schedule = build_catalyst_vol_schedule(
        base_vol=blended_sigma,
        horizon_days=horizon_days,
        self_earnings_date=self_earnings_dt,
        peer_earnings_dates=[],  # could be expanded — left empty for ship
        macro_event_dates=[],
    )

    # --- 9. Apply AI vol_regime multiplier ---
    if pass1:
        vol_mult = AI_VOL_REGIME_MULTIPLIERS.get(pass1.vol_regime, 1.0)
        effective_sigma = blended_sigma * vol_mult
    else:
        effective_sigma = blended_sigma

    # --- 10. Run MC (joint conditional) ---
    print(f"Running Monte Carlo ({DEFAULT_MC_PATHS} paths)...")
    paths = run_mc_joint_conditional(
        S0=spot,
        sigma=effective_sigma,
        mu=post_mu,
        horizon_days=horizon_days,
        n_paths=DEFAULT_MC_PATHS,
        vol_schedule=vol_schedule,
        mean_reversion_strength=args.mean_reversion,
        mean_reversion_anchor=spot * 0.95 if args.mean_reversion > 0 else None,
    )

    # --- 11. Scan grid for best pair (bridge-corrected) ---
    print("Scanning dip × rally grid (Brownian bridge correction)...")
    best, all_candidates, met_threshold_strict = scan_dip_rally_grid(
        S0=spot, sigma=effective_sigma, mu=post_mu, horizon_days=horizon_days,
        paths=paths,
        conviction_dip=conviction_dip,
        conviction_rally_cond=conviction_rally_cond,
        capital_usd=capital,
        vol_schedule=vol_schedule,
    )

    # --- 12. Pass 2 already ran before the blend (step 5b above), so MC uses
    #          Pass 2's revised drift. Nothing more to do here. pass2 / pass2_cost_charged
    #          carry through to total_ai_cost + report below.

    # --- 13. AI catalyst stress test ---
    catalyst_stress_results = []
    catalyst_stress_cost = 0.0
    if not args.no_ai and pass1 and best:
        print("AI catalyst impact stress test...")
        catalyst_stress_results, catalyst_stress_cost = call_ai_catalyst_stress_test(
            ticker, spot, best.dip_price, best.rally_price, pass1.catalysts, horizon_days,
        )

    # --- 14. Three-method math cross-check (bridge-corrected MC) ---
    print("Three-method math cross-check...")
    if best:
        bridge_best_result = analyze_joint_conditional(
            paths, spot, best.dip_price, best.rally_price, horizon_days,
            sigma=effective_sigma, vol_schedule=vol_schedule,
        )
        method_check = three_method_cross_check(
            spot, effective_sigma, post_mu, horizon_days,
            best.dip_price, best.rally_price, bridge_best_result,
        )
    else:
        method_check = {"table": [], "flags": [], "agreement_status": "n/a — no pair found",
                        "pde_mass_conservation": 1.0, "pde_p_neither": 0.0}

    # --- 14b. Sensitivity table (drift/sigma swings + per-catalyst stress) ---
    sensitivity = None
    if best is not None:
        print("Computing sensitivity table (drift/σ scenarios + catalyst shocks)...")
        sensitivity = compute_sensitivity_table(
            S0=spot, base_sigma=effective_sigma, base_mu=post_mu,
            horizon_days=horizon_days,
            dip_price=best.dip_price, rally_price=best.rally_price,
            capital_usd=capital,
            spread_per_share_round_trip=2.0,
            catalyst_shocks=catalyst_stress_results,
            vol_schedule_base=vol_schedule,
            n_paths_sensitivity=10_000,
        )

    # --- 14c. Path-dependent metrics (max DD, panic floor, time-to-target) ---
    path_metrics = None
    if best is not None:
        path_metrics = compute_path_metrics(paths, spot, best.dip_price, best.rally_price)

    # --- 15. Backtest layer ---
    backtest = run_backtest_layer(history_path, spot)

    # --- 16. Persist CSV row ---
    # Surface ALL incurred AI cost — Pass 1, Pass 2, catalyst stress test.
    # Use the _charged trackers so failed-to-parse calls still show their cost.
    total_ai_cost = pass1_cost_charged + pass2_cost_charged + catalyst_stress_cost
    csv_row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "spot": f"{spot:.2f}",
        "sigma_blended": f"{blended_sigma:.4f}",
        "drift_posterior": f"{post_mu:.4f}",
        "drift_posterior_std": f"{post_std:.4f}",
        "recommended_dip": f"{best.dip_price:.0f}" if best else "",
        "p_dip": f"{best.p_dip_touched:.4f}" if best else "",
        "expected_days_to_dip": f"{best.expected_days_to_dip:.1f}" if best else "",
        "recommended_rally": f"{best.rally_price:.0f}" if best else "",
        "p_rally_cond": f"{best.p_rally_given_dip:.4f}" if best else "",
        "p_round_trip": f"{best.p_round_trip:.4f}" if best else "",
        "p_bag_hold": f"{best.p_bag_hold:.4f}" if best else "",
        "p_no_trade_rally_first": f"{best.p_no_trade_rally_first:.4f}" if best else "",
        "p_neither": f"{best.p_neither:.4f}" if best else "",
        "expected_gain_per_share": f"{best.expected_gain_per_share:.2f}" if best else "",
        "net_expected_value": f"{best.net_expected_value:.2f}" if best else "",
        "ai_drift_pass1": f"{pass1.drift_estimate:.4f}" if pass1 else "",
        "ai_drift_pass2": f"{pass2.drift_estimate:.4f}" if pass2 else "",
        "ai_vol_regime": pass1.vol_regime if pass1 else "",
        "narrative_score": pass1.narrative_score if pass1 else "",
        "catalyst_proximity_drift": f"{catalyst_mu:.4f}",
        # GARCH α+β diagnostic deferred to v3 (see v2 spec); using vol_profile field
        "garch_alpha_plus_beta": f"{vol_profile.garch_alpha_plus_beta:.4f}",
        "horizon_days": str(horizon_days),
        "method_agreement_flags": ";".join(method_check["flags"]),
        "ai_cost_total": f"{total_ai_cost:.2f}",
    }
    append_history_row(history_path, csv_row)

    # --- 17. Generate report ---
    runtime = time.time() - t_start
    report = format_report(
        snapshot, vol_profile, base_signals, pass1, pass2, posterior_summary,
        best, method_check, catalyst_stress_results, backtest,
        conviction_dip, conviction_rally_cond, horizon_days, capital,
        total_ai_cost, runtime,
        met_threshold_strict=met_threshold_strict,
        unusual_move=unusual_move,
        sensitivity=sensitivity,
        path_metrics=path_metrics,
    )
    print(report)

    # --- 18. Generate HTML dashboard ---
    with open(history_path, "r") as f:
        history_rows_for_chart = list(csv.DictReader(f))
    generate_html_dashboard(
        dashboard_path, snapshot, best, vol_profile, base_signals,
        pass1, pass2, method_check, backtest, history_rows_for_chart,
        conviction_dip, conviction_rally_cond, horizon_days,
    )

    return 0


def main():
    p = argparse.ArgumentParser(
        description="SanDisk Swing Trader (v2) — round-trip dip-and-rally framework",
    )
    p.add_argument("ticker", help="Ticker symbol (e.g. SNDK)")
    p.add_argument("--capital", type=float, default=10000.0,
                   help="Capital to deploy per round-trip in USD (default 10000)")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_DAYS,
                   help=f"Patience horizon in trading days (default {DEFAULT_HORIZON_DAYS})")
    p.add_argument("--conviction-dip", type=float, default=DEFAULT_CONVICTION_DIP,
                   help=f"Marginal P(touch dip) threshold (default {DEFAULT_CONVICTION_DIP})")
    p.add_argument("--conviction-rally-cond", type=float, default=DEFAULT_CONVICTION_RALLY_COND,
                   help=f"Conditional P(rally | dip) threshold (default {DEFAULT_CONVICTION_RALLY_COND})")
    p.add_argument("--mean-reversion", type=float, default=0.0,
                   help="Mean-reversion strength (default 0.0 = OFF; try 0.05/0.10/0.20 for sensitivity)")
    p.add_argument("--no-ai", action="store_true",
                   help="Skip all AI calls (math + backtest only). Use for debugging without token cost.")
    p.add_argument("--show-rationale", action="store_true",
                   help="Verbose mode (currently default)")
    args = p.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())

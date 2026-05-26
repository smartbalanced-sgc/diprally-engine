"""Single-ticker pipeline: data → vol → signals → AI → MC → grid → CSV → report.

W0 inlines the v2 monolith's run_pipeline. W2 adds multi-ticker orchestration
on top via a separate src/orchestrator.py.
"""
from __future__ import annotations

import argparse  # noqa: F401  (kept available for run_pipeline arg type hints)
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import ai_cache
from src.ai_tiers import resolve_tier, t0
from src.ambiguity import compute_ambiguity
from src.registry import resolve_peers
from src.sigma_classifier import (
    class_conviction,
    classify_sigma,
    reconcile_with_registry,
)
from src.ai_layer import (
    apply_catalyst_verification,
    build_ai_pass1_prompt,
    build_ai_pass2_prompt,
    call_ai_catalyst_stress_test,
    call_ai_catalyst_verification,
    call_ai_pass,
    parse_ai_pass1,
    parse_ai_pass2,
)
from src.config import (
    BACKTEST_MIN_SAMPLES,
    BAYESIAN_DEFAULT_PRIOR_STD,
    BAYESIAN_DEFAULT_TODAY_STD,
    BAYESIAN_STD_FLOOR,
    BLEND_WEIGHTS_V2,
    DRIFT_CAP,
    EV_HURDLE_BPS_OF_DIP,
    GARCH_FALLBACK_SIGMA,
    GRID_PREFILTER_LOOSENESS,
    MEAN_REVERSION_ANCHOR_PCT_BELOW_SPOT,
    PASS2_CLOSED_FORM_BRACKET_PCT,
    SENSITIVITY_SCENARIOS,
    SIGMA_CLASSES,
    PARABOLA_FILTER_MOM_30D_THRESHOLD,
    TREND_FILTER_MOM_30D_THRESHOLD,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MC_PATHS,
    MC_DISTRIBUTION,
    MODEL_OPUS,
    MODEL_SONNET,
)
from src.data_fetch import (
    FetchError,
    fetch_analyst_summary,
    fetch_analyst_targets,
    fetch_company_profile,
    fetch_fundamentals,
    fetch_grades_history,
    fetch_history,
    fetch_macro_indicators,
    fetch_next_earnings,
    fetch_options_iv,
    fetch_peer_history,
    fetch_sector_perf,
    fetch_short_interest,
)
from src.math_utils import (
    analyze_joint_conditional,
    build_catalyst_vol_schedule,
    closed_touch_down,
    closed_touch_up,
    compute_path_metrics,
    compute_realized_vol,
    compute_rsi_14,
    apply_enrichment_to_drift,
    enrichment_drift,
    fit_garch_11_full,
    precompute_first_touch_days,
    run_mc_joint_conditional,
    three_method_cross_check,
    triangulate_sigma,
)
from src.signals import (
    _none_signal,
    apply_bull_bear_arithmetic,
    bayesian_update,
    blend_with_uncertainty,
    compute_unusual_move_z,
    detect_swing_regime,
    parse_catalyst_date,  # noqa: F401  (re-export for callers)
    signal_from_analyst_targets,
    signal_from_catalyst_proximity,
    signal_from_fundamentals,
    signal_from_historical,
    signal_from_macro,
    signal_from_peer_rs,
    signal_from_revision_momentum,
    signal_from_sector,
    signal_from_sector_decoupling,
    signal_from_short_interest,
    signal_from_structural_narrative,
)


# =============================================================================
# Typed containers (v2 dataclasses)
# =============================================================================

@dataclass
class MarketSnapshot:
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
    near_unit_root: bool


@dataclass
class DriftSignal:
    name: str
    mu_annual: float
    confidence: str
    source_quality: str
    weight: float
    rationale: str
    is_absent: bool = False  # NONE_FOUND / None-drift signal — reporter prints "n/a"
    effective_weight: float = 0.0  # post LOW-halve, post normalization (see D-W2-12)


@dataclass
class AIPassOutput:
    pass_number: int
    drift_estimate: float
    drift_range: tuple
    confidence: str
    vol_regime: str
    narrative_score: str
    catalysts: list
    bull_factors: list
    bear_factors: list
    key_risks: list
    revision_from_prior_pass: Optional[float]
    cost_usd: float
    raw_sources_cited: int
    # 2026-05-24 audit fix: capture AI reasoning fields previously
    # requested in the prompt but discarded by the parser. Token cost
    # was already being paid; without capture the operator had no
    # audit trail for WHY each Pass 2 revision was made.
    narrative_evidence: list = field(default_factory=list)   # Pass 1 — [{claim, source}, ...]
    agreement_with_pass1: str = ""                            # Pass 2 — agree / partial_disagree / strong_disagree
    revision_reasoning: str = ""                              # Pass 2 — why drift was revised
    vol_regime_reasoning: str = ""                            # Pass 2 — why vol_regime
    narrative_reasoning: str = ""                             # Pass 2 — why narrative_score
    catalysts_reasoning: str = ""                             # Pass 2 — why catalyst set differs


@dataclass
class JointConditionalResult:
    dip_price: float
    rally_price: float
    p_dip_touched: float
    p_rally_given_dip: float
    p_round_trip: float
    p_bag_hold: float
    p_no_trade_rally_first: float
    p_neither: float
    expected_days_to_dip: float
    expected_days_dip_to_rally: float
    expected_gain_per_share: float
    expected_bag_hold_loss: float
    net_ev_per_share: float       # per-share EV (replaces capital-scaled net_expected_value)
    ev_pct_of_dip: float          # net_ev_per_share / dip_price (trader-comparable across tickers)


# =============================================================================
# Signal dict → display list adapter
# =============================================================================

def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text to <=max_chars at the last word boundary, append '…' if
    truncated. Avoids the D-W2-11 mid-parenthesis-chop class of bugs where
    naive [:N] truncation produced labels like 'Q4 guidance bar very high (rev $7.7'."""
    if len(text) <= max_chars:
        return text
    # Reserve 1 char for the ellipsis
    cut = text[: max_chars - 1]
    last_space = cut.rfind(" ")
    if last_space > max_chars // 2:  # only walk back if we don't lose too much
        cut = cut[:last_space]
    return cut + "…"


def _has_bearish_derating_catalyst(effective_ai, horizon_days: int) -> bool:
    """PR #41 / #45 helper (mirror of sacred #14, tightened): returns True
    iff Pass 1/Pass 2 surfaced at least one in-horizon catalyst with
    direction_risk == "bearish". This is the ONLY structural reason to
    expect a parabolic stock to mean-revert within a swing window.

    PR #45 design change: dropped "two-sided" from the accepted set.
    Two-sided catalysts (generic earnings, sector readthrough, macro)
    are the math layer's DEFAULT assumption — they don't specifically
    point toward de-rating. The parabola filter is a top-level VETO
    that should only relax for specific bearish catalysts (overhang
    closes, secondary offerings, regulatory actions, peer disappoint-
    ments). Otherwise +92%-in-30-days "two-sided earnings" lets every
    parabola slip through the gate, defeating the purpose.

    Asymmetry with sacred #14 is intentional:
      #14 (falling knife):  accepts bullish OR two-sided to override
                            (anything pointing up rescues a falling knife)
      #41 (parabola):       requires bearish specifically to override
                            (only a bearish thesis explains mean-reversion)

    In --no-ai mode effective_ai is None → False (strict reading
    refuses parabolic dip-buys without a thesis)."""
    if not effective_ai or not effective_ai.catalysts:
        return False
    from datetime import datetime as _dt
    from src.signals import parse_catalyst_date
    from src.market_calendar import add_trading_days
    today = _dt.now().date()
    # PR #76: horizon_days is TRADING days (MC dt=1/252). Calendar-day
    # math was creating a ~24-day dead zone at the tail (28% of the
    # actual horizon).
    horizon_end = add_trading_days(today, horizon_days)
    for c in effective_ai.catalysts:
        if not isinstance(c, dict):
            continue
        dir_risk = str(c.get("direction_risk", "")).lower()
        # PR #46: accept any "bearish*" variant (Pass 1/Pass 2 emit
        # "bearish", "bearish-skew", "bearish/skew" etc.). Still excludes
        # "two-sided" and "bullish*" — semantic intent unchanged from
        # PR #45 ("specifically bearish thesis required"), just more
        # lexically tolerant. Pass 2 commonly emits "bearish-skew" for
        # mean-reversion catalysts (e.g. "profit-taking after +204% YTD").
        if not dir_risk.startswith("bearish"):
            continue
        cdate = parse_catalyst_date(c.get("date_or_window", ""))
        if cdate is None:
            continue
        if today <= cdate <= horizon_end:
            return True
    return False


def _has_supporting_catalyst(effective_ai, horizon_days: int) -> bool:
    """Sacred decision #14 helper: returns True iff Pass 1/Pass 2 surfaced at
    least one in-horizon catalyst with direction_risk in (bullish, two-sided).

    Bearish-only catalysts do NOT rescue a falling knife — they confirm it.
    A two-sided catalyst counts because it provides a real event that could
    re-rate the stock either direction; the trader's thesis becomes 'will
    the dip be exhausted before the event resolves.'

    In --no-ai mode effective_ai is None → False (no catalysts known →
    can't disprove falling-knife → strict reading refuses).
    """
    if not effective_ai or not effective_ai.catalysts:
        return False
    from datetime import datetime as _dt
    from src.signals import parse_catalyst_date
    from src.market_calendar import add_trading_days
    today = _dt.now().date()
    # PR #76: horizon_days is TRADING days — see note on
    # _has_bearish_derating_catalyst above.
    horizon_end = add_trading_days(today, horizon_days)
    for c in effective_ai.catalysts:
        if not isinstance(c, dict):
            continue
        dir_risk = str(c.get("direction_risk", "")).lower()
        if dir_risk not in ("bullish", "two-sided"):
            continue
        cdate = parse_catalyst_date(c.get("date_or_window", ""))
        if cdate is None:
            continue
        if today <= cdate <= horizon_end:
            return True
    return False


def _signals_dict_to_display_list(signals_dict, weights, blend=None):
    """Convert v1 signal dict format → list[DriftSignal] for display.

    is_absent=True when source_quality == NONE_FOUND OR drift is None.
    Reporter prints "n/a" instead of "+0.0%" for absent signals — this
    prevents the W0/W1 silent-failure pattern where a broken signal looked
    like a legitimate zero-drift signal.

    effective_weight computed from the live blend's `weights` field (after
    LOW-halving, NONE_FOUND-zeroing) renormalized to sum-to-1.0 across the
    surviving set. Surfaces what's ACTUALLY driving the blend point estimate,
    not the nominal design weight (D-W2-12).
    """
    pretty_names = {
        "historical": "Historical (GARCH + enrichment)",
        "analyst": "Analyst (price-target-summary)",
        "sector": "Sector momentum",
        "macro": "Macro regime (VIX/SPY)",
        "short_interest": "Short interest (squeeze tail)",
        "fundamentals": "Fundamentals (FCF + leverage + margin trend)",
        "revision_momentum": "Analyst revision momentum (90d, time-decayed)",
        "peer_rs": "Peer RS (60d)",
        "sector_decoupling": "Sector decoupling (vs sector, 30d)",
        "ai": "AI analyst",
        "catalyst_proximity": "Catalyst proximity (AI-generated)",
        "narrative": "Structural narrative score",
    }

    # Compute effective normalized weights from the live blend, if provided.
    effective_weights: dict[str, float] = {}
    if blend and isinstance(blend.get("weights"), dict):
        post_gate = blend["weights"]  # already has LOW-halve / NONE_FOUND-zero applied
        total = sum(post_gate.values()) or 1.0
        effective_weights = {n: w / total for n, w in post_gate.items()}

    out: list[DriftSignal] = []
    for name, info in signals_dict.items():
        drift = info.get("drift")
        source_quality = str(info.get("source_quality", "PRIMARY"))
        is_absent = (drift is None) or (source_quality == "NONE_FOUND")
        if drift is None:
            drift = 0.0
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
            source_quality=source_quality,
            weight=float(weights.get(name, 0.0)),
            rationale=str(info.get("notes", "")),
            is_absent=is_absent,
            effective_weight=float(effective_weights.get(name, 0.0)),
        ))
    return out


# =============================================================================
# Grid scan — find best dip × rally pair maximizing net expected value
# =============================================================================

def scan_dip_rally_grid(
    S0,
    sigma,
    mu,
    horizon_days,
    paths,
    conviction_dip,
    conviction_rally_cond,
    sigma_class,
    vol_schedule=None,
):
    """Scan (dip × rally) grid with Brownian bridge correction.

    Returns (best, candidates, met_threshold_strict).

    W3 PR #22 (D-W3-1): grid step + depth + reach come from the
    per-σ-class table in config/diprally.yaml. Step is % of spot, so
    the engine is price-agnostic across the universe.

    W3 PR #23: friction is per-σ-class round-trip bps applied to the
    average traded notional (dip+rally)/2 — institutional convention,
    proportional to what the trader actually transacts on each leg.

    Sacred decision #6: no capital concept. EV is reported per-share + as
    a percent of dip-entry price. The trader sizes externally; engine is
    a recommendation tool, not a position-sizer.
    """
    n_paths, n_days = paths.shape
    class_entry = SIGMA_CLASSES[sigma_class]
    class_grid = class_entry.grid
    friction_bps_rt = class_entry.friction_bps_round_trip
    dip_step = S0 * class_grid.dip_step_pct
    rally_step = S0 * class_grid.rally_step_pct
    dip_min = S0 * (1.0 - class_grid.dip_max_depth_pct)
    dip_max = S0 * 0.99
    rally_min = S0 * 1.01
    rally_max = S0 * (1.0 + class_grid.rally_max_reach_pct)

    dip_grid = np.arange(dip_min, dip_max, dip_step)
    rally_grid = np.arange(rally_min, rally_max, rally_step)

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

            if p_dip < conviction_dip - GRID_PREFILTER_LOOSENESS:
                continue
            if p_rally_cond < conviction_rally_cond - GRID_PREFILTER_LOOSENESS:
                continue

            # PR #23: round-trip friction proportional to average leg
            # notional. Two legs at their respective prices; bps_rt
            # spread across the trip.
            friction_per_share = (
                (float(dip) + float(rally)) / 2.0 * friction_bps_rt / 10000.0
            )
            gain_per_share = float(rally) - float(dip) - friction_per_share
            bag_hold_loss_per_share = float(dip) - result["bag_hold_terminal_median"]
            net_ev_per_share = (
                result["p_round_trip"] * gain_per_share
                + result["p_bag_hold"] * (-bag_hold_loss_per_share)
            )
            # EV as % of dip entry — the trader-meaningful comparable across
            # tickers regardless of price level. SNDK $5/share on $1500 dip
            # vs LWLG $0.50/share on $13 dip both compute to ~30bps.
            ev_pct_of_dip = net_ev_per_share / float(dip)

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
                net_ev_per_share=net_ev_per_share,
                ev_pct_of_dip=ev_pct_of_dip,
            )
            candidates.append(jc)

    qualified = [
        c for c in candidates
        if c.p_dip_touched >= conviction_dip and c.p_rally_given_dip >= conviction_rally_cond
    ]
    if qualified:
        qualified.sort(key=lambda c: c.net_ev_per_share, reverse=True)
        best = qualified[0]
        met_threshold_strict = True
    else:
        candidates_sorted = sorted(candidates, key=lambda c: c.net_ev_per_share, reverse=True)
        best = candidates_sorted[0] if candidates_sorted else None
        met_threshold_strict = False

    return best, candidates, met_threshold_strict


# Sacred-decision verdict tree — single source of truth shared between
# the engine (writes verdict_state to CSV), the reporter headline_card
# (per-ticker view), and the orchestrator dashboard (aggregate view).
# Mirrors reporter._headline_card priority exactly; if the priority
# changes there, change it here. Audit finding 2026-05-24: previously
# the dashboard reconstructed verdict from CSV dip/EV alone, which
# silently misclassified sacred-#14 / #18 / #16 refusals as BUY or
# WAIT.
ENGINE_VERDICT_STATES = (
    "REFUSED-TREND", "REFUSED-PARABOLA", "REFUSED-METHOD",
    "REFUSED-EV", "WAIT", "BELOW-THRESHOLD", "NEGATIVE-EV", "BUY",
)


def _compute_verdict_state(*, best, met_threshold_strict, method_check,
                            trend_filter_refused: bool,
                            parabola_filter_refused: bool,
                            ev_hurdle_refused: bool) -> str:
    """Compute the engine-level verdict state for this run. Pure
    function; no side effects."""
    if trend_filter_refused and best is not None:
        return "REFUSED-TREND"
    if parabola_filter_refused and best is not None:
        return "REFUSED-PARABOLA"
    # Method-disagreement nulls `best`; distinguish from genuine
    # no-qualifying-pair WAIT.
    if (best is None and method_check
            and method_check.get("refused")
            and not method_check.get("is_anchor")):
        return "REFUSED-METHOD"
    if ev_hurdle_refused and best is not None:
        return "REFUSED-EV"
    if best is None:
        return "WAIT"
    if not met_threshold_strict:
        return "BELOW-THRESHOLD"
    if best.net_ev_per_share < 0:
        return "NEGATIVE-EV"
    return "BUY"


def _all_refusal_reasons(*, best, method_check,
                          trend_filter_refused: bool,
                          parabola_filter_refused: bool,
                          ev_hurdle_refused: bool) -> list:
    """PR #81 (audit #13): collect ALL refusals that fired this run.

    `_compute_verdict_state` returns a single label (operator-facing
    headline). When method-disagreement nulls `best`, the EV-hurdle
    branch can't fire, so a run where BOTH method AND EV would refuse
    is recorded as just REFUSED-METHOD — EV refusals get under-counted
    in W10 calibration aggregates.

    This helper returns the full list in the same priority order as
    `_compute_verdict_state`. CSV captures it in `refusal_reasons_all`
    so analytics can see joint refusals. UI keeps showing one label.
    """
    reasons = []
    if trend_filter_refused:
        reasons.append("TREND")
    if parabola_filter_refused:
        reasons.append("PARABOLA")
    if (method_check and method_check.get("refused")
            and not method_check.get("is_anchor")):
        reasons.append("METHOD")
    if ev_hurdle_refused:
        # ev_hurdle_refused is set BEFORE method-disagreement nulls
        # best, so this captures EV-failed runs that the verdict-state
        # tree would have hidden under REFUSED-METHOD.
        reasons.append("EV")
    return reasons


def compute_sensitivity_table(
    S0, base_sigma, base_mu, horizon_days,
    dip_price, rally_price,
    sigma_class,
    catalyst_shocks=None, vol_schedule_base=None, n_paths_sensitivity=10_000,
):
    """Small MCs with shifted (drift, sigma) for each scenario.

    Scenario list sourced from config sensitivity_scenarios (sacred #17,
    D-W2-7). Researchers can add/remove/re-parameterize rows via YAML
    without code edits. Per-catalyst stress shocks from AI continue to
    append after these baseline scenarios.

    W3 PR #23: friction is per-σ-class bps applied to (dip+rally)/2,
    matching the grid scan's institutional round-trip convention.
    """
    friction_bps_rt = SIGMA_CLASSES[sigma_class].friction_bps_round_trip
    friction_per_share = (dip_price + rally_price) / 2.0 * friction_bps_rt / 10000.0
    if catalyst_shocks is None:
        catalyst_shocks = []
    scenarios = [
        (s["label"],
         base_mu + s["drift_offset"],
         base_sigma * s["sigma_multiplier"])
        for s in SENSITIVITY_SCENARIOS
    ]
    for shock in catalyst_shocks[:3]:
        try:
            name = str(shock.get("catalyst_name") or shock.get("name") or "catalyst")
            pp = float(shock.get("drift_shock_pp_on_disappointment") or 0.0)
            # D-W2-11 fix: word-boundary truncation. Previous behavior was
            # name[:35] which chopped mid-parenthesis (e.g. "rev $7.7"
            # truncated from "Q4 guidance bar very high (rev $7.75-8.25B, EPS $30-33)").
            # Now: truncate at most 32 chars, walk back to last space, add ellipsis.
            max_label_width = 32
            short_name = _truncate_at_word_boundary(name, max_label_width)
            label = f"{short_name} ({pp:+.0f}pp)"
            scenarios.append((label, base_mu + pp / 100.0, base_sigma))
        except (TypeError, ValueError):
            continue

    rows = []
    for label, mu_s, sigma_s in scenarios:
        if vol_schedule_base is not None and base_sigma > 0:
            scale = sigma_s / base_sigma
            vs = vol_schedule_base * scale
        else:
            vs = None
        # W9 PR #48: sensitivity MCs use the same distribution/df as
        # the main MC — keep stress-test innovations consistent with
        # the recommendation's underlying assumption.
        paths_s = run_mc_joint_conditional(
            S0=S0, sigma=sigma_s, mu=mu_s,
            horizon_days=horizon_days, n_paths=n_paths_sensitivity,
            vol_schedule=vs, seed=42 + len(rows),
            distribution=MC_DISTRIBUTION.default,
            df=MC_DISTRIBUTION.per_class.get(
                sigma_class, MC_DISTRIBUTION.default_df,
            ),
        )
        # PR #78 (audit #10): pass the same per-scenario seed offset
        # used for run_mc_joint_conditional above so the bridge-touch
        # randomness inside analyze_joint_conditional varies across
        # scenarios. Previously all scenarios shared seeds (42, 43) →
        # bridge correction was identical across stress points,
        # defeating scenario independence.
        result = analyze_joint_conditional(
            paths_s, S0, dip_price, rally_price, horizon_days,
            sigma=sigma_s, vol_schedule=vs,
            seed=42 + len(rows),
        )
        gain_per_share = rally_price - dip_price - friction_per_share
        bag_hold_loss = dip_price - result["bag_hold_terminal_median"]
        net_ev_per_share = (
            result["p_round_trip"] * gain_per_share
            + result["p_bag_hold"] * (-bag_hold_loss)
        )
        ev_pct_of_dip = net_ev_per_share / dip_price
        rows.append({
            "label": label,
            "mu": mu_s,
            "sigma": sigma_s,
            "p_round_trip": result["p_round_trip"],
            "p_bag_hold": result["p_bag_hold"],
            "p_no_trade": result["p_no_trade_rally_first"],
            "net_ev_per_share": net_ev_per_share,
            "ev_pct_of_dip": ev_pct_of_dip,
        })
    return rows


# =============================================================================
# CSV persistence
# =============================================================================

# CSV schema bump for sacred #6 (no capital concept):
# - Drop net_expected_value (was capital × per-share, capital-scaled)
# - Add net_ev_per_share + ev_pct_of_dip (capital-independent, comparable
#   across tickers regardless of price level)
# Old history files written before this bump are not backwards-compatible
# with the new schema. N=1 day at this point (no calibration accumulated)
# so the clean break is acceptable.
CSV_COLUMNS = [
    "date", "spot", "sigma_blended", "sigma_class",
    "drift_posterior", "drift_posterior_std",
    "recommended_dip", "p_dip", "expected_days_to_dip",
    "recommended_rally", "p_rally_cond",
    "p_round_trip", "p_bag_hold", "p_no_trade_rally_first", "p_neither",
    "expected_gain_per_share", "net_ev_per_share", "ev_pct_of_dip",
    "ai_drift_pass1", "ai_drift_pass2", "ai_vol_regime",
    "narrative_score", "catalyst_proximity_drift",
    "garch_alpha_plus_beta", "horizon_days",
    "method_agreement_flags", "ai_cost_total", "data_source",
    "ai_tier", "ambiguity_score",
    # W10 PR #47 — calibration harness. Outcome fields populated by
    # src.calibration.resolve_outcomes() at each engine run; the row's
    # prediction is locked at write time but its outcome accumulates as
    # the horizon window plays out. Empty strings until resolved.
    "outcome_status",          # OPEN / RESOLVED / EXPIRED
    "dip_touched",             # 1 / 0 (within horizon)
    "dip_touch_day",           # day index from prediction date (0..horizon-1)
    "rally_touched_after_dip", # 1 / 0 — rally after the dip touch
    "rally_touch_day",         # day index from prediction date
    "round_trip_completed",    # 1 / 0 — sacred ordering: dip THEN rally
    "bag_hold_realized",       # 1 / 0 — dip touched, rally never (or close < dip at horizon)
    "realized_max_drawdown",   # decimal — max DD from spot during window
    "realized_terminal_return",# decimal — terminal close vs spot
    "resolved_at",             # YYYY-MM-DD when status flipped to RESOLVED/EXPIRED
    # W10 PR #54 — D-W10-1 catalyst-accuracy capture. Snapshots Pass 1
    # and Pass 2's catalyst output AND the verification verdicts at run
    # time so future analysis can compute per-ticker hallucination rates
    # by checking whether each predicted catalyst actually occurred as
    # described. Compact JSON serialization keeps the CSV machine-
    # parsable without exploding row width. Empty string when AI didn't
    # run (T0 tier) or no catalysts surfaced.
    "pass1_catalysts_json",      # JSON list of Pass 1 catalysts (name/date/type/direction/magnitude)
    "pass2_catalysts_json",      # JSON list of Pass 2 revised catalysts (same shape)
    "verification_verdicts_json",# JSON list aligned to top-3 Pass 2 catalysts: verdict + reasoning
    # W10 PR #61 — catalyst-stress capture. The AI stress test models
    # "drift shock if catalyst disappoints 20%" per top-3 catalyst.
    # Currently shown in the report but not persisted. Capturing now
    # so future W10 analysis can correlate predicted stress shocks
    # against realized drawdowns once 30+ days of resolved predictions
    # accumulate ("did the predicted -12pp earnings shock match the
    # realized post-earnings drift?").
    "catalyst_stress_json",      # JSON list of stress shocks (name + drift_shock_pp)
    # 2026-05-24 audit fix — engine-level verdict_state, single source of
    # truth for sacred-decision refusals. Mirrors reporter headline_card
    # priority. Orchestrator aggregate dashboard reads this directly
    # instead of reconstructing from dip/EV (which silently misclassified
    # sacred-#14 / #18 / #16 refusals as BUY/WAIT).
    "verdict_state",
    # 2026-05-24 audit fix round 2 — capture Pass 2's agreement classification
    # and per-revision reasoning to CSV. The reporter already surfaces them
    # (audit-round-1 work) but they weren't being persisted, breaking the
    # D-W10-1 future analysis that wants to correlate "strong_disagree"
    # frequency / revision rationale patterns with realized outcomes.
    "pass2_agreement",           # "" / "agree" / "partial_disagree" / "strong_disagree"
    "pass2_revision_reasoning",  # truncated text
    # PR #81 (audit #13) — comma-separated list of ALL refusals that
    # fired this run (TREND / PARABOLA / METHOD / EV). `verdict_state`
    # still records a single headline label; this column lets W10
    # calibration aggregates count joint refusals correctly (EV failures
    # masked by METHOD failures were previously undercounted).
    "refusal_reasons_all",
]


def _compact_catalysts_json(catalysts) -> str:
    """W10 PR #54 helper: serialize a Pass 1/Pass 2 catalyst list for
    CSV capture. Only retains the fields D-W10-1 analysis needs
    (name / type / date_or_window / direction_risk / magnitude). Drops
    'sources' and any verification_* fields appended by PR #33's
    apply_catalyst_verification — those go in their own column.
    Empty list → "" (engine doesn't write JSON-for-empty)."""
    import json as _json
    if not catalysts:
        return ""
    keep = []
    for c in catalysts:
        if not isinstance(c, dict):
            continue
        keep.append({
            k: c.get(k) for k in (
                "name", "type", "date_or_window",
                "direction_risk", "magnitude",
            )
        })
    if not keep:
        return ""
    return _json.dumps(keep, separators=(",", ":"))


def _compact_verdicts_json(verifications) -> str:
    """W10 PR #54 helper: serialize PR #33's catalyst-verification
    verdicts for CSV capture. Aligns 1:1 with the top-3 Pass 2 catalysts."""
    import json as _json
    if not verifications:
        return ""
    keep = []
    for v in verifications:
        if not isinstance(v, dict):
            continue
        keep.append({
            "catalyst_name": v.get("catalyst_name", ""),
            "verdict": v.get("verdict", "UNVERIFIED"),
            "reasoning": v.get("reasoning", ""),
        })
    if not keep:
        return ""
    return _json.dumps(keep, separators=(",", ":"))


def _compact_stress_json(stress_results) -> str:
    """W10 PR #61 helper: serialize catalyst stress test results for
    CSV capture. AI emits {catalyst_name, drift_shock_pp_on_disappointment,
    reasoning} per top-3 catalyst. We keep name + shock for the future
    W10 analysis ("did the predicted shock match realized drift?"),
    drop reasoning (too verbose for CSV; available in stdout log)."""
    import json as _json
    if not stress_results:
        return ""
    keep = []
    for s in stress_results:
        if not isinstance(s, dict):
            continue
        try:
            shock = float(s.get("drift_shock_pp_on_disappointment") or 0.0)
        except (TypeError, ValueError):
            continue
        keep.append({
            "name": str(s.get("catalyst_name", "")),
            "shock_pp": shock,
        })
    if not keep:
        return ""
    return _json.dumps(keep, separators=(",", ":"))


def append_history_row(history_path, row):
    """Write a row, replacing any existing row with the same date (#11)."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    today_str = row.get("date", "")

    if not history_path.exists():
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        return

    try:
        with open(history_path, "r", newline="") as f:
            existing = list(csv.DictReader(f))
    except Exception:
        existing = []

    existing = [r for r in existing if r.get("date", "") != today_str]
    existing.append(row)

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in existing:
            writer.writerow(r)


def load_prior_posterior(history_path):
    """Most-recent row's posterior drift for Bayesian smoothing.

    Same-day guard (#12): if last row is today's, skip prior.
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
        try:
            if last_date_str:
                last_dt = datetime.strptime(last_date_str[:10], "%Y-%m-%d").date()
                if last_dt == datetime.now().date():
                    print(f"   Bayesian prior skipped: last row is from today ({last_date_str}); "
                          f"same-day artifact prevention.")
                    return None
        except ValueError:
            pass
        mu_raw = last.get("drift_posterior", "")
        if mu_raw in (None, ""):
            return None
        return {
            "mu": float(mu_raw),
            "std": float(last.get("drift_posterior_std") or BAYESIAN_DEFAULT_PRIOR_STD),
            "date": last_date_str,
        }
    except Exception:
        return None


# =============================================================================
# Backtest layer
# =============================================================================

def run_backtest_layer(history_path, current_price):
    """Walk through CSV history, compute calibration metrics."""
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

    dip_predictions_resolved = 0
    dip_hits = 0
    rally_predictions_resolved = 0
    rally_hits = 0

    today = datetime.now().date()
    from src.market_calendar import trading_days_after as _tda
    for row in rows:
        try:
            row_date = datetime.strptime(row["date"][:10], "%Y-%m-%d").date()
        except Exception:
            continue
        # PR #76: horizon is TRADING days (MC dt=1/252). Was using
        # calendar elapsed → over-resolved by ~28%.
        td_elapsed = _tda(row_date, today)
        # PR #78 (audit #9): coerce empty-string and None safely.
        # `int(row.get(k, default))` returns the default only when the
        # key is MISSING; empty-string cells pass through and raise
        # ValueError, swallowed by the surrounding except and silently
        # dropping the row from backtest aggregation.
        horizon = int(row.get("horizon_days") or DEFAULT_HORIZON_DAYS)
        if td_elapsed < horizon:
            continue
        try:
            dip_pred = float(row.get("recommended_dip", 0))
            rally_pred = float(row.get("recommended_rally", 0))
        except Exception:
            continue
        dip_predictions_resolved += 1
        rally_predictions_resolved += 1

    return {
        "n_samples": n,
        "sufficient_data": True,
        "dip_predictions_resolved": dip_predictions_resolved,
        "rally_predictions_resolved": rally_predictions_resolved,
        "per_day_status": _build_per_day_status(rows, current_price),
    }


def _build_per_day_status(rows, current_price):
    """For each prior prediction, classify status."""
    today = datetime.now().date()
    from src.market_calendar import trading_days_after as _tda
    out = []
    for row in rows:
        try:
            row_date = datetime.strptime(row["date"][:10], "%Y-%m-%d").date()
        except Exception:
            continue
        # PR #76: count trading days, not calendar days, for the
        # 'remaining' bar in the per-day status display.
        td_elapsed = _tda(row_date, today)
        # PR #78 (audit #9): coerce empty-string and None safely.
        # `int(row.get(k, default))` returns the default only when the
        # key is MISSING; empty-string cells pass through and raise
        # ValueError, swallowed by the surrounding except and silently
        # dropping the row from backtest aggregation.
        horizon = int(row.get("horizon_days") or DEFAULT_HORIZON_DAYS)
        remaining = max(0, horizon - td_elapsed)
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
            "days_elapsed": td_elapsed,
            "remaining": remaining,
            "status": status,
        })
    return out


# =============================================================================
# Main pipeline (single ticker)
# =============================================================================

def run_pipeline(args) -> int:
    t_start = time.time()
    ticker = args.ticker.upper()
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        print("ERROR: FMP_API_KEY not set")
        return 1

    horizon_days = args.horizon
    # W3: conviction thresholds resolved AFTER blended_sigma is known so the
    # σ-class default table can apply. CLI flags (when supplied) still win.
    cli_conviction_dip = args.conviction_dip
    cli_conviction_rally_cond = args.conviction_rally_cond

    # W4 PR #27: resolve AI tier once. --no-ai forces T0 (math only);
    # otherwise use the explicit --tier (defaults to T3 to preserve
    # pre-W4 single-ticker behavior). The multi-ticker broker (PR #29)
    # will override --tier per ticker under the $2/day cap.
    tier_name = getattr(args, "tier", None) or "T3"
    tier = t0() if args.no_ai else resolve_tier(tier_name)
    print(f"AI tier: {tier.name}  (estimated cost ${tier.estimated_cost_usd:.2f})")

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"round_trip_history_{ticker}.csv"
    dashboard_path = output_dir / f"{ticker.lower()}_dipnrally_dashboard.html"

    # --- 1. Fetch data ---
    print(f"Fetching data for {ticker}...")
    try:
        history_df = fetch_history(ticker, api_key, DEFAULT_LOOKBACK_DAYS)
    except FetchError as e:
        # Graceful exit instead of Python stack trace. W5 batch orchestrator
        # will catch the same FetchError to skip the ticker and continue.
        print(f"ERROR: {e}")
        return 1
    # PR #73 — limited-history detection. GARCH + σ-triangulation are
    # unreliable when the available price series is shorter than the
    # institutional standard (250 trading days ≈ 12 months). Broker
    # forces T2 critique for these to compensate. Flag also surfaces
    # as a reliability chip on the per-ticker dashboard.
    from src.broker import LIMITED_HISTORY_THRESHOLD
    history_bars = len(history_df) if history_df is not None else 0
    limited_history = history_bars > 0 and history_bars < LIMITED_HISTORY_THRESHOLD
    if limited_history:
        print(f"   ⚠ Limited history: only {history_bars} bars available "
              f"(< {LIMITED_HISTORY_THRESHOLD} threshold). Math layer "
              f"uncertainty higher than the σ-class baseline.")
    data_source = history_df.attrs.get("data_source", "unknown") if history_df is not None else "unknown"
    if data_source == "yfinance":
        print(f"   ℹ data source: yfinance (FMP fell back — see warning above)")
    if history_df is None or history_df.empty:
        print(f"ERROR: empty history returned for {ticker}")
        return 1
    spot = float(history_df["Close"].iloc[-1])
    closes_series = history_df["Close"]
    closes = closes_series.values
    returns = np.log(closes_series / closes_series.shift(1)).dropna()

    rsi = compute_rsi_14(closes_series)
    mom_5d = float((closes[-1] / closes[-6] - 1.0)) if len(closes) > 5 else 0.0
    mom_30d = float((closes[-1] / closes[-31] - 1.0)) if len(closes) > 30 else 0.0
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

    profile_beta = None
    try:
        profile_beta = float(profile.get("beta") or 1.0)
    except (TypeError, ValueError):
        profile_beta = 1.0
    unusual_move = compute_unusual_move_z(history_df, beta=profile_beta, lookback=60)

    # --- 2. Volatility profile ---
    print("Computing volatility triangulation (GARCH α+β fit)...")
    garch = fit_garch_11_full(returns)
    if garch["fit_ok"] and garch["forecast_variance"] > 0:
        garch_sigma = float(np.sqrt(garch["forecast_variance"] * 252))
    else:
        garch_sigma = float(returns.tail(90).std() * np.sqrt(252)) if len(returns) >= 90 else GARCH_FALLBACK_SIGMA
    alpha_plus_beta = float(garch["alpha"] + garch["beta"])

    realized_vol_dict = compute_realized_vol(returns, windows=(30, 60, 90))
    iv_data = fetch_options_iv(ticker, target_dte_days=horizon_days)

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

    # W3: classify σ-class from blended_sigma (structural, pre AI vol_regime).
    # Data wins for current run; registry hint surfaces as advisory only.
    auto_sigma_class = classify_sigma(blended_sigma)
    sigma_class, sigma_class_mismatch = reconcile_with_registry(
        ticker, auto_sigma_class
    )
    class_dip, class_rally_cond = class_conviction(sigma_class)
    conviction_dip = (
        cli_conviction_dip if cli_conviction_dip is not None else class_dip
    )
    conviction_rally_cond = (
        cli_conviction_rally_cond
        if cli_conviction_rally_cond is not None
        else class_rally_cond
    )

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

    # --- 3. Drift base + 8 signals ---
    print("Computing 8 base drift signals...")
    # DRIFT_CAP imported from config (sacred #17 — D-W2-5).
    mu_hist = float(returns.mean() * 252)
    mu_capped = max(-DRIFT_CAP, min(DRIFT_CAP, mu_hist))
    enr = enrichment_drift(rsi, mom_5d)
    # PR #80 (audit #11): see math_utils.apply_enrichment_to_drift for
    # the full reasoning. Pre-fix this line was
    #   mu_effective_historical = mu_capped + enr * 252 / horizon_days
    # which made shorter horizons produce LARGER drift adjustments.
    mu_effective_historical = apply_enrichment_to_drift(mu_capped, rsi, mom_5d)

    targets = fetch_analyst_targets(ticker, api_key)
    summary = fetch_analyst_summary(ticker, api_key)
    sector_perf = fetch_sector_perf(sector, api_key) if sector and sector != "Unknown" else None
    regime = detect_swing_regime(rsi, mom_5d, mom_30d * 100,
                                  blended_sigma, ytd_return * 100)
    macro = fetch_macro_indicators(api_key)
    # Sacred decision #15: insider signal dropped (Form 4 lag + noise).
    short_data = fetch_short_interest(ticker, api_key)
    # W6 PR #34: TTM FCF + leverage + margin trend.
    fundamentals = fetch_fundamentals(ticker, api_key, market_cap=market_cap)
    # W6 PR #35: analyst upgrade/downgrade history.
    grades_history = fetch_grades_history(ticker, api_key)

    # Peer resolution via registry (D-W2-1 closed). CLI --peers is an override;
    # absent --peers falls back to config/diprally.yaml's per-ticker entry
    # (stock_peers preferred, etf_peer if no stocks, [] if neither configured).
    # Tickers not in the universe get [] and the peer_rs signal degrades to
    # _none_signal cleanly — no SNDK-specific hardcode anywhere (sacred #4).
    if args.peers:
        peer_tickers = list(args.peers)
    else:
        peer_tickers = resolve_peers(ticker)
    # CRITICAL: peer history needs DEFAULT_LOOKBACK_DAYS calendar days (730), NOT 60.
    # signal_from_peer_rs computes a 60-trading-day return which needs 61 trading
    # bars (~85 calendar days). 60 calendar days = ~43 trading bars — insufficient.
    # Previously this silently failed: n_day_return returned None on every peer
    # fetch, signal_from_peer_rs returned _none_signal, and the display masked
    # the absence as "+0.0% LOW". Result: 10% of the blend weight was inert from
    # W0 through W1 across every smoke run. Aligning peer lookback with own-ticker
    # lookback eliminates the implicit calendar-vs-trading-days assumption and
    # future-proofs against signals that want longer lookback windows.
    peer_dfs = fetch_peer_history(peer_tickers, api_key,
                                    lookback_days=DEFAULT_LOOKBACK_DAYS) if peer_tickers else {}

    self_earnings = fetch_next_earnings(ticker, api_key)
    self_earnings_dt = None
    if self_earnings:
        try:
            self_earnings_dt = datetime.strptime(self_earnings.get("date", "")[:10], "%Y-%m-%d")
        except Exception:
            pass

    # Peer earnings within horizon — fed into the catalyst-aware vol_schedule.
    # Previously this was left empty, meaning MU/WDC/competitor earnings in
    # SNDK's horizon never spiked SNDK's σ. Sacred decision #9 (bridge MC)
    # depends on vol_schedule being accurate; passing [] was degrading bridge
    # fidelity on every full-AI run with peer earnings in window.
    peer_earnings_dts = []
    from src.market_calendar import trading_days_after as _tda
    today_d = datetime.now().date()
    for p in peer_tickers:
        try:
            pe = fetch_next_earnings(p, api_key)
            if pe and pe.get("date"):
                dt = datetime.strptime(pe["date"][:10], "%Y-%m-%d")
                # PR #76: horizon_days is TRADING days. Calendar-day
                # filter was dropping peer events 60-84 cal days out
                # that fall inside the 60-trading-day horizon.
                td_away = _tda(today_d, dt.date())
                if 0 <= td_away < horizon_days:
                    peer_earnings_dts.append(dt)
                    print(f"   Peer earnings in horizon: {p} on {pe['date']} ({td_away} trading d)")
        except Exception as e:
            print(f"   WARNING: peer earnings fetch failed for {p}: {e}")

    signals_dict = {
        "historical": signal_from_historical(mu_effective_historical, mu_hist, blended_sigma),
        "analyst": signal_from_analyst_targets(targets, spot,
                                                price_history_df=history_df,
                                                summary=summary),
        "sector": signal_from_sector(sector_perf, swing_regime=regime),
        "macro": signal_from_macro(macro),
        # sacred #15: insider dropped (D-W2-16)
        "short_interest": signal_from_short_interest(short_data),
        "fundamentals": signal_from_fundamentals(fundamentals),
        "revision_momentum": signal_from_revision_momentum(grades_history),
        "peer_rs": signal_from_peer_rs(history_df, peer_dfs, lookback_days=60, ticker=ticker),
        "sector_decoupling": signal_from_sector_decoupling(history_df, sector_perf,
                                                            lookback_days=30, ticker=ticker),
    }

    # --- 4. AI Pass 1 (cache-aware) ---
    # Same-day cache: if today's run for this ticker already exists and spot
    # has moved < 1%, replay the cached AI outputs with cost = $0.00. Cache
    # invalidation triggers a fresh run. Sacred decision #11 extension.
    pass1 = None
    pass1_cost_charged = 0.0
    pass2 = None
    pass2_cost_charged = 0.0
    cached_stress_results = None  # set on cache hit, used in step 13
    cached_stress_cost = 0.0
    cache_hit = False

    # Raw outputs accumulated for end-of-pipeline cache write (miss path only).
    pass1_raw_for_cache = None
    pass1_sources_for_cache = 0
    pass2_raw_for_cache = None

    # PR #38: --bust-cache forces a fresh AI run even when today's payload
    # exists. Used to re-validate newly-deployed AI steps (e.g. PR #33
    # catalyst verification) on tickers whose cache predates the new step.
    bust_cache = getattr(args, "bust_cache", False)
    # PR #77: pass the current tier so ai_cache can refuse a lower-tier
    # replay. A T1 cache (Pass 1 only) cannot serve a T3 request, which
    # requires Pass 2 + verification + stress. Previously the engine
    # claimed tier=T3 at $0.00 cost while serving T1 data.
    cache_payload = (None if (not tier.runs_ai or bust_cache)
                     else ai_cache.get_cached(ticker, spot,
                                              current_tier=tier.name))
    if bust_cache:
        print("AI cache bypass (--bust-cache): forcing fresh Pass 1/2/verify/stress")
    if cache_payload:
        print(f"   AI cache HIT for {ticker} ({cache_payload.get('date')}, "
              f"spot ${cache_payload.get('spot'):.2f}) — Pass 1/2/stress replayed at $0.00")
        cache_hit = True
        p1_raw = cache_payload.get("pass1_raw")
        p1_sources = int(cache_payload.get("pass1_sources", 0))
        if p1_raw:
            pass1 = parse_ai_pass1(p1_raw, p1_sources, 0.0)  # cost=0.00 on cache hit
        p2_raw = cache_payload.get("pass2_raw")
        if p2_raw and pass1:
            pass2 = parse_ai_pass2(p2_raw, pass1, 0.0)
        cached_stress_results = cache_payload.get("stress_results") or []
        cached_stress_cost = 0.0
    elif not tier.runs_ai:
        print(f"AI Pass 1 skipped (tier {tier.name})")
    else:
        print(f"AI Pass 1 (data gathering + multi-hypothesis catalysts) — model={tier.pass1_model}, web_search≤{tier.pass1_web_search_max}")
        display_signals_for_prompt = _signals_dict_to_display_list(signals_dict, BLEND_WEIGHTS_V2)
        pass1_prompt = build_ai_pass1_prompt(
            ticker, snapshot, vol_profile, horizon_days, display_signals_for_prompt,
            self_earnings_dt, peer_tickers,
        )
        pass1_raw, pass1_cost, pass1_sources = call_ai_pass(
            pass1_prompt, max_tokens=tier.pass1_max_tokens, pass_label="Pass 1",
            model=tier.pass1_model, web_search_max_uses=tier.pass1_web_search_max,
        )
        pass1 = parse_ai_pass1(pass1_raw, pass1_sources, pass1_cost) if pass1_raw else None
        pass1_cost_charged = pass1_cost
        pass1_raw_for_cache = pass1_raw
        pass1_sources_for_cache = pass1_sources

    # --- 5. AI Pass 2 (adversarial critique). Runs BEFORE the AI-derived
    #        signals so Pass 2's revised vol_regime / narrative / catalysts
    #        drive those signals, not Pass 1's. Sacred decision #7 in full.
    if cache_hit:
        # pass2 already loaded from cache_payload above
        pass2_raw_for_cache = cache_payload.get("pass2_raw") if cache_payload else None
    elif not tier.runs_ai:
        print(f"AI Pass 2 skipped (tier {tier.name})")
        pass2_raw_for_cache = None
    elif tier.pass2_model is None:
        print(f"AI Pass 2 skipped (tier {tier.name} has no Pass 2)")
        pass2_raw_for_cache = None
    elif pass1:
        print("AI Pass 2 (adversarial critique — revises drift / vol_regime / narrative / catalysts)...")
        T_years = horizon_days / 252.0
        prelim_mu_for_closed = float(pass1.drift_estimate)
        try:
            bracket = PASS2_CLOSED_FORM_BRACKET_PCT
            p_up_10 = closed_touch_up(spot, spot * (1.0 + bracket), T_years, prelim_mu_for_closed, blended_sigma)
            p_down_10 = closed_touch_down(spot, spot * (1.0 - bracket), T_years, prelim_mu_for_closed, blended_sigma)
            mc_marginal_summary = {
                "p_up": f"{p_up_10*100:.0f}%",
                "p_down": f"{p_down_10*100:.0f}%",
                "bracket_pct_str": f"{bracket*100:.0f}%",
            }
        except Exception:
            mc_marginal_summary = {
                "p_up": "n/a", "p_down": "n/a",
                "bracket_pct_str": f"{PASS2_CLOSED_FORM_BRACKET_PCT*100:.0f}%",
            }
        sigma_summary = {"blended": blended_sigma, "divergence": divergence_pp}
        pass2_prompt = build_ai_pass2_prompt(
            ticker, snapshot, pass1, mc_marginal_summary, sigma_summary,
            None,
        )
        # Pass 2 critique uses Sonnet 4.6 (structured-output JSON critique,
        # doesn't need Opus depth) with no web_search (relies on Pass 1's
        # sourced material + math context embedded in the prompt). Saves
        # ~$0.40-0.50 per full-AI run vs Opus+web.
        pass2_raw, pass2_cost, _ = call_ai_pass(
            pass2_prompt, max_tokens=tier.pass2_max_tokens, pass_label="Pass 2",
            model=tier.pass2_model, web_search_max_uses=0,
        )
        pass2_cost_charged = pass2_cost
        pass2_raw_for_cache = pass2_raw
        if pass2_raw:
            pass2 = parse_ai_pass2(pass2_raw, pass1, pass2_cost)
    else:
        pass2_raw_for_cache = None

    # PR #63: Pass 2 fact-discipline programmatic enforcer. Validates
    # Pass 2's critique text against ground-truth spot, flags dollar
    # mentions that contradict reality. Phase 1: detect-and-flag only
    # (operator sees violations in report, judges severity); future
    # PR #64+ will decide enforcement actions based on real patterns.
    pass2_fact_violations = []
    if pass2 and pass2.key_risks:
        from src.pass2_fact_check import validate_pass2_critique
        # parse_ai_pass2 stores primary_critique as first element of key_risks.
        critique_text = pass2.key_risks[0] if pass2.key_risks else ""
        pass2_fact_violations = validate_pass2_critique(critique_text, spot)
        if pass2_fact_violations:
            n_high = sum(1 for v in pass2_fact_violations
                          if v.kind == "spot_contradiction")
            n_low = len(pass2_fact_violations) - n_high
            print(f"⚠ Pass 2 fact-discipline: {n_high} high-conf contradiction(s), "
                  f"{n_low} outlier price(s) — see report for details")

    # The "Pass 2 wins" projection: every downstream consumer reads from
    # `effective_ai`. Pass 2 if it ran and parsed; otherwise Pass 1. None
    # in --no-ai or full AI failure.
    effective_ai = pass2 if pass2 else pass1

    # --- 5a. Catalyst verification (W6 PR #33, closes D-W5-1).
    # Haiku-constrained primary-source check on top-3 catalysts.
    # UNVERIFIED → magnitude downgrade to "low"; REFUTED → drop. Same-day
    # cache replays the verification result with $0 cost.
    catalyst_verifications = []
    verification_cost = 0.0
    if cache_hit and cache_payload is not None:
        catalyst_verifications = cache_payload.get("catalyst_verifications") or []
    elif (tier.catalyst_verification_model is not None
          and effective_ai and effective_ai.catalysts):
        print(f"AI catalyst verification (model={tier.catalyst_verification_model})...")
        catalyst_verifications, verification_cost = call_ai_catalyst_verification(
            ticker, effective_ai.catalysts, horizon_days,
            verification_model=tier.catalyst_verification_model,
        )
    if catalyst_verifications and effective_ai:
        before = len(effective_ai.catalysts)
        effective_ai.catalysts = apply_catalyst_verification(
            effective_ai.catalysts, catalyst_verifications,
        )
        n_refuted = sum(1 for v in catalyst_verifications if v.get("verdict") == "REFUTED")
        n_unverified = sum(1 for v in catalyst_verifications if v.get("verdict") == "UNVERIFIED")
        n_verified = sum(1 for v in catalyst_verifications if v.get("verdict") == "VERIFIED")
        print(
            f"   Verification: {n_verified} VERIFIED, {n_unverified} UNVERIFIED "
            f"(magnitude → low), {n_refuted} REFUTED (dropped). "
            f"Catalysts: {before} → {len(effective_ai.catalysts)}"
        )

    # --- 5b. AI-derived signals (catalyst_proximity, narrative, factor bias)
    #          built from Pass 2's revised values when available.
    catalyst_mu, catalyst_conf, catalyst_rat = (0.0, "LOW", "no AI catalysts")
    narrative_mu, narrative_conf, narrative_rat = (0.0, "LOW", "no AI narrative")
    factor_bias, factor_rat = (0.0, "no factor analysis")
    if effective_ai:
        catalyst_mu, catalyst_conf, catalyst_rat = signal_from_catalyst_proximity(
            effective_ai.catalysts, horizon_days,
        )
        evidence_count = sum(
            1 for c in effective_ai.catalysts
            if isinstance(c, dict) and c.get("sources") and len(c.get("sources", [])) >= 2
        )
        narrative_mu, narrative_conf, narrative_rat = signal_from_structural_narrative(
            effective_ai.narrative_score, evidence_count,
        )
        # Bull/bear factor arithmetic stays sourced from Pass 1 only — Pass 2
        # is a critique not a re-extraction. If Pass 2 disagreed it would be
        # via revised_drift / revised_confidence, both already captured.
        if pass1:
            factor_bias, factor_rat = apply_bull_bear_arithmetic(
                pass1.bull_factors, pass1.bear_factors,
            )

    if effective_ai:
        signals_dict["catalyst_proximity"] = {
            "drift": catalyst_mu, "confidence": catalyst_conf,
            "source_quality": "PRIMARY",
            "sources_count": len(effective_ai.catalysts),
            "notes": catalyst_rat,
        }
        signals_dict["narrative"] = {
            "drift": narrative_mu, "confidence": narrative_conf,
            "source_quality": "PRIMARY",
            "sources_count": 0,
            "notes": narrative_rat,
        }
    else:
        # No AI ran (--no-ai or Pass 1 failed). catalyst_proximity and
        # narrative are AI-DERIVED — without AI they are genuinely MISSING,
        # not "active at zero." Mark as NONE_FOUND so blend_with_uncertainty's
        # phantom-signal std accounting captures all three AI-derived weights
        # (ai 25% + catalyst_proximity 10% + narrative 10% = 45% total
        # phantom contribution to within_var).
        signals_dict["catalyst_proximity"] = _none_signal("AI absent — no catalyst extraction")
        signals_dict["narrative"] = _none_signal("AI absent — no narrative synthesis")

    if effective_ai:
        if pass2:
            signals_dict["ai"] = {
                "drift": pass2.drift_estimate,
                "confidence": pass2.confidence,
                "source_quality": "REPUTABLE",
                "sources_count": pass1.raw_sources_cited if pass1 else 0,
                "notes": f"Pass 2 revised ({pass2.revision_from_prior_pass:+.1%} vs Pass 1)",
            }
        else:
            signals_dict["ai"] = {
                "drift": pass1.drift_estimate,
                "confidence": pass1.confidence,
                "source_quality": "REPUTABLE",
                "sources_count": pass1.raw_sources_cited,
                "notes": f"Pass 1 estimate ({pass1.raw_sources_cited} sources)",
            }
    else:
        signals_dict["ai"] = _none_signal("AI Pass 1 failed")

    # --- 6. Blend ---
    print(f"Blending {len(signals_dict)} signals + bull/bear arithmetic...")
    blend = blend_with_uncertainty(signals_dict, weights_dict=BLEND_WEIGHTS_V2)
    if blend and blend.get("blended") is not None:
        today_mu = float(blend["blended"]) + factor_bias
        today_std = float(blend.get("std", 0.20))
    else:
        today_mu = mu_effective_historical + factor_bias
        today_std = BAYESIAN_DEFAULT_TODAY_STD

    # --- 7. Bayesian smoothing ---
    prior_v2 = load_prior_posterior(history_path)
    if prior_v2:
        prior_age_days = max(1, (datetime.now().date() -
                                  datetime.strptime(prior_v2["date"][:10], "%Y-%m-%d").date()).days)
        prior_std_safe = max(BAYESIAN_STD_FLOOR, float(prior_v2.get("std") or BAYESIAN_DEFAULT_PRIOR_STD))
        today_std_safe = max(BAYESIAN_STD_FLOOR, float(today_std))
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
        "prior_std": prior_v2["std"] if prior_v2 else BAYESIAN_DEFAULT_PRIOR_STD,
        "today_mu": today_mu, "today_std": today_std,
        "post_mu": post_mu, "post_std": post_std,
        "prior_weight": prior_weight,
        "phantom_signals": blend.get("phantom_signals", []) if blend else [],
        "phantom_std_inflation": blend.get("phantom_std_inflation", 0.0) if blend else 0.0,
        "today_weight": 1 - prior_weight,
    }

    base_signals = _signals_dict_to_display_list(signals_dict, BLEND_WEIGHTS_V2, blend=blend)

    # --- 8. Vol schedule (catalyst-aware) ---
    # peer_earnings_dts populated at fetch time (step 3.5). macro_event_dates
    # still empty — populated in W4/W5 alongside the FOMC/CPI calendar.
    vol_schedule = build_catalyst_vol_schedule(
        base_vol=blended_sigma,
        horizon_days=horizon_days,
        self_earnings_date=self_earnings_dt,
        peer_earnings_dates=peer_earnings_dts,
        macro_event_dates=[],
    )

    # --- 9. Apply AI vol_regime multiplier (uses Pass 2's revised regime
    #         when available — Pass 2 wins, sacred decision #7).
    # PR #24: vol_regime multiplier is per-σ-class. The same "HIGH"
    # vol_regime call means something different on a MID name (already
    # high σ; amplify more aggressively) vs EXTREME (already extreme;
    # less room to amplify).
    if effective_ai:
        class_vol_mults = SIGMA_CLASSES[sigma_class].ai_vol_regime_multipliers
        vol_mult = class_vol_mults.get(effective_ai.vol_regime, 1.0)
        effective_sigma = blended_sigma * vol_mult
    else:
        effective_sigma = blended_sigma

    # --- 10. Run MC ---
    # W9 PR #48: fat-tail innovations (Student-t) when configured;
    # df is per-σ-class so EXTREME names use heavier tails than MID.
    mc_dist = MC_DISTRIBUTION.default
    mc_df = MC_DISTRIBUTION.per_class.get(sigma_class, MC_DISTRIBUTION.default_df)
    print(f"Running Monte Carlo ({DEFAULT_MC_PATHS} paths, "
          f"distribution={mc_dist}{f', df={mc_df}' if mc_dist == 'student_t' else ''})...")
    paths = run_mc_joint_conditional(
        S0=spot,
        sigma=effective_sigma,
        mu=post_mu,
        horizon_days=horizon_days,
        n_paths=DEFAULT_MC_PATHS,
        vol_schedule=vol_schedule,
        mean_reversion_strength=args.mean_reversion,
        mean_reversion_anchor=spot * (1.0 - MEAN_REVERSION_ANCHOR_PCT_BELOW_SPOT) if args.mean_reversion > 0 else None,
        distribution=mc_dist,
        df=mc_df,
    )

    # --- 11. Scan grid ---
    print("Scanning dip × rally grid (Brownian bridge correction)...")
    best, all_candidates, met_threshold_strict = scan_dip_rally_grid(
        S0=spot, sigma=effective_sigma, mu=post_mu, horizon_days=horizon_days,
        paths=paths,
        conviction_dip=conviction_dip,
        conviction_rally_cond=conviction_rally_cond,
        sigma_class=sigma_class,
        vol_schedule=vol_schedule,
    )

    # --- 11b. Sacred decision #13 — EV-hurdle hard gate.
    # Refuse to recommend if EV < +EV_HURDLE_BPS_OF_DIP of dip after friction.
    # Post-sacred-#6: ev_pct_of_dip is computed per-pair in scan_dip_rally_grid
    # (no capital concept anywhere). Just read best.ev_pct_of_dip and gate.
    # KEEP best (so sensitivity + path metrics still render — trader needs the
    # context to understand WHY refusal fired). The refusal flag overrides the
    # headline section in reporter.format_report.
    # PR #70 (2026-05-24): per-σ-class EV-hurdle. Engine now reads the
    # class-specific threshold from sigma_classes[<CLASS>].ev_hurdle_bps
    # instead of the legacy global. MID stays at 50 bps (0.50%);
    # HIGH/EXTREME drop to 25 bps (0.25%) to acknowledge active-trader
    # execution alpha (manual stop / exit timing). Falls back to global
    # when class entry omits the field (backward compatibility).
    ev_hurdle_refused = False
    ev_pct_of_dip = None
    class_ev_hurdle_bps = getattr(
        SIGMA_CLASSES[sigma_class], "ev_hurdle_bps", None
    ) or EV_HURDLE_BPS_OF_DIP
    if best is not None:
        ev_pct_of_dip = best.ev_pct_of_dip
        ev_hurdle_threshold = class_ev_hurdle_bps / 10000.0
        if ev_pct_of_dip < ev_hurdle_threshold:
            ev_hurdle_refused = True
            ev_bps = ev_pct_of_dip * 10000.0
            print(f"⛔ Sacred #13 EV-hurdle refusal: EV/dip = "
                  f"{ev_bps:+.1f} bps ({ev_pct_of_dip*100:+.2f}%) "
                  f"< required {class_ev_hurdle_bps} bps "
                  f"({class_ev_hurdle_bps/100:.2f}%) for σ-class {sigma_class}")
            met_threshold_strict = False  # cascade — no clean-recommendation headline

    # --- 11c. Sacred decision #14 — trend filter (D-W2-15).
    # Refuse to recommend a dip buy when mom_30d is below threshold AND no
    # in-horizon catalyst (bullish or two-sided) was surfaced by AI. The
    # idea: don't catch falling knives without a thesis. A stock down >25%
    # in 30 days entered at the dip without verifiable catalyst support is
    # empirically negative-EV — institutional discipline says pass.
    #
    # In --no-ai mode no catalysts are known, so strict reading: refuse.
    # The operator can lift the threshold via YAML or pass --no-ai with
    # the understanding that the trend filter will fire on every dip-momentum
    # name without AI-sourced catalyst verification.
    trend_filter_refused = False
    if best is not None and snapshot.mom_30d < TREND_FILTER_MOM_30D_THRESHOLD:
        if not _has_supporting_catalyst(effective_ai, horizon_days):
            trend_filter_refused = True
            print(f"⛔ Sacred #14 trend-filter refusal: mom_30d = "
                  f"{snapshot.mom_30d*100:+.1f}% < {TREND_FILTER_MOM_30D_THRESHOLD*100:+.0f}%, "
                  f"no in-horizon bullish/two-sided catalyst")
            met_threshold_strict = False

    # PR #41 / PR #44: mirror of sacred #14 for blow-off tops. 30-day
    # momentum is the symmetric trigger — sacred #14 refuses on mom_30d
    # < -25% (falling knife); we refuse on mom_30d > +50% (parabola)
    # absent a bearish/two-sided de-rating catalyst.
    # PR #44 redesigned away from RSI+YTD: the INTC smoke (RSI=66.2,
    # YTD=+204%, mom_30d=+92%) bypassed the original gate because RSI
    # lags the explosive-move phase by ~10 days.
    # PR #70 (2026-05-24): per-σ-class parabola threshold. MID stays at
    # +50% (large-cap momentum at +50%/30d IS exceptional); HIGH lifted
    # to +80% (AI/semi names routinely 50-80%/30d in trending regimes);
    # EXTREME lifted to +100% (only true doubling-in-month is exhaustion).
    # Falls back to global when class entry omits the field.
    class_parabola_threshold = getattr(
        SIGMA_CLASSES[sigma_class], "parabola_mom_30d_threshold", None
    ) or PARABOLA_FILTER_MOM_30D_THRESHOLD
    parabola_filter_refused = False
    if (best is not None
            and snapshot.mom_30d >= class_parabola_threshold):
        if not _has_bearish_derating_catalyst(effective_ai, horizon_days):
            parabola_filter_refused = True
            print(f"⛔ Parabola-filter refusal (PR #41/#44, σ-class-aware PR #70): "
                  f"mom_30d = {snapshot.mom_30d*100:+.1f}% ≥ "
                  f"{class_parabola_threshold*100:+.0f}% threshold for "
                  f"σ-class {sigma_class}, no in-horizon bearish de-rating catalyst")
            met_threshold_strict = False

    # --- 13. AI catalyst stress test — uses Pass 2's revised catalyst list
    #          when available (effective_ai), else Pass 1's. Sacred #7.
    catalyst_stress_results = []
    catalyst_stress_cost = 0.0
    if cache_hit and cached_stress_results is not None:
        # Replay from cache. Cost = $0.00.
        catalyst_stress_results = cached_stress_results
        catalyst_stress_cost = 0.0
    elif tier.stress_model is not None and effective_ai and best:
        print(f"AI catalyst impact stress test (model={tier.stress_model})...")
        catalyst_stress_results, catalyst_stress_cost = call_ai_catalyst_stress_test(
            ticker, spot, best.dip_price, best.rally_price,
            effective_ai.catalysts, horizon_days,
            stress_model=tier.stress_model,
        )

    # Write the AI cache for this ticker-date AFTER all AI work is complete.
    # On cache miss we save everything we computed; on cache hit we skip the
    # write (cache file already exists and is correct).
    if not cache_hit and not args.no_ai and pass1_raw_for_cache:
        try:
            ai_cache.save(ticker, spot, {
                # PR #77: persist the tier so a later same-day rerun at
                # a higher tier can correctly invalidate (vs blindly
                # replaying a Pass-1-only cache as if it were a Pass-2+
                # T3 result).
                "tier_name": tier.name,
                "pass1_raw": pass1_raw_for_cache,
                "pass1_cost": pass1_cost_charged,
                "pass1_sources": pass1_sources_for_cache,
                "pass2_raw": pass2_raw_for_cache,
                "pass2_cost": pass2_cost_charged,
                "catalyst_verifications": catalyst_verifications,
                "verification_cost": verification_cost,
                "stress_results": catalyst_stress_results,
                "stress_cost": catalyst_stress_cost,
                "models_used": {
                    "pass1": tier.pass1_model,
                    "pass2": tier.pass2_model,
                    "stress": tier.stress_model,
                    "catalyst_verification": tier.catalyst_verification_model,
                },
            })
        except Exception as e:
            print(f"   WARNING: ai_cache.save failed (run still succeeds): {e}")

    # --- 14. Three-method math cross-check + hard refusal gate (sacred #16) ---
    # Sacred #8: math cross-check runs on EVERY run, not just qualified pairs.
    # PR #25 (D-W3-3 closed): when no pair qualifies, the check runs against
    # a deterministic class-anchor pair (mid-depth dip × mid-reach rally) so
    # the math layer's health is observable on WAIT-verdict tickers too.
    # Verification-anchor runs never trigger the sacred #16 refusal — that
    # gate only applies to recommendations, and we're not recommending.
    print("Three-method math cross-check...")
    method_check_is_anchor = best is None
    if method_check_is_anchor:
        class_grid = SIGMA_CLASSES[sigma_class].grid
        anchor_dip = float(spot * (1.0 - class_grid.dip_max_depth_pct / 2.0))
        anchor_rally = float(spot * (1.0 + class_grid.rally_max_reach_pct / 2.0))
        check_dip, check_rally = anchor_dip, anchor_rally
    else:
        check_dip, check_rally = best.dip_price, best.rally_price

    bridge_check_result = analyze_joint_conditional(
        paths, spot, check_dip, check_rally, horizon_days,
        sigma=effective_sigma, vol_schedule=vol_schedule,
    )
    method_check = three_method_cross_check(
        spot, effective_sigma, post_mu, horizon_days,
        check_dip, check_rally, bridge_check_result,
        # PR #82: pass the vol_schedule so PDE/closed-form use the
        # RMS-equivalent constant sigma matching the MC. Without this,
        # any in-horizon earnings/macro spike causes structural MC vs
        # PDE divergence (unmasked by PR #76's correct schedule
        # indexing), tripping sacred #16 on stable MID/HIGH names that
        # the math layer is otherwise confident about.
        vol_schedule=vol_schedule,
    )
    method_check["is_anchor"] = method_check_is_anchor
    method_check["anchor_dip"] = check_dip if method_check_is_anchor else None
    method_check["anchor_rally"] = check_rally if method_check_is_anchor else None

    # Sacred decision #16 — hard refusal when MC and PDE/closed-form
    # diverge beyond the σ-scaled refusal threshold. Only applied to
    # actual recommendations; anchor-pair verification never blocks.
    if not method_check_is_anchor and method_check.get("refused"):
        print(f"⛔ Method-disagreement refusal triggered: {'; '.join(method_check['refusals'])}")
        best = None  # blocks recommendation; report prints refusal headline
        met_threshold_strict = False

    # --- 14a. Ambiguity score (W4 PR #28) ---
    # Computed AFTER all T0 outputs (math, σ triangulation, method check)
    # but BEFORE any AI dispatch in batch mode. Single scalar in [0, 1];
    # the broker (PR #29) ranks tickers by this to allocate T3→T2→T1
    # within the $2/day cap.
    method_table = method_check.get("table") or []
    method_max_delta_pp = max((abs(row[3]) for row in method_table), default=0.0)
    tols = method_check.get("tolerances") or {}
    method_refuse_pp = float(tols.get("refuse_first_passage_pp", 999.0))
    ambiguity = compute_ambiguity(
        best_p_dip=(best.p_dip_touched if best is not None else None),
        conviction_dip=conviction_dip,
        best_ev_pct_of_dip=(best.ev_pct_of_dip if best is not None else None),
        sigma_divergence_pp=divergence_pp,
        method_max_delta_pp=method_max_delta_pp,
        method_refuse_threshold_pp=method_refuse_pp,
        mom_30d=mom_30d,
    )
    print(f"Ambiguity score: {ambiguity.overall:.2f}  (broker sort key)")

    # --- 14b. Sacred T2+ gate (W4 PR #30) — pre-AI net EV positive AND
    # conviction met. Broker (PR #29) reads this from the snapshot to
    # decide whether a ticker is eligible for T2/T3 tiers.
    qualifies_for_t2_plus = bool(
        best is not None
        and met_threshold_strict
        and best.net_ev_per_share > 0
    )
    print(
        f"Pre-AI T2+ gate: {'PASS' if qualifies_for_t2_plus else 'FAIL'} "
        f"(conviction_met={met_threshold_strict}, "
        f"ev_positive={best is not None and best.net_ev_per_share > 0})"
    )

    # --- 14b. Sensitivity table ---
    sensitivity = None
    if best is not None:
        print("Computing sensitivity table (drift/σ scenarios + catalyst shocks)...")
        sensitivity = compute_sensitivity_table(
            S0=spot, base_sigma=effective_sigma, base_mu=post_mu,
            horizon_days=horizon_days,
            dip_price=best.dip_price, rally_price=best.rally_price,
            sigma_class=sigma_class,
            catalyst_shocks=catalyst_stress_results,
            vol_schedule_base=vol_schedule,
            n_paths_sensitivity=10_000,
        )

    # --- 14c. Path metrics ---
    path_metrics = None
    if best is not None:
        path_metrics = compute_path_metrics(
            paths, spot, best.dip_price, best.rally_price,
            panic_floor_pct=SIGMA_CLASSES[sigma_class].panic_floor_pct,
        )

    # --- 15. Backtest layer ---
    backtest = run_backtest_layer(history_path, spot)

    # --- 15b. W10 calibration harness (PR #47): resolve outcomes for
    # any prior predictions whose horizon window has now closed. Updates
    # CSV in-place. Idempotent — already-resolved rows pass through
    # unchanged. Runs BEFORE the new prediction row is written so the
    # current row enters as OPEN.
    try:
        from src.calibration import resolve_history as _resolve_history
        if history_path.exists():
            with open(history_path, "r") as _f:
                _existing_rows = list(csv.DictReader(_f))
            _resolved_rows, _n_newly = _resolve_history(
                _existing_rows, history_df,
            )
            if _n_newly > 0:
                print(f"Calibration: resolved {_n_newly} prior prediction"
                      f"{'s' if _n_newly != 1 else ''} (horizon window closed)")
                # Rewrite CSV with resolved outcomes.
                with open(history_path, "w", newline="") as _f:
                    _w = csv.DictWriter(_f, fieldnames=CSV_COLUMNS,
                                          extrasaction="ignore")
                    _w.writeheader()
                    for _r in _resolved_rows:
                        _w.writerow(_r)
    except Exception as _e:
        print(f"   WARNING: calibration resolver skipped: {_e}")

    # --- 16. Persist CSV row ---
    total_ai_cost = (pass1_cost_charged + pass2_cost_charged
                     + verification_cost + catalyst_stress_cost)
    csv_row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "spot": f"{spot:.2f}",
        "sigma_blended": f"{blended_sigma:.4f}",
        "sigma_class": sigma_class,
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
        "net_ev_per_share": f"{best.net_ev_per_share:.4f}" if best else "",
        "ev_pct_of_dip": f"{best.ev_pct_of_dip:.6f}" if best else "",
        "ai_drift_pass1": f"{pass1.drift_estimate:.4f}" if pass1 else "",
        "ai_drift_pass2": f"{pass2.drift_estimate:.4f}" if pass2 else "",
        "ai_vol_regime": effective_ai.vol_regime if effective_ai else "",
        "narrative_score": effective_ai.narrative_score if effective_ai else "",
        "catalyst_proximity_drift": f"{catalyst_mu:.4f}",
        "garch_alpha_plus_beta": f"{vol_profile.garch_alpha_plus_beta:.4f}",
        "horizon_days": str(horizon_days),
        "method_agreement_flags": ";".join(method_check["flags"]),
        "ai_cost_total": f"{total_ai_cost:.2f}",
        "data_source": data_source,
        "ai_tier": tier.name,
        "ambiguity_score": f"{ambiguity.overall:.4f}",
        # W10 PR #54 — D-W10-1 catalyst-accuracy capture.
        "pass1_catalysts_json": _compact_catalysts_json(
            pass1.catalysts if pass1 else []
        ),
        "pass2_catalysts_json": _compact_catalysts_json(
            pass2.catalysts if pass2 else []
        ),
        "verification_verdicts_json": _compact_verdicts_json(
            catalyst_verifications
        ),
        "catalyst_stress_json": _compact_stress_json(catalyst_stress_results),
        "verdict_state": _compute_verdict_state(
            best=best,
            met_threshold_strict=met_threshold_strict,
            method_check=method_check,
            trend_filter_refused=trend_filter_refused,
            parabola_filter_refused=parabola_filter_refused,
            ev_hurdle_refused=ev_hurdle_refused,
        ),
        # PR #81 (audit #13): joint-refusal record. See _all_refusal_reasons.
        "refusal_reasons_all": ",".join(_all_refusal_reasons(
            best=best,
            method_check=method_check,
            trend_filter_refused=trend_filter_refused,
            parabola_filter_refused=parabola_filter_refused,
            ev_hurdle_refused=ev_hurdle_refused,
        )),
        # 2026-05-24 audit fix round 2 — persist Pass 2 reasoning + agreement.
        # Stored on the dataclass by parse_ai_pass2 (round-1 audit fix);
        # this is the CSV write so future calibration analysis (D-W10-1)
        # can correlate "strong_disagree" frequency with realized outcomes.
        # Truncate reasoning to keep CSV row width manageable.
        "pass2_agreement": (getattr(pass2, "agreement_with_pass1", "") or "") if pass2 else "",
        "pass2_revision_reasoning": (
            (getattr(pass2, "revision_reasoning", "") or "")[:300]
            if pass2 else ""
        ),
    }
    append_history_row(history_path, csv_row)

    # --- 17. Report ---
    from src.reporter import format_report, generate_html_dashboard
    runtime = time.time() - t_start
    report = format_report(
        snapshot, vol_profile, base_signals, pass1, pass2, posterior_summary,
        best, method_check, catalyst_stress_results, backtest,
        conviction_dip, conviction_rally_cond, horizon_days,
        total_ai_cost, runtime,
        met_threshold_strict=met_threshold_strict,
        unusual_move=unusual_move,
        sensitivity=sensitivity,
        path_metrics=path_metrics,
        ev_hurdle_refused=ev_hurdle_refused,
        ev_pct_of_dip=ev_pct_of_dip,
        trend_filter_refused=trend_filter_refused,
        parabola_filter_refused=parabola_filter_refused,
        sigma_class=sigma_class,
        sigma_class_mismatch=sigma_class_mismatch,
        ambiguity=ambiguity,
        tier=tier,
        pass2_fact_violations=pass2_fact_violations,
    )
    print(report)

    # --- 18. HTML dashboard ---
    with open(history_path, "r") as f:
        history_rows_for_chart = list(csv.DictReader(f))
    generate_html_dashboard(
        dashboard_path, snapshot, best, vol_profile, base_signals,
        pass1, pass2, method_check, backtest, history_rows_for_chart,
        conviction_dip, conviction_rally_cond, horizon_days,
        sigma_class_mismatch=sigma_class_mismatch,
    )

    # --- 19. Optional W4 broker snapshot (W5 orchestrator collects these) ---
    if getattr(args, "emit_snapshot", False):
        import json
        snap_payload = {
            "ticker": ticker,
            "ambiguity": ambiguity.overall,
            "qualifies_for_t2_plus": qualifies_for_t2_plus,
            "sigma_class": sigma_class,
            "limited_history": limited_history,    # PR #73
        }
        print("BROKER_SNAPSHOT_JSON=" + json.dumps(snap_payload))

    return 0

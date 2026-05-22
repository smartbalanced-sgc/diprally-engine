"""Configuration constants — pricing, blend weights, vol schedule, threshold table.

W0: inlined from seed v1 + v2. Per-ticker registry (W2), σ-class table (W3),
and AI budget broker tiers (W4) will move here in later waves.
"""
from __future__ import annotations


# -----------------------------------------------------------
# Data sources
# -----------------------------------------------------------
FMP_BASE = "https://financialmodelingprep.com/stable"
DEFAULT_LOOKBACK_DAYS = 730


# -----------------------------------------------------------
# Anthropic pricing — Opus 4.7, Sonnet 4.6, Haiku 4.5
# Verify against https://docs.anthropic.com/en/api/pricing before each wave ships
# -----------------------------------------------------------
OPUS_INPUT_PER_TOKEN = 15.00 / 1_000_000
OPUS_OUTPUT_PER_TOKEN = 75.00 / 1_000_000
SONNET_INPUT_PER_TOKEN = 3.00 / 1_000_000
SONNET_OUTPUT_PER_TOKEN = 15.00 / 1_000_000
HAIKU_INPUT_PER_TOKEN = 1.00 / 1_000_000
HAIKU_OUTPUT_PER_TOKEN = 5.00 / 1_000_000
WEB_SEARCH_PER_USE = 0.01

# Canonical model IDs (centralised so model swaps are one-line)
MODEL_OPUS = "claude-opus-4-7"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

# (model_id_prefix, input_rate, output_rate)
_AI_PRICING = (
    ("opus",   OPUS_INPUT_PER_TOKEN,   OPUS_OUTPUT_PER_TOKEN),
    ("sonnet", SONNET_INPUT_PER_TOKEN, SONNET_OUTPUT_PER_TOKEN),
    ("haiku",  HAIKU_INPUT_PER_TOKEN,  HAIKU_OUTPUT_PER_TOKEN),
)


def pricing_for_model(model_id: str) -> tuple[float, float]:
    """Return (input_per_token, output_per_token) for the given model ID.
    Matches by case-insensitive substring on the canonical family name.
    Defaults to Opus pricing for unknown IDs (conservative — overstates cost
    rather than understates).
    """
    lower = (model_id or "").lower()
    for prefix, in_rate, out_rate in _AI_PRICING:
        if prefix in lower:
            return in_rate, out_rate
    return OPUS_INPUT_PER_TOKEN, OPUS_OUTPUT_PER_TOKEN


# -----------------------------------------------------------
# v1 blend weights — kept for any v1-style callers; v2 uses BLEND_WEIGHTS_V2
# -----------------------------------------------------------
BLEND_WEIGHTS = {
    "historical":         0.10,
    "analyst":            0.15,
    "sector":             0.08,
    "macro":              0.07,
    "insider":            0.05,
    "ai":                 0.30,
    "short_interest":     0.05,
    "peer_rs":            0.10,
    "sector_decoupling":  0.10,
}


# Standard error mapping per confidence tier (decimal annualised return).
CONFIDENCE_TO_SE = {"HIGH": 0.05, "MEDIUM": 0.10, "LOW": 0.20}


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

# Path-metrics panic floor (W3 will class-vary it; W0 keeps the v2 30% literal
# but lifted out of compute_path_metrics so it stops looking ticker-specific).
PANIC_FLOOR_PCT = 0.30

# AI vol_regime → vol_mult mapping
AI_VOL_REGIME_MULTIPLIERS = {
    "HIGH": 1.15,
    "MEDIUM": 1.00,
    "LOW": 0.90,
}

# Structural narrative score → drift adjustment (annualised pp)
NARRATIVE_DRIFT_ADJUSTMENT = {
    "strong": 0.05,
    "neutral": 0.00,
    "weak": -0.05,
}

# Bull/bear factor arithmetic weights
FACTOR_WEIGHTS = {"high": 3, "med": 2, "low": 1}
FACTOR_NET_THRESHOLD = 4
FACTOR_TAIL_BIAS = 0.05

# Catalyst Z-score threshold (pattern from src/sentiment.py:112)
CATALYST_Z_THRESHOLD = 3.0

# Vol schedule multipliers around catalysts
VOL_SCHEDULE_MULTIPLIERS = {
    "self_earnings_day": 3.0,
    "self_earnings_pre_post": 1.5,
    "self_earnings_window_days": 2,
    "peer_earnings_day": 1.8,
    "peer_earnings_pre_post": 1.3,
    "peer_earnings_window_days": 1,
    "macro_event_day": 1.5,
}

# Three-method agreement tolerance — σ-SCALED.
# The Brownian bridge correction on the MC has an irreducible residual that
# scales with σ: at σ=30% it's ~0.3-0.8pp, at σ=100% it's ~2-3pp, at σ=150%
# it can reach 3-5pp. A constant threshold is wrong for any ticker outside
# the σ-class it was tuned against. The σ-scaled functions below give:
#   σ=0.30 (MID INTC):    flag 2.0pp, refuse 3.6pp
#   σ=1.00 (EXTREME SNDK): flag 3.0pp, refuse 5.4pp
#   σ=1.50 (high LWLG-like): flag 4.5pp, refuse 8.1pp
# W10 will calibrate these multipliers empirically from realized residuals.
METHOD_AGREEMENT_FLOOR_PP = 2.0      # tolerance never drops below this at low σ
METHOD_AGREEMENT_MULTIPLIER = 3.0    # tolerance ~= 3.0 × σ_effective at high σ
METHOD_AGREEMENT_FIRST_PASSAGE_FLOOR_PP = 3.0
METHOD_AGREEMENT_FIRST_PASSAGE_MULTIPLIER = 4.0
# Refusal threshold = REFUSAL_MULT × flag-threshold. Sacred decision #16
# enforced as a hard gate: MC vs PDE/closed-form diverge beyond this →
# refuse to recommend.
METHOD_REFUSAL_MULTIPLIER = 1.8


def method_tolerance_pp(sigma_effective: float, kind: str = "marginal") -> float:
    """σ-scaled flag tolerance (pp). kind in {'marginal', 'first_passage'}.
    Marginal = P(touch ever). First-passage = P(dip first | rally first).
    """
    if kind == "first_passage":
        return max(METHOD_AGREEMENT_FIRST_PASSAGE_FLOOR_PP,
                   METHOD_AGREEMENT_FIRST_PASSAGE_MULTIPLIER * sigma_effective)
    return max(METHOD_AGREEMENT_FLOOR_PP,
               METHOD_AGREEMENT_MULTIPLIER * sigma_effective)


def method_refusal_pp(sigma_effective: float, kind: str = "marginal") -> float:
    """Hard refusal threshold (pp). Triggers the sacred-decision-#16 gate."""
    return METHOD_REFUSAL_MULTIPLIER * method_tolerance_pp(sigma_effective, kind)


# Legacy constants kept temporarily for backwards-compatible imports.
# Anyone still reading these gets the value at σ=1.0 (the most common case
# in the seed's SNDK-tuned regime). Deprecate-and-remove in W3.
METHOD_AGREEMENT_TOLERANCE_PP_MARGINAL = METHOD_AGREEMENT_MULTIPLIER * 1.0
METHOD_AGREEMENT_TOLERANCE_PP_FIRST_PASSAGE = METHOD_AGREEMENT_FIRST_PASSAGE_MULTIPLIER * 1.0
METHOD_AGREEMENT_TOLERANCE_PP = METHOD_AGREEMENT_TOLERANCE_PP_FIRST_PASSAGE

BAG_HOLD_TERMINAL_ASSUMPTION = "median_terminal_dip_paths"

# Backtest gate
BACKTEST_MIN_SAMPLES = 30

# Sacred decision #13 — EV-hurdle gate.
# "Refuse to recommend if EV < +50 bps of dip after friction."
# Net expected $/trade divided by capital = EV % of dip (see derivation in
# engine.py). When that ratio falls below this threshold (expressed in bps),
# the engine refuses the recommendation regardless of conviction thresholds
# being met. Marginal positive-EV trades at extreme valuation contexts must
# not be authorized as clean recommendations.
EV_HURDLE_BPS_OF_DIP = 50

# Analyst signal extreme-outlier threshold (D-W2-13).
# When |implied_drift| > this value, the analyst signal's confidence gets
# downgraded one notch (HIGH → MEDIUM → LOW). Catches data-quality issues
# (wrong-ticker, stale, sparse-coverage) before they drive the blend.
# Surfaced by MOG-A smoke (2026-05-22): FMP returned -58.9% implied drift
# HIGH conf on a $10B defense industrial up only +27% YTD — either genuine
# deep-sell consensus or bad data. Either way, |drift| > 0.50/yr deserves
# scrutiny, not full HIGH-conf weight in the blend.
ANALYST_EXTREME_DRIFT_THRESHOLD = 0.50

# v2 blend weights — 10 signals
BLEND_WEIGHTS_V2 = {
    "historical":          0.05,
    "analyst":             0.15,
    "sector":              0.04,
    "macro":               0.07,
    "insider":             0.02,
    "short_interest":      0.02,
    "peer_rs":             0.10,
    "sector_decoupling":   0.10,
    "ai":                  0.25,
    "catalyst_proximity":  0.10,
    "narrative":           0.10,
}


# v3 review criteria — LOCKED at v2 ship, executed at 30 days of runtime data
V3_REVIEW_CRITERIA = {
    "n_days_min": 30,
    "calibration_dip_target": (0.60, 0.70),
    "calibration_rally_cond_target": (0.70, 0.80),
    "ai_pass2_critique_rate_min": 0.20,
    "catalyst_signal_correlation_min": 0.10,
    "bag_hold_rate_target": (0.10, 0.20),
}

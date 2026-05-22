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

# Three-method agreement tolerance.
METHOD_AGREEMENT_TOLERANCE_PP_MARGINAL = 3.0       # P(touch ever)
METHOD_AGREEMENT_TOLERANCE_PP_FIRST_PASSAGE = 4.0  # P(dip first), P(rally first)
METHOD_AGREEMENT_TOLERANCE_PP = METHOD_AGREEMENT_TOLERANCE_PP_FIRST_PASSAGE

BAG_HOLD_TERMINAL_ASSUMPTION = "median_terminal_dip_paths"

# Backtest gate
BACKTEST_MIN_SAMPLES = 30

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

"""Config loader. Sacred decision #17 — all configurable values live in
`config/diprally.yaml`. This module:

  1. Reads the YAML at import time
  2. Validates the schema via pydantic (typed, catches errors loudly)
  3. Exposes backwards-compatible module-level constants so every existing
     `from src.config import X` import keeps working without changes

Anywhere a constant is needed across `src/`, import from this module exactly
as before. To change a value, edit `config/diprally.yaml`. No code edit, PR,
or deploy required for threshold tuning — exactly what sacred #17 demands.

W2 foundation scope: top-level constants previously hardcoded here. Future
W2 sessions lift embedded thresholds from signals.py / data_fetch.py /
math_utils.py and add the ticker registry.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Pydantic schemas — mirror the YAML structure exactly
# =============================================================================

class _StrictModel(BaseModel):
    """Forbid extra keys. Typos in YAML keys must fail loudly, not be silently
    ignored. (Pydantic's default is permissive on extras; we override.)"""
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_StrictModel):
    fmp_base_url: str
    default_lookback_days: int


class AIPricingConfig(_StrictModel):
    opus_input_per_token: float
    opus_output_per_token: float
    sonnet_input_per_token: float
    sonnet_output_per_token: float
    haiku_input_per_token: float
    haiku_output_per_token: float
    web_search_per_use: float


class AIModelsConfig(_StrictModel):
    opus: str
    sonnet: str
    haiku: str


class ConvictionConfig(_StrictModel):
    dip_marginal: float = Field(gt=0.0, lt=1.0)
    rally_conditional: float = Field(gt=0.0, lt=1.0)
    ev_hurdle_bps_of_dip: float = Field(ge=0.0)


class SigmaClassBoundariesConfig(_StrictModel):
    """W3: σ-class auto-detection boundaries. extreme_min > high_min must hold;
    pydantic model_validator below enforces. Both in decimal (0.95 = 95%)."""
    extreme_min: float = Field(gt=0.0, lt=10.0)
    high_min: float = Field(gt=0.0, lt=10.0)


class SigmaClassConvictionConfig(_StrictModel):
    """Per-class conviction thresholds (W3, PR #21 scope: conviction only)."""
    dip: float = Field(gt=0.0, lt=1.0)
    rally_conditional: float = Field(gt=0.0, lt=1.0)


class SigmaClassGridConfig(_StrictModel):
    """Per-class dip/rally grid sizing (W3, PR #22). All fields are
    fractions of spot — price-agnostic across the universe. dip steps
    scan DOWN from spot; rally steps scan UP. max_depth/max_reach
    bound the grid range."""
    dip_step_pct: float = Field(gt=0.0, lt=1.0)
    rally_step_pct: float = Field(gt=0.0, lt=1.0)
    dip_max_depth_pct: float = Field(gt=0.0, lt=1.0)
    rally_max_reach_pct: float = Field(gt=0.0)


class SigmaClassThresholdConfig(_StrictModel):
    """One row in the sigma_classes table. PR #21 added conviction;
    PR #22 added grid; PR #23 added friction_bps_round_trip;
    panic/ai_vol_mult slot in via PR #24."""
    conviction: SigmaClassConvictionConfig
    grid: SigmaClassGridConfig
    friction_bps_round_trip: float = Field(ge=0.0)


class HorizonConfig(_StrictModel):
    default_days: int = Field(gt=0)
    default_mc_paths: int = Field(gt=0)
    deep_dip_autoscale_threshold: float = Field(gt=0.0, lt=1.0)
    deep_dip_autoscale_paths: int = Field(gt=0)


class GridConfig(_StrictModel):
    """Legacy global grid container — step/depth/reach moved to
    sigma_classes.<CLASS>.grid in W3 PR #22. panic_floor stays here
    until PR #24 moves it per-class."""
    panic_floor_pct: float = Field(gt=0.0, lt=1.0)


class MethodToleranceConfig(_StrictModel):
    marginal_floor_pp: float = Field(ge=0.0)
    marginal_multiplier: float = Field(ge=0.0)
    first_passage_floor_pp: float = Field(ge=0.0)
    first_passage_multiplier: float = Field(ge=0.0)
    refusal_multiplier: float = Field(gt=1.0)


class BacktestConfig(_StrictModel):
    min_samples: int = Field(ge=1)


class FactorWeightsConfig(_StrictModel):
    high: int = Field(ge=1)
    med: int = Field(ge=1)
    low: int = Field(ge=1)


class FactorArithmeticConfig(_StrictModel):
    weights: FactorWeightsConfig
    net_threshold: int = Field(ge=0)
    tail_bias: float = Field(ge=0.0)


class CatalystConfig(_StrictModel):
    z_threshold: float = Field(gt=0.0)


class VolScheduleMultipliersConfig(_StrictModel):
    self_earnings_day: float = Field(gt=0.0)
    self_earnings_pre_post: float = Field(gt=0.0)
    self_earnings_window_days: int = Field(ge=0)
    peer_earnings_day: float = Field(gt=0.0)
    peer_earnings_pre_post: float = Field(gt=0.0)
    peer_earnings_window_days: int = Field(ge=0)
    macro_event_day: float = Field(gt=0.0)


class AICacheConfig(_StrictModel):
    spot_move_invalidation_pct: float = Field(gt=0.0, lt=1.0)


class TrendFilterConfig(_StrictModel):
    """Sacred decision #14 — refuse dip if mom_30d below this threshold AND
    no fundamental catalyst (bullish or two-sided) in horizon."""
    mom_30d_threshold: float = Field(lt=0.0, gt=-1.0)


class MacroRegimeConfig(_StrictModel):
    """D-W2-8: macro regime detection thresholds (VIX + SPY-vs-MA50)."""
    vix_risk_off_threshold: float = Field(gt=0.0)
    vix_risk_on_threshold: float = Field(gt=0.0)
    spy_risk_off_threshold: float = Field(lt=0.0)
    spy_risk_on_threshold: float = Field(gt=0.0)
    vix_default_fallback: float = Field(gt=0.0)


class OptionsIVConfig(_StrictModel):
    """D-W2-8: options IV liquidity gate."""
    liquidity_max_bid_ask_pct: float = Field(gt=0.0, lt=1.0)
    default_target_dte_days: int = Field(gt=0)
    dte_window_min: int = Field(ge=0)
    dte_window_max_multiplier: float = Field(gt=1.0)


class SectorPerfConfig(_StrictModel):
    default_lookback_days: int = Field(gt=0)


class PDEGridConfig(_StrictModel):
    """D-W2-9: PDE Crank-Nicolson grid resolution."""
    n_space: int = Field(gt=10)
    n_time: int = Field(gt=10)


class GARCHConfig(_StrictModel):
    """D-W2-9: GARCH(1,1) fit parameters."""
    min_data_bars: int = Field(gt=10)
    fallback_bars: int = Field(gt=10)
    initial_omega: float = Field(gt=0.0, lt=1.0)
    initial_omega_full: float = Field(gt=0.0, lt=1.0)
    initial_alpha: float = Field(ge=0.0, lt=1.0)
    initial_beta: float = Field(ge=0.0, lt=1.0)


class EngineConfig(_StrictModel):
    """D-W2-5 + D-W2-7: engine-level scattered tunables.
    spread_per_share_round_trip RETIRED in W3 PR #23 — friction is now
    per-σ-class bps."""
    drift_cap: float = Field(gt=0.0)
    garch_fallback_sigma: float = Field(gt=0.0, lt=10.0)
    grid_prefilter_looseness: float = Field(ge=0.0, lt=1.0)


class BayesianConfig(_StrictModel):
    """D-W2-7: Bayesian smoothing parameters."""
    prior_age_inflation_per_day: float = Field(ge=0.0)
    default_prior_std: float = Field(gt=0.0, lt=1.0)
    std_floor: float = Field(gt=0.0, lt=1.0)
    default_today_std_when_blend_fails: float = Field(gt=0.0, lt=1.0)


class MeanReversionConfig(_StrictModel):
    """D-W2-7: mean-reversion anchor position when MR enabled via CLI."""
    anchor_pct_below_spot: float = Field(ge=0.0, lt=1.0)


class Pass2PromptConfig(_StrictModel):
    """D-W2-7: Pass 2 closed-form math context bracket."""
    closed_form_bracket_pct: float = Field(gt=0.0, lt=1.0)


class SensitivityScenarioConfig(_StrictModel):
    """D-W2-7: one row in the sensitivity table."""
    label: str
    drift_offset: float
    sigma_multiplier: float = Field(gt=0.0)


# ---- D-W2-6: signal-level embedded thresholds ----

class AnalystSignalConfig(_StrictModel):
    last_month_min_n_for_use: int = Field(ge=1)
    last_month_high_conf_n: int = Field(ge=1)
    last_quarter_min_n_for_use: int = Field(ge=1)
    last_quarter_medium_conf_n: int = Field(ge=1)
    staleness_move_60d: float = Field(gt=0.0, lt=1.0)
    spread_high_conf: float = Field(gt=0.0, lt=1.0)
    spread_medium_conf: float = Field(gt=0.0, lt=1.0)


class SectorMomentumCapsConfig(_StrictModel):
    """Regime-conditional sector momentum (high_cap, low_cap) tuples."""
    post_parabola: tuple[float, float]
    momentum: tuple[float, float]
    default: tuple[float, float]


class HistoricalSignalConfig(_StrictModel):
    cap_binding_abs_drift: float = Field(gt=0.0)
    medium_gating_abs_drift: float = Field(gt=0.0)


class ShortInterestBracketConfig(_StrictModel):
    threshold_lt: float = Field(gt=0.0, le=1.0)
    drift: float
    confidence: str = Field(pattern=r"^(HIGH|MEDIUM|LOW)$")
    note: str


class PeerRSConfig(_StrictModel):
    drift_cap_abs: float = Field(gt=0.0)
    dispersion_high_conf: float = Field(gt=0.0)
    dispersion_medium_conf: float = Field(gt=0.0)


class SectorDecouplingConfig(_StrictModel):
    drift_cap_abs: float = Field(gt=0.0)
    magnitude_low_conf: float = Field(gt=0.0)
    magnitude_medium_conf: float = Field(gt=0.0)


class RegimeDetectionConfig(_StrictModel):
    sigma_high_threshold: float = Field(gt=0.0)
    mom_5d_threshold: float = Field(gt=0.0)
    mom_30d_pct_threshold: float = Field(gt=0.0)
    rsi_overbought: float = Field(gt=50.0, le=100.0)
    rsi_oversold: float = Field(ge=0.0, lt=50.0)
    ytd_parabola_pct: float = Field(gt=0.0)


class CatalystProximityConfig(_StrictModel):
    magnitude_drift_map: dict[str, float]
    drift_cap_abs: float = Field(gt=0.0)
    in_window_count_high_conf: int = Field(ge=1)
    in_window_count_medium_conf: int = Field(ge=1)


class SignalsConfig(_StrictModel):
    """D-W2-6: aggregate of all signal-level embedded thresholds."""
    analyst: AnalystSignalConfig
    sector_momentum_caps: SectorMomentumCapsConfig
    macro_drift_levels: dict[str, float]
    historical: HistoricalSignalConfig
    short_interest_brackets: list[ShortInterestBracketConfig]
    peer_rs: PeerRSConfig
    sector_decoupling: SectorDecouplingConfig
    regime_detection: RegimeDetectionConfig
    catalyst_proximity: CatalystProximityConfig


class V3ReviewCriteriaConfig(_StrictModel):
    n_days_min: int = Field(ge=1)
    calibration_dip_target: tuple[float, float]
    calibration_rally_cond_target: tuple[float, float]
    ai_pass2_critique_rate_min: float = Field(ge=0.0, le=1.0)
    catalyst_signal_correlation_min: float = Field(ge=-1.0, le=1.0)
    bag_hold_rate_target: tuple[float, float]


# Per-ticker registry entry. The universe is data, not code (sacred #17 +
# universe-is-config). Adding/removing tickers is a YAML edit.
class TickerConfig(_StrictModel):
    sigma_class: str = Field(pattern=r"^(EXTREME|HIGH|MID)$")
    sector_expected: str
    stock_peers: list[str]
    etf_peer: str = ""              # "" when not configured
    # Per-provider symbol translation (D-W2-14 fallback foundation).
    # When empty, the canonical symbol (the YAML key) is used as-is.
    # Override for tickers where providers diverge (e.g. BRK.B on FMP vs
    # BRK-B on yfinance; today's universe doesn't need it but the registry
    # supports it for future additions).
    fmp_symbol: str = ""
    yf_symbol: str = ""


class DiprallyConfig(_StrictModel):
    version: str
    data: DataConfig
    ai_pricing: AIPricingConfig
    ai_models: AIModelsConfig
    blend_weights_v1: dict[str, float]
    blend_weights_v2: dict[str, float]
    confidence_to_se: dict[str, float]
    conviction: ConvictionConfig
    sigma_class_boundaries: SigmaClassBoundariesConfig
    sigma_classes: dict[str, SigmaClassThresholdConfig]
    horizon: HorizonConfig
    grid: GridConfig
    ai_vol_regime_multipliers: dict[str, float]
    narrative_drift_adjustment: dict[str, float]
    factor_arithmetic: FactorArithmeticConfig
    catalyst: CatalystConfig
    vol_schedule_multipliers: VolScheduleMultipliersConfig
    method_tolerance: MethodToleranceConfig
    backtest: BacktestConfig
    analyst_outlier_threshold: float = Field(gt=0.0)
    trend_filter: TrendFilterConfig
    macro_regime: MacroRegimeConfig
    options_iv: OptionsIVConfig
    sector_perf: SectorPerfConfig
    pde_grid: PDEGridConfig
    garch: GARCHConfig
    realized_vol_windows: list[int]
    engine: EngineConfig
    bayesian: BayesianConfig
    mean_reversion: MeanReversionConfig
    pass2_prompt: Pass2PromptConfig
    sensitivity_scenarios: list[SensitivityScenarioConfig]
    signals: SignalsConfig
    phantom_signal_se: float = Field(gt=0.0, le=1.0)
    ai_cache: AICacheConfig
    bag_hold_terminal_assumption: str
    tickers: dict[str, TickerConfig]
    v3_review_criteria: V3ReviewCriteriaConfig


# =============================================================================
# Load + validate
# =============================================================================

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "diprally.yaml"


def _load_config(path: Path = CONFIG_PATH) -> DiprallyConfig:
    """Load and validate the YAML config. Raises pydantic.ValidationError on
    schema violations (typos, missing keys, out-of-range values, type errors).
    Module-level call — engine cannot start if config is malformed.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = DiprallyConfig(**raw)

    # Cross-field invariants (W3):
    b = config.sigma_class_boundaries
    if b.extreme_min <= b.high_min:
        raise ValueError(
            f"sigma_class_boundaries: extreme_min ({b.extreme_min}) must be "
            f"> high_min ({b.high_min}) — monotonicity violated"
        )
    required_classes = {"EXTREME", "HIGH", "MID"}
    missing = required_classes - set(config.sigma_classes.keys())
    if missing:
        raise ValueError(
            f"sigma_classes table missing required classes: {missing}"
        )
    return config


_CONFIG: DiprallyConfig = _load_config()


def reload_config(path: Path = CONFIG_PATH) -> DiprallyConfig:
    """Reload config from disk. Useful for tests that perturb the YAML
    to verify behavior change without code edits (sacred-#17 acceptance)."""
    global _CONFIG
    _CONFIG = _load_config(path)
    _rebind_module_constants()
    return _CONFIG


# =============================================================================
# Backwards-compatible module-level constants
# Every existing `from src.config import X` import keeps working.
# The values are now sourced from YAML but exposed under the same names.
# =============================================================================

def _rebind_module_constants() -> None:
    """Rebind all module-level constants from _CONFIG. Called at import time
    and again by reload_config() so tests can perturb the YAML and observe
    behavior change with no code edits.
    """
    g = globals()

    # Version + bag hold
    g["V2_VERSION"] = _CONFIG.version
    g["BAG_HOLD_TERMINAL_ASSUMPTION"] = _CONFIG.bag_hold_terminal_assumption

    # Data
    g["FMP_BASE"] = _CONFIG.data.fmp_base_url
    g["DEFAULT_LOOKBACK_DAYS"] = _CONFIG.data.default_lookback_days

    # AI pricing
    g["OPUS_INPUT_PER_TOKEN"] = _CONFIG.ai_pricing.opus_input_per_token
    g["OPUS_OUTPUT_PER_TOKEN"] = _CONFIG.ai_pricing.opus_output_per_token
    g["SONNET_INPUT_PER_TOKEN"] = _CONFIG.ai_pricing.sonnet_input_per_token
    g["SONNET_OUTPUT_PER_TOKEN"] = _CONFIG.ai_pricing.sonnet_output_per_token
    g["HAIKU_INPUT_PER_TOKEN"] = _CONFIG.ai_pricing.haiku_input_per_token
    g["HAIKU_OUTPUT_PER_TOKEN"] = _CONFIG.ai_pricing.haiku_output_per_token
    g["WEB_SEARCH_PER_USE"] = _CONFIG.ai_pricing.web_search_per_use

    # AI models
    g["MODEL_OPUS"] = _CONFIG.ai_models.opus
    g["MODEL_SONNET"] = _CONFIG.ai_models.sonnet
    g["MODEL_HAIKU"] = _CONFIG.ai_models.haiku

    # Pricing dispatch tuple
    g["_AI_PRICING"] = (
        ("opus",   _CONFIG.ai_pricing.opus_input_per_token,   _CONFIG.ai_pricing.opus_output_per_token),
        ("sonnet", _CONFIG.ai_pricing.sonnet_input_per_token, _CONFIG.ai_pricing.sonnet_output_per_token),
        ("haiku",  _CONFIG.ai_pricing.haiku_input_per_token,  _CONFIG.ai_pricing.haiku_output_per_token),
    )

    # Blend weights
    g["BLEND_WEIGHTS"] = dict(_CONFIG.blend_weights_v1)
    g["BLEND_WEIGHTS_V2"] = dict(_CONFIG.blend_weights_v2)
    g["CONFIDENCE_TO_SE"] = dict(_CONFIG.confidence_to_se)

    # Conviction (flat — used as fallback when σ-class auto-detect can't fire)
    g["DEFAULT_CONVICTION_DIP"] = _CONFIG.conviction.dip_marginal
    g["DEFAULT_CONVICTION_RALLY_COND"] = _CONFIG.conviction.rally_conditional
    g["EV_HURDLE_BPS_OF_DIP"] = _CONFIG.conviction.ev_hurdle_bps_of_dip

    # σ-class table (W3 PR #21 — conviction-only this PR; subsequent W3
    # PRs add grid / friction / panic / ai_vol_regime per class)
    g["SIGMA_CLASS_BOUNDARIES"] = _CONFIG.sigma_class_boundaries
    g["SIGMA_CLASSES"] = _CONFIG.sigma_classes

    # Horizon
    g["DEFAULT_HORIZON_DAYS"] = _CONFIG.horizon.default_days
    g["DEFAULT_MC_PATHS"] = _CONFIG.horizon.default_mc_paths
    g["DEEP_DIP_AUTOSCALE_THRESHOLD"] = _CONFIG.horizon.deep_dip_autoscale_threshold
    g["DEEP_DIP_AUTOSCALE_PATHS"] = _CONFIG.horizon.deep_dip_autoscale_paths

    # Grid — step/depth/reach now per-σ-class (W3 PR #22); only
    # panic_floor remains global until PR #24.
    g["PANIC_FLOOR_PCT"] = _CONFIG.grid.panic_floor_pct

    # AI vol regime + narrative
    g["AI_VOL_REGIME_MULTIPLIERS"] = dict(_CONFIG.ai_vol_regime_multipliers)
    g["NARRATIVE_DRIFT_ADJUSTMENT"] = dict(_CONFIG.narrative_drift_adjustment)

    # Factor arithmetic
    g["FACTOR_WEIGHTS"] = {
        "high": _CONFIG.factor_arithmetic.weights.high,
        "med": _CONFIG.factor_arithmetic.weights.med,
        "low": _CONFIG.factor_arithmetic.weights.low,
    }
    g["FACTOR_NET_THRESHOLD"] = _CONFIG.factor_arithmetic.net_threshold
    g["FACTOR_TAIL_BIAS"] = _CONFIG.factor_arithmetic.tail_bias

    # Catalyst
    g["CATALYST_Z_THRESHOLD"] = _CONFIG.catalyst.z_threshold

    # Vol schedule
    vsm = _CONFIG.vol_schedule_multipliers
    g["VOL_SCHEDULE_MULTIPLIERS"] = {
        "self_earnings_day": vsm.self_earnings_day,
        "self_earnings_pre_post": vsm.self_earnings_pre_post,
        "self_earnings_window_days": vsm.self_earnings_window_days,
        "peer_earnings_day": vsm.peer_earnings_day,
        "peer_earnings_pre_post": vsm.peer_earnings_pre_post,
        "peer_earnings_window_days": vsm.peer_earnings_window_days,
        "macro_event_day": vsm.macro_event_day,
    }

    # Method tolerance
    g["METHOD_AGREEMENT_FLOOR_PP"] = _CONFIG.method_tolerance.marginal_floor_pp
    g["METHOD_AGREEMENT_MULTIPLIER"] = _CONFIG.method_tolerance.marginal_multiplier
    g["METHOD_AGREEMENT_FIRST_PASSAGE_FLOOR_PP"] = _CONFIG.method_tolerance.first_passage_floor_pp
    g["METHOD_AGREEMENT_FIRST_PASSAGE_MULTIPLIER"] = _CONFIG.method_tolerance.first_passage_multiplier
    g["METHOD_REFUSAL_MULTIPLIER"] = _CONFIG.method_tolerance.refusal_multiplier
    # Legacy aliases (deprecate in W3)
    g["METHOD_AGREEMENT_TOLERANCE_PP_MARGINAL"] = _CONFIG.method_tolerance.marginal_multiplier * 1.0
    g["METHOD_AGREEMENT_TOLERANCE_PP_FIRST_PASSAGE"] = _CONFIG.method_tolerance.first_passage_multiplier * 1.0
    g["METHOD_AGREEMENT_TOLERANCE_PP"] = g["METHOD_AGREEMENT_TOLERANCE_PP_FIRST_PASSAGE"]

    # Backtest
    g["BACKTEST_MIN_SAMPLES"] = _CONFIG.backtest.min_samples

    # Outlier gate + phantom SE
    g["ANALYST_EXTREME_DRIFT_THRESHOLD"] = _CONFIG.analyst_outlier_threshold
    g["TREND_FILTER_MOM_30D_THRESHOLD"] = _CONFIG.trend_filter.mom_30d_threshold

    # Macro regime (D-W2-8)
    g["VIX_RISK_OFF_THRESHOLD"] = _CONFIG.macro_regime.vix_risk_off_threshold
    g["VIX_RISK_ON_THRESHOLD"] = _CONFIG.macro_regime.vix_risk_on_threshold
    g["SPY_RISK_OFF_THRESHOLD"] = _CONFIG.macro_regime.spy_risk_off_threshold
    g["SPY_RISK_ON_THRESHOLD"] = _CONFIG.macro_regime.spy_risk_on_threshold
    g["VIX_DEFAULT_FALLBACK"] = _CONFIG.macro_regime.vix_default_fallback

    # Options IV liquidity gate (D-W2-8)
    g["OPTIONS_IV_LIQUIDITY_MAX_SPREAD"] = _CONFIG.options_iv.liquidity_max_bid_ask_pct
    g["OPTIONS_IV_DEFAULT_TARGET_DTE_DAYS"] = _CONFIG.options_iv.default_target_dte_days
    g["OPTIONS_IV_DTE_WINDOW_MIN"] = _CONFIG.options_iv.dte_window_min
    g["OPTIONS_IV_DTE_WINDOW_MAX_MULTIPLIER"] = _CONFIG.options_iv.dte_window_max_multiplier

    # Sector perf (D-W2-8)
    g["SECTOR_PERF_DEFAULT_LOOKBACK_DAYS"] = _CONFIG.sector_perf.default_lookback_days

    # PDE grid (D-W2-9)
    g["PDE_N_SPACE"] = _CONFIG.pde_grid.n_space
    g["PDE_N_TIME"] = _CONFIG.pde_grid.n_time

    # GARCH (D-W2-9)
    g["GARCH_MIN_DATA_BARS"] = _CONFIG.garch.min_data_bars
    g["GARCH_FALLBACK_BARS"] = _CONFIG.garch.fallback_bars
    g["GARCH_INITIAL_OMEGA"] = _CONFIG.garch.initial_omega
    g["GARCH_INITIAL_OMEGA_FULL"] = _CONFIG.garch.initial_omega_full
    g["GARCH_INITIAL_ALPHA"] = _CONFIG.garch.initial_alpha
    g["GARCH_INITIAL_BETA"] = _CONFIG.garch.initial_beta

    # Realized vol windows (D-W2-9)
    g["REALIZED_VOL_WINDOWS"] = tuple(_CONFIG.realized_vol_windows)

    # Engine scattered (D-W2-5 + D-W2-7). SPREAD_PER_SHARE_ROUND_TRIP
    # retired in W3 PR #23 — friction is per-σ-class bps now.
    g["DRIFT_CAP"] = _CONFIG.engine.drift_cap
    g["GARCH_FALLBACK_SIGMA"] = _CONFIG.engine.garch_fallback_sigma
    g["GRID_PREFILTER_LOOSENESS"] = _CONFIG.engine.grid_prefilter_looseness

    # Bayesian (D-W2-7)
    g["BAYESIAN_PRIOR_AGE_INFLATION_PER_DAY"] = _CONFIG.bayesian.prior_age_inflation_per_day
    g["BAYESIAN_DEFAULT_PRIOR_STD"] = _CONFIG.bayesian.default_prior_std
    g["BAYESIAN_STD_FLOOR"] = _CONFIG.bayesian.std_floor
    g["BAYESIAN_DEFAULT_TODAY_STD"] = _CONFIG.bayesian.default_today_std_when_blend_fails

    # Mean reversion + Pass 2 prompt (D-W2-7)
    g["MEAN_REVERSION_ANCHOR_PCT_BELOW_SPOT"] = _CONFIG.mean_reversion.anchor_pct_below_spot
    g["PASS2_CLOSED_FORM_BRACKET_PCT"] = _CONFIG.pass2_prompt.closed_form_bracket_pct

    # Sensitivity scenarios (D-W2-7) — list of dicts for downstream consumers
    g["SENSITIVITY_SCENARIOS"] = [
        {"label": s.label, "drift_offset": s.drift_offset,
         "sigma_multiplier": s.sigma_multiplier}
        for s in _CONFIG.sensitivity_scenarios
    ]

    # Signal-level embedded thresholds (D-W2-6)
    sig = _CONFIG.signals
    g["SIGNAL_ANALYST"] = sig.analyst
    g["SIGNAL_SECTOR_MOMENTUM_CAPS"] = sig.sector_momentum_caps
    g["SIGNAL_MACRO_DRIFT_LEVELS"] = dict(sig.macro_drift_levels)
    g["SIGNAL_HISTORICAL"] = sig.historical
    g["SIGNAL_SHORT_INTEREST_BRACKETS"] = sig.short_interest_brackets
    g["SIGNAL_PEER_RS"] = sig.peer_rs
    g["SIGNAL_SECTOR_DECOUPLING"] = sig.sector_decoupling
    g["SIGNAL_REGIME_DETECTION"] = sig.regime_detection
    g["SIGNAL_CATALYST_PROXIMITY"] = sig.catalyst_proximity
    g["PHANTOM_SIGNAL_SE_CONFIG"] = _CONFIG.phantom_signal_se  # signals.py reads PHANTOM_SIGNAL_SE locally

    # AI cache
    g["AI_CACHE_SPOT_MOVE_INVALIDATION_PCT"] = _CONFIG.ai_cache.spot_move_invalidation_pct

    # v3 review criteria
    v3 = _CONFIG.v3_review_criteria
    g["V3_REVIEW_CRITERIA"] = {
        "n_days_min": v3.n_days_min,
        "calibration_dip_target": v3.calibration_dip_target,
        "calibration_rally_cond_target": v3.calibration_rally_cond_target,
        "ai_pass2_critique_rate_min": v3.ai_pass2_critique_rate_min,
        "catalyst_signal_correlation_min": v3.catalyst_signal_correlation_min,
        "bag_hold_rate_target": v3.bag_hold_rate_target,
    }


# Bind on import so all `from src.config import X` calls work.
_rebind_module_constants()


# =============================================================================
# Public helpers (unchanged signatures)
# =============================================================================

def pricing_for_model(model_id: str) -> tuple[float, float]:
    """Return (input_per_token, output_per_token) for the given model ID.
    Matches by case-insensitive substring on the canonical family name.
    Defaults to Opus pricing for unknown IDs (conservative — overstates cost
    rather than understates).
    """
    lower = (model_id or "").lower()
    for prefix, in_rate, out_rate in _AI_PRICING:  # noqa: F821 (set by _rebind)
        if prefix in lower:
            return in_rate, out_rate
    return OPUS_INPUT_PER_TOKEN, OPUS_OUTPUT_PER_TOKEN  # noqa: F821


def method_tolerance_pp(sigma_effective: float, kind: str = "marginal") -> float:
    """σ-scaled flag tolerance (pp). kind in {'marginal', 'first_passage'}.
    Marginal = P(touch ever). First-passage = P(dip first | rally first).
    """
    if kind == "first_passage":
        return max(METHOD_AGREEMENT_FIRST_PASSAGE_FLOOR_PP,  # noqa: F821
                   METHOD_AGREEMENT_FIRST_PASSAGE_MULTIPLIER * sigma_effective)  # noqa: F821
    return max(METHOD_AGREEMENT_FLOOR_PP,  # noqa: F821
               METHOD_AGREEMENT_MULTIPLIER * sigma_effective)  # noqa: F821


def method_refusal_pp(sigma_effective: float, kind: str = "marginal") -> float:
    """Hard refusal threshold (pp). Triggers the sacred-decision-#16 gate."""
    return METHOD_REFUSAL_MULTIPLIER * method_tolerance_pp(sigma_effective, kind)  # noqa: F821

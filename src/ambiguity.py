"""Per-ticker ambiguity score (W4 PR #28).

Single scalar in [0, 1] indicating "how worth it is to spend AI tokens
on this ticker." Higher = more ambiguous = AI catalyst/narrative input
is more likely to flip the decision. Computed from T0 (math-only)
outputs, so the broker can rank all 17 tickers BEFORE any AI is
dispatched.

Components (each normalized to [0, 1]; weighted sum is the overall):

  conviction_proximity (0.40)  — best pair's p_dip near the conviction
                                 threshold. p_dip == threshold → 1.0;
                                 falls to 0 when 20pp away in either
                                 direction (clearly qualifies OR clearly
                                 falls short, both unambiguous).
  ev_hurdle_proximity  (0.25)  — best pair's ev_pct_of_dip near sacred
                                 #13 hurdle (50 bps default). 50bps → 1.0;
                                 decays to 0 when 50bps away.
  sigma_divergence     (0.15)  — anchor disagreement on σ — high
                                 divergence means we don't trust our
                                 own vol estimate, so AI vol_regime call
                                 has leverage.
  method_proximity     (0.10)  — MC vs PDE delta near refusal threshold.
                                 Borderline-disagreement runs benefit
                                 from AI sanity-check; clean agreement
                                 doesn't.
  trend_proximity      (0.10)  — mom_30d near sacred #14 trend filter
                                 (-25%). Near-knife tickers need AI
                                 catalyst surface to know if there's a
                                 thesis; clean uptrends don't.

Special cases:
  - No best pair found (math returned None): conviction_proximity = 0.5
    (we have no idea if AI catalyst input would surface a setup or
    not). Other components from available data.
  - All-zero inputs (test smoke): overall = 0.0 (don't spend AI).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.config import EV_HURDLE_BPS_OF_DIP, TREND_FILTER_MOM_30D_THRESHOLD


@dataclass(frozen=True)
class AmbiguityScore:
    overall: float                  # ∈ [0, 1]
    components: dict[str, float] = field(default_factory=dict)


# Component weights — sum to 1.0. Class-level constant so the report
# can document the formula and the broker can introspect.
WEIGHTS = {
    "conviction_proximity": 0.40,
    "ev_hurdle_proximity":  0.25,
    "sigma_divergence":     0.15,
    "method_proximity":     0.10,
    "trend_proximity":      0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "ambiguity weights must sum to 1.0"


def _tent(value: float, target: float, half_width: float) -> float:
    """Tent function — 1.0 at target, decaying linearly to 0 at ±half_width.
    Used for "proximity to boundary" metrics where being EXACTLY on the
    boundary is maximum ambiguity."""
    if half_width <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(value - target) / half_width)


def _saturate(value: float, full_scale: float) -> float:
    """Linear ramp 0→1 over [0, full_scale], clamped at 1.0. Used for
    "more is more" metrics (divergence — bigger = more ambiguous)."""
    if full_scale <= 0:
        return 0.0
    return max(0.0, min(1.0, value / full_scale))


def compute_ambiguity(
    best_p_dip: Optional[float],
    conviction_dip: float,
    best_ev_pct_of_dip: Optional[float],
    sigma_divergence_pp: float,
    method_max_delta_pp: float,
    method_refuse_threshold_pp: float,
    mom_30d: float,
) -> AmbiguityScore:
    """Compute the ambiguity score from T0 math outputs.

    All inputs come from a single ticker's math pipeline; this function
    is pure (no I/O, no config reads beyond module-load constants).

    best_p_dip / best_ev_pct_of_dip are None when no candidate pair
    survived the grid prefilter — handled as "moderately ambiguous"
    (overall .conviction component = 0.5) since AI catalyst data could
    plausibly flip a no-pair run by tightening σ or shifting drift.
    """
    # 1. Conviction proximity: best pair's marginal p_dip near threshold.
    if best_p_dip is None:
        conviction_proximity = 0.5
    else:
        # Half-width 0.20: if p_dip is more than 20pp away from threshold,
        # the call is unambiguous (clearly qualifies or clearly doesn't).
        conviction_proximity = _tent(best_p_dip, conviction_dip, 0.20)

    # 2. EV-hurdle proximity: best pair's ev_pct_of_dip near sacred #13 hurdle.
    ev_hurdle_threshold = EV_HURDLE_BPS_OF_DIP / 10000.0
    if best_ev_pct_of_dip is None:
        ev_hurdle_proximity = 0.5
    else:
        # Half-width = the hurdle itself (50bps) — being on the wrong
        # side by 50bps means it's a clean refuse, on the right side by
        # 50bps means it's a clean qualify.
        ev_hurdle_proximity = _tent(
            best_ev_pct_of_dip, ev_hurdle_threshold, ev_hurdle_threshold
        )

    # 3. σ divergence: anchor disagreement. Higher = more ambiguous.
    # Full-scale 20pp: σ anchors disagreeing by 20pp is very bad.
    sigma_divergence = _saturate(sigma_divergence_pp, 20.0)

    # 4. Method-disagreement proximity: how close are we to sacred #16
    # refusal? Tent peaks AT the refusal threshold (right on the edge
    # is maximum ambiguity), decays as delta gets smaller (clean math)
    # or further past the threshold (clean refusal).
    method_proximity = _tent(
        method_max_delta_pp,
        method_refuse_threshold_pp,
        method_refuse_threshold_pp,
    )

    # 5. Trend filter proximity (sacred #14): mom_30d near -25%.
    # Half-width 0.10 (10pp of momentum): mom_30d at exactly the trend
    # filter is the most ambiguous spot — AI catalyst surface could
    # tell us whether there's a thesis to override.
    trend_proximity = _tent(mom_30d, TREND_FILTER_MOM_30D_THRESHOLD, 0.10)

    components = {
        "conviction_proximity": conviction_proximity,
        "ev_hurdle_proximity": ev_hurdle_proximity,
        "sigma_divergence": sigma_divergence,
        "method_proximity": method_proximity,
        "trend_proximity": trend_proximity,
    }
    overall = sum(WEIGHTS[k] * v for k, v in components.items())
    # Clamp defensively — float rounding could push fractionally above 1.0.
    overall = max(0.0, min(1.0, overall))
    return AmbiguityScore(overall=overall, components=components)

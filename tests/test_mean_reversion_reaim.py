"""Regression tests for the 2026-05-30 mean-reversion re-aim.

Pre-2026-05-30: YAML had `anchor_pct_below_spot: 0.05` (5% BELOW spot)
and CLI --mean-reversion defaulted to 0.0 (OFF). With anchor BELOW spot,
the layer pulled paths DOWN from current spot — wrong direction for a
dip-rally thesis.

Post-2026-05-30: YAML has `anchor_pct_above_spot: 0.0` (at-spot) and
`default_strength: 2.0` (ON by default). At-spot anchor provides
symmetric vol-drag suppression without an editorial direction bias.

These tests guard:
  1. The YAML loads with the new keys (catches a partial rename)
  2. MR-ON with at-spot anchor lifts EV vs MR-OFF baseline at high σ
  3. Falling-knife (-30% μ) stays REFUSED — no false-positive rescue
  4. MR layer is INERT when strength = 0.0 (legacy behavior preserved)
"""
from __future__ import annotations

import numpy as np

from src.config import (
    MEAN_REVERSION_ANCHOR_PCT_ABOVE_SPOT,
    MEAN_REVERSION_DEFAULT_STRENGTH,
    PATIENCE_WINDOW_TD,
)
from src.math_utils import (
    run_mc_joint_conditional,
    precompute_first_touch_days,
    compute_dual_ev,
)


S0 = 100.0
H = 20
N_PATHS = 100_000
SEED = 42


def _paths(sigma: float, mu: float, k: float, df: float = 4.0) -> np.ndarray:
    """Generate MC paths with optional mean-reversion. Anchor = S0 * (1 + above_pct)
    pulled from config to match production exactly."""
    anchor = (
        S0 * (1.0 + MEAN_REVERSION_ANCHOR_PCT_ABOVE_SPOT)
        if k > 0 else None
    )
    return run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu, horizon_days=H, n_paths=N_PATHS,
        distribution="student_t", df=df, seed=SEED,
        mean_reversion_strength=k,
        mean_reversion_anchor=anchor,
    )


def _dual(paths: np.ndarray, sigma: float, dip: float, rally: float,
          friction_bps_rt: float, swing_stop_pct: float) -> dict:
    """Bridge-corrected first-touch + dual-EV with stop layer."""
    dip_first = precompute_first_touch_days(
        paths, S0, np.array([dip]), sigma, None, "down", seed=42,
    )[:, 0]
    rally_first = precompute_first_touch_days(
        paths, S0, np.array([rally]), sigma, None, "up", seed=43,
    )[:, 0]
    fric_w = (dip + rally) / 2 * friction_bps_rt / 10000
    fric_d = (S0 + rally) / 2 * friction_bps_rt / 10000
    return compute_dual_ev(
        paths, S0, dip, rally, fric_w,
        dip_first_days=dip_first, rally_first_days=rally_first,
        patience_window_td=PATIENCE_WINDOW_TD,
        swing_stop_pct=swing_stop_pct,
        friction_per_share_direct=fric_d,
    )


# ---------------------------------------------------------------------------
# 1. YAML migration guards
# ---------------------------------------------------------------------------

def test_yaml_loads_new_mean_reversion_keys():
    """Catches a partial rename where one key changed but the other didn't,
    or someone reverted only YAML / only the schema."""
    # at-spot anchor as the post-2026-05-30 default
    assert MEAN_REVERSION_ANCHOR_PCT_ABOVE_SPOT == 0.0
    # strength turned ON, k=2.0
    assert MEAN_REVERSION_DEFAULT_STRENGTH == 2.0


# ---------------------------------------------------------------------------
# 2. MR-ON improves EV at high σ (the case 0-BUY audit cared about)
# ---------------------------------------------------------------------------

def test_mr_on_lifts_ev_above_mr_off_at_extreme_sigma():
    """EXTREME σ=1.30 μ=+12% (LWLG-shaped) — at-spot MR with k=2 must lift
    the chosen EV by at least 8 bps (0.08%) above the MR-OFF baseline.
    Synth-measured swing was +12 bps (+0.12%); test asserts ≥ 8 bps margin
    for MC noise tolerance."""
    sigma = 1.30
    mu = 0.12
    dip, rally = 92.0, 110.0
    fric_bps = 70.0

    off = _dual(_paths(sigma, mu, k=0.0), sigma, dip, rally, fric_bps, 0.10)
    on = _dual(_paths(sigma, mu, k=2.0), sigma, dip, rally, fric_bps, 0.10)

    chosen_off = max(off["ev_wait_pct_of_dip"], off["ev_direct_pct_of_spot"])
    chosen_on = max(on["ev_wait_pct_of_dip"], on["ev_direct_pct_of_spot"])
    lift = chosen_on - chosen_off
    assert lift > 0.0008, (
        f"expected MR(k=2) to lift chosen EV by ≥ +8 bps (+0.08%) at "
        f"EXTREME σ=1.30 μ=+12%, got {lift*100:+.3f}% "
        f"(off={chosen_off*100:+.2f}% / on={chosen_on*100:+.2f}%)"
    )


# ---------------------------------------------------------------------------
# 3. Falling knife stays REFUSED — MR doesn't fantasy-rescue impaired thesis
# ---------------------------------------------------------------------------

def test_mr_does_not_create_false_positive_buy_on_falling_knife():
    """Sacred-#14-shaped setup: EXTREME σ=1.50 μ=-30% (deeply impaired
    drift). The conservative at-spot k=2.0 calibration must keep this
    REFUSED, i.e. chosen EV < EXTREME's 25 bps (0.25%) hurdle.

    More aggressive calibrations (anchor +5%, k=5) DO create spurious
    BUYs here; that's why we picked conservative. Sacred #14's trend
    filter is a SEPARATE gate at mom_30d < -25%, but this test guards
    the MC-only layer too so the engine has defense-in-depth."""
    sigma = 1.50
    mu = -0.30
    dip, rally = 92.0, 110.0
    fric_bps = 70.0
    paths = _paths(sigma, mu, k=2.0)
    out = _dual(paths, sigma, dip, rally, fric_bps, 0.10)
    chosen = max(out["ev_wait_pct_of_dip"], out["ev_direct_pct_of_spot"])
    # 25 bps (0.25%) is EXTREME's σ-class hurdle.
    assert chosen < 0.0025, (
        f"FALSE-POSITIVE GUARD failed: -30%/yr falling knife produces "
        f"chosen EV {chosen*100:+.2f}% ≥ +0.25% hurdle. MR calibration "
        f"is too aggressive — rescuing impaired-thesis paths the engine "
        f"should be refusing. Tighten anchor_pct_above_spot or "
        f"default_strength in config/diprally.yaml."
    )


# ---------------------------------------------------------------------------
# 4. Backward compatibility — k=0 reproduces pure GBM
# ---------------------------------------------------------------------------

def test_mr_strength_zero_reproduces_pure_gbm():
    """Passing mean_reversion_strength=0.0 to run_mc_joint_conditional
    must produce identical paths to the no-MR call (bit-exact, same seed).
    Guards against accidental side-effect when MR is supposed to be inert."""
    common = dict(
        S0=S0, sigma=0.80, mu=0.10, horizon_days=H, n_paths=10_000,
        distribution="student_t", df=5.0, seed=42,
    )
    paths_off = run_mc_joint_conditional(
        **common, mean_reversion_strength=0.0, mean_reversion_anchor=None,
    )
    paths_off_with_anchor = run_mc_joint_conditional(
        **common, mean_reversion_strength=0.0, mean_reversion_anchor=105.0,
    )
    # When strength is 0, the anchor value is irrelevant — both calls
    # must produce bit-exact identical paths.
    assert np.array_equal(paths_off, paths_off_with_anchor)

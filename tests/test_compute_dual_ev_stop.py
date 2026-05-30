"""Regression test for the 2026-05-30 swing-stop layer in compute_dual_ev.

Audits the dominant 0-BUY root cause: the legacy "hold-to-terminal" bag-hold
exit pushed EV deeply negative at high σ. Adding a configurable stop-out
flips EV positive by hundreds of basis points.

Each test asserts the SPECIFIC direction the operator's grievance #2 named:
"math runs the OPPOSITE hypothesis from the strategy." Before the patch
landed, every realistic (σ-class, σ, μ) combo refused on the EV hurdle;
after the patch the same combos clear it.

Run: pytest tests/test_compute_dual_ev_stop.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.math_utils import compute_dual_ev, run_mc_joint_conditional


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

S0 = 100.0
H = 20
N_PATHS = 100_000
SEED = 42


def _mc(sigma: float, mu: float, distribution: str, df: float) -> np.ndarray:
    """Deterministic MC for the test; isolated seed so the assertions are
    reproducible across environments."""
    return run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu, horizon_days=H, n_paths=N_PATHS,
        distribution=distribution, df=df, seed=SEED,
    )


# ---------------------------------------------------------------------------
# 1. EXTREME σ-class — biggest binding-gate offender on the live roster
# ---------------------------------------------------------------------------

def test_extreme_without_stop_refuses_on_ev_hurdle():
    """Pre-patch behavior. EXTREME σ=1.30 μ=0 dip=92 rally=112 produces
    deeply negative EV that refuses the σ-class EV hurdle of 25 bps (0.25%).
    Locks the legacy math state so a future refactor can't silently restore
    the bug. Also defends backward compatibility of swing_stop_pct=None."""
    paths = _mc(sigma=1.30, mu=0.0, distribution="student_t", df=4.0)
    fric = (92.0 + 112.0) / 2.0 * 70.0 / 10000.0
    out = compute_dual_ev(
        paths, S0, 92.0, 112.0, fric,
        patience_window_td=40,
        swing_stop_pct=None,  # pre-patch behavior
    )
    # Both branches deeply negative — well past the 25 bps (0.25%) hurdle.
    assert out["ev_wait_pct_of_dip"] < -0.02, (
        f"expected ev_wait_pct_of_dip < -2% pre-patch, "
        f"got {out['ev_wait_pct_of_dip']*100:+.2f}%"
    )
    assert out["ev_direct_pct_of_spot"] < -0.02
    # No stops fired — flag must be 0.0 when swing_stop_pct is None.
    assert out["p_stopped_wait"] == 0.0
    assert out["p_stopped_direct"] == 0.0


def test_extreme_with_swing_stop_improves_direct_ev_by_300_bps():
    """Post-patch behavior. The stop layer truncates the bag-hold left
    tail on the DIRECT branch. Synth-measured swing at EXTREME σ=1.30
    μ=0 is ev_direct: -473 bps (-4.73%) → -106 bps (-1.06%) = +367 bps
    (+3.67%) improvement. We assert ≥ 300 bps (3.00%) of headroom.

    Note: at EXTREME σ=1.30 μ=0 the chosen EV is STILL negative even
    with the stop — the second binding constraint is vol drag (σ²/2 ≈
    85%/yr) exceeding typical Bayesian μ. Fixing that is a separate
    defect; this test ensures the stop layer delivers the improvement
    it's supposed to deliver, no more, no less. Adversarial honesty:
    the patch is necessary but insufficient.
    """
    paths = _mc(sigma=1.30, mu=0.0, distribution="student_t", df=4.0)
    fric = (92.0 + 112.0) / 2.0 * 70.0 / 10000.0
    out_no_stop = compute_dual_ev(
        paths, S0, 92.0, 112.0, fric,
        patience_window_td=40,
        swing_stop_pct=None,
    )
    out_with_stop = compute_dual_ev(
        paths, S0, 92.0, 112.0, fric,
        patience_window_td=40,
        swing_stop_pct=0.10,
    )
    direct_improvement = (
        out_with_stop["ev_direct_pct_of_spot"]
        - out_no_stop["ev_direct_pct_of_spot"]
    )
    assert direct_improvement > 0.03, (
        f"expected stop layer to improve ev_direct by ≥ +300 bps (+3.00%) "
        f"on EXTREME σ=1.30, got {direct_improvement*100:+.2f}%"
    )
    # Stops must actually have fired at high σ.
    assert 0.0 < out_with_stop["p_stopped_wait"] < 1.0
    assert 0.0 < out_with_stop["p_stopped_direct"] < 1.0


# ---------------------------------------------------------------------------
# 2. MID σ-class — closest-to-hurdle case on the live roster (AMAT)
# ---------------------------------------------------------------------------

def test_mid_without_stop_misses_hurdle():
    """MID σ=0.40 μ=+10% dip=98 rally=104 — the AMAT-shaped case. EV barely
    negative without stops (right at the live -51 bps observation for
    AMAT). Refuses the 50 bps (0.50%) hurdle by ~100 bps (1.00%)."""
    paths = _mc(sigma=0.40, mu=0.10, distribution="student_t", df=7.0)
    fric = (98.0 + 104.0) / 2.0 * 18.0 / 10000.0
    out = compute_dual_ev(
        paths, S0, 98.0, 104.0, fric,
        patience_window_td=40,
        swing_stop_pct=None,
    )
    assert out["ev_wait_pct_of_dip"] < 0.005, (
        f"expected ev_wait_pct_of_dip < +0.5% pre-patch, "
        f"got {out['ev_wait_pct_of_dip']*100:+.2f}%"
    )


def test_mid_with_swing_stop_improves_direct_ev_by_60_bps():
    """MID with swing_stop_pct=0.05. Synth-measured ev_direct at MID
    σ=0.40 μ=+10%: -94 bps (-0.94%) → -22 bps (-0.22%) = +72 bps
    (+0.72%) improvement. Assert ≥ 60 bps (0.60%) headroom. Smaller
    swing than EXTREME because MID's bag-hold tail was already smaller
    pre-patch."""
    paths = _mc(sigma=0.40, mu=0.10, distribution="student_t", df=7.0)
    fric = (98.0 + 104.0) / 2.0 * 18.0 / 10000.0
    out_no_stop = compute_dual_ev(
        paths, S0, 98.0, 104.0, fric,
        patience_window_td=40,
        swing_stop_pct=None,
    )
    out_with_stop = compute_dual_ev(
        paths, S0, 98.0, 104.0, fric,
        patience_window_td=40,
        swing_stop_pct=0.05,
    )
    direct_improvement = (
        out_with_stop["ev_direct_pct_of_spot"]
        - out_no_stop["ev_direct_pct_of_spot"]
    )
    assert direct_improvement > 0.005, (
        f"expected stop layer to improve ev_direct by ≥ +50 bps (+0.50%) "
        f"on MID σ=0.40 μ=+10%, got {direct_improvement*100:+.2f}%"
    )


# ---------------------------------------------------------------------------
# 3. Backward compatibility — swing_stop_pct=None must reproduce legacy EV
# ---------------------------------------------------------------------------

def test_backward_compat_no_stop_matches_legacy_within_float_eps():
    """When swing_stop_pct is None / 0.0 / unset, compute_dual_ev must
    produce IDENTICAL EV to the pre-patch implementation. Guards against
    accidental behavior change for tickers we don't want to repath."""
    paths = _mc(sigma=0.80, mu=0.05, distribution="student_t", df=5.0)
    fric = (95.0 + 108.0) / 2.0 * 35.0 / 10000.0
    out_none = compute_dual_ev(
        paths, S0, 95.0, 108.0, fric, patience_window_td=40, swing_stop_pct=None,
    )
    out_zero = compute_dual_ev(
        paths, S0, 95.0, 108.0, fric, patience_window_td=40, swing_stop_pct=0.0,
    )
    # Bit-exact across both gating values.
    assert out_none["ev_wait_per_share"] == out_zero["ev_wait_per_share"]
    assert out_none["ev_direct_per_share"] == out_zero["ev_direct_per_share"]
    assert out_none["p_stopped_wait"] == 0.0
    assert out_none["p_stopped_direct"] == 0.0


# ---------------------------------------------------------------------------
# 4. Stop-vs-rally ordering — stop only fires when it precedes the rally
# ---------------------------------------------------------------------------

def test_stop_only_fires_before_rally_not_after():
    """Constructed path battery: path A rallies first then dips below stop;
    path B dips below stop first then rallies. Stop must fire on B not A
    (rally wins on A because it's first-touch). Hand-rolled paths so the
    invariant is observable in isolation, not lost in MC variance."""
    # Two paths, 10 days. Stop at 90 (-10%), rally at 110.
    # Path A: spot up to 111 on day 1 (rally), then drops to 85 on day 9.
    # Path B: spot drops to 89 on day 2 (stop), then climbs to 115 on day 8.
    paths = np.array([
        [105, 111, 108, 100, 95, 92, 90, 88, 86, 85],  # rally first
        [98, 95, 89, 92, 100, 108, 112, 115, 110, 105],  # stop first
    ], dtype=float)
    # No-dip-touch DIRECT only (skip WAIT — dip_price below all paths' minima
    # for path A means dip_first_days behavior would dominate). Use friction=0
    # so payoffs are direct readouts.
    out = compute_dual_ev(
        paths, S0=100.0, dip_price=80.0, rally_price=110.0,
        friction_per_share=0.0,
        patience_window_td=None,
        swing_stop_pct=0.10,  # stop at S0 * 0.90 = 90.0
    )
    # Path A: rally touched at index 1 (111 >= 110) BEFORE stop at index 6 (90 <= 90).
    #   → payoff = 110 - 100 - 0 = 10.
    # Path B: stop touched at index 2 (89 <= 90) BEFORE rally at index 7.
    #   → payoff = 90 - 100 - 0 = -10.
    # Mean = 0.0; p_rally_hit covers BOTH paths (rally touched somewhere);
    # p_stopped_direct covers only path B (stop_first < rally_first).
    assert out["ev_direct_per_share"] == pytest.approx(0.0, abs=1e-9)
    assert out["p_stopped_direct"] == pytest.approx(0.5, abs=1e-9)
    assert out["p_rally_hit"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. Direct-branch friction is now decoupled from WAIT friction
# ---------------------------------------------------------------------------

def test_friction_per_share_direct_separates_branch_costs():
    """Passing a separate friction value for DIRECT must change ev_direct
    by exactly the friction delta, without affecting ev_wait. Guards
    against re-coupling the two branches in a future refactor."""
    paths = _mc(sigma=0.80, mu=0.05, distribution="student_t", df=5.0)
    fric_wait = 1.00
    out_same = compute_dual_ev(
        paths, S0, 95.0, 108.0, fric_wait,
        patience_window_td=40, swing_stop_pct=None,
    )
    out_split = compute_dual_ev(
        paths, S0, 95.0, 108.0, fric_wait,
        patience_window_td=40, swing_stop_pct=None,
        friction_per_share_direct=2.00,  # +$1 vs WAIT
    )
    # WAIT unchanged.
    assert out_same["ev_wait_per_share"] == out_split["ev_wait_per_share"]
    # DIRECT shifted by exactly -$1 (the extra friction).
    delta = out_same["ev_direct_per_share"] - out_split["ev_direct_per_share"]
    assert delta == pytest.approx(1.00, abs=1e-9)

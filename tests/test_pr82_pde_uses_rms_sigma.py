"""Tests for PR #82 + PR #84 — RMS-equivalent sigma for cross-check.

PR #82 introduced `sigma_eq` to align PDE/closed-form with the MC's
variance over the path. PR #84 fixed the formula bug: vol_schedule
is ABSOLUTE volatility per day (`base_sigma × multipliers`), not
dimensionless multipliers. Original formula `sigma * sqrt(mean(vs**2))`
double-counted sigma, putting closed-form vol an order of magnitude
too low → 15-22pp marginal-touch divergence vs MC → false-positive
sacred-#16 refusal on ~5/26 stable names every cycle. Fix:

    sigma_eq = sqrt(mean(vol_schedule**2))

Numerically verified: MC and closed-form now agree within ~1pp for
both flat and earnings-spike schedules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _mk_mc(p_round_trip=0.30, p_bag_hold=0.10, p_no_trade_rally_first=0.20,
            p_neither=0.40):
    return {
        "p_round_trip": p_round_trip,
        "p_bag_hold": p_bag_hold,
        "p_no_trade_rally_first": p_no_trade_rally_first,
        "p_neither": p_neither,
        "p_dip_touched_any": p_round_trip + p_bag_hold,
        "p_rally_touched_any": p_round_trip + p_no_trade_rally_first,
    }


# =============================================================================
# Backward-compat / no-schedule path
# =============================================================================

def test_no_vol_schedule_uses_constant_sigma():
    """Legacy callers that don't pass vol_schedule (e.g.
    test_refusal_gate.py) get sigma_eq = sigma. Output unchanged."""
    from src.math_utils import three_method_cross_check
    chk_implicit = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.05, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
    )
    chk_explicit_none = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.05, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
        vol_schedule=None,
    )
    assert chk_implicit["tolerances"]["sigma_eq_pde"] == 0.30
    assert chk_implicit == chk_explicit_none


# =============================================================================
# sigma_eq formula — uses ABSOLUTE vol_schedule, NOT multipliers
# =============================================================================

def test_flat_absolute_schedule_recovers_constant_sigma():
    """When vol_schedule is flat at the base sigma value
    (vol_schedule = [σ, σ, σ, ...]), sigma_eq must equal σ. Earlier
    bug (PR #82 → PR #84): formula was `sigma * sqrt(mean(vs²))`
    which gave σ² when vs = σ — i.e., too small by factor 1/σ."""
    from src.math_utils import three_method_cross_check
    sigma = 0.30
    flat = np.full(60, sigma)  # absolute vol per day, flat
    chk = three_method_cross_check(
        S0=100.0, sigma=sigma, mu=0.05, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
        vol_schedule=flat,
    )
    assert chk["tolerances"]["sigma_eq_pde"] == pytest.approx(sigma)


def test_earnings_spike_schedule_inflates_sigma_eq():
    """vol_schedule with a 1.5× earnings spike around 4 days →
    sigma_eq slightly larger than baseline. Magnitude bounded."""
    from src.math_utils import three_method_cross_check
    sigma = 0.40
    sched = np.full(60, sigma)
    sched[20:24] *= 1.5  # 4-day earnings spike at 1.5× baseline vol
    chk = three_method_cross_check(
        S0=100.0, sigma=sigma, mu=0.0, horizon_days=60,
        dip_price=95.0, rally_price=110.0, mc_result=_mk_mc(),
        vol_schedule=sched,
    )
    sigma_eq = chk["tolerances"]["sigma_eq_pde"]
    expected = float(np.sqrt(np.mean(sched ** 2)))
    assert sigma_eq == pytest.approx(expected)
    assert sigma_eq > sigma                      # inflated by the spike
    assert sigma_eq < sigma * 1.5                # but bounded


def test_sigma_eq_surfaced_with_metadata():
    """sigma_eq_pde, fp_widening_factor, schedule_heterogeneity all
    surface so the operator can see what the cross-check actually used."""
    from src.math_utils import three_method_cross_check
    sigma = 0.40
    sched = np.full(60, sigma)
    sched[20] = sigma * 2.0
    chk = three_method_cross_check(
        S0=100.0, sigma=sigma, mu=0.0, horizon_days=60,
        dip_price=95.0, rally_price=110.0, mc_result=_mk_mc(),
        vol_schedule=sched,
    )
    tols = chk["tolerances"]
    assert "sigma_eq_pde" in tols
    assert "fp_widening_factor" in tols
    assert "schedule_heterogeneity" in tols
    # sigma_eq differs from sigma_used (the schedule has a spike).
    assert tols["sigma_eq_pde"] != tols["sigma_used"]


# =============================================================================
# End-to-end MC vs closed-form convergence — the regression test that
# would have caught PR #82's bug.
# =============================================================================

@pytest.mark.parametrize("scenario", [
    {"name": "flat",  "build": lambda sig, h: np.full(h, sig)},
    {"name": "spike", "build": lambda sig, h: (lambda s: (s.__setitem__(slice(20, 24), sig * 1.5), s)[1])(np.full(h, sig))},
])
def test_mc_marginal_matches_closed_form_within_2pp(scenario):
    """The regression that would have prevented three speculative PRs.

    Run MC with a vol_schedule (absolute vol per day, flat or spiked),
    compute marginal touch probabilities, and verify they match the
    closed-form with sigma_eq within 2pp.

    AMAT cycle-3 evidence: pre-PR-#84, MC vs CF marginal gap was
    15-22pp. Post-#84: < 1pp on the AMAT inputs. Threshold 2pp here
    leaves headroom for MC sampling noise at 100k paths.
    """
    from src.math_utils import (
        analyze_joint_conditional,
        closed_touch_down,
        closed_touch_up,
        run_mc_joint_conditional,
    )
    # AMAT-like setup.
    S0, sigma, mu = 455.81, 0.53, 0.05
    dip, rally = 430.0, 510.0
    horizon = 60
    T = horizon / 252.0

    sched = scenario["build"](sigma, horizon)

    paths = run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu, horizon_days=horizon,
        n_paths=100_000, vol_schedule=sched, seed=42,
    )
    mc = analyze_joint_conditional(
        paths, S0, dip, rally, horizon, sigma=sigma, vol_schedule=sched,
    )
    sigma_eq = float(np.sqrt(np.mean(sched ** 2)))
    cf_dip = closed_touch_down(S0, dip, T, mu, sigma_eq)
    cf_rally = closed_touch_up(S0, rally, T, mu, sigma_eq)

    gap_dip = abs(mc["p_dip_touched_any"] - cf_dip) * 100
    gap_rally = abs(mc["p_rally_touched_any"] - cf_rally) * 100
    assert gap_dip < 2.0, (
        f"[{scenario['name']}] dip marginal gap {gap_dip:.2f}pp > 2.0pp — "
        f"MC={mc['p_dip_touched_any']*100:.2f}% CF={cf_dip*100:.2f}%"
    )
    assert gap_rally < 2.0, (
        f"[{scenario['name']}] rally marginal gap {gap_rally:.2f}pp > 2.0pp — "
        f"MC={mc['p_rally_touched_any']*100:.2f}% CF={cf_rally*100:.2f}%"
    )

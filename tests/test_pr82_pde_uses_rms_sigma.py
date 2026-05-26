"""Tests for PR #82 — audit finding #14 (unmasked by PR #76).

Pre-PR-#76, `vol_schedule` was indexed by calendar offset; events past
trading-day-60 fell out of bounds and were silently dropped. The MC was
effectively running near-constant sigma, so the cross-check vs PDE
(constant sigma) agreed within tolerance.

PR #76 correctly indexes the schedule by trading day, so every in-horizon
earnings / macro event lands inside the simulation window and elevates
vol. The MC's touch probabilities now diverge materially from PDE's
constant-sigma probabilities — tripping sacred #16's refusal threshold
on stable MID/HIGH names with quarterly earnings in horizon (6/26
tickers REFUSED-METHOD on the first post-PR-#76 cycle).

Fix: pass `vol_schedule` to `three_method_cross_check`. It computes the
RMS-equivalent constant sigma — variance-preserving — and feeds that to
PDE / closed-form. Restores apples-to-apples comparison.
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
    """Build a fake mc_result dict matching the keys cross-check reads."""
    return {
        "p_round_trip": p_round_trip,
        "p_bag_hold": p_bag_hold,
        "p_no_trade_rally_first": p_no_trade_rally_first,
        "p_neither": p_neither,
        "p_dip_touched_any": p_round_trip + p_bag_hold,
        "p_rally_touched_any": p_round_trip + p_no_trade_rally_first,
    }


def test_backwards_compat_no_vol_schedule_passes_through():
    """When called without vol_schedule (legacy callers + the existing
    test suite), `sigma_eq = sigma`. Output unchanged."""
    from src.math_utils import three_method_cross_check
    chk_old = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.05, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
    )
    chk_explicit_none = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.05, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
        vol_schedule=None,
    )
    assert chk_old["tolerances"]["sigma_eq_pde"] == 0.30
    assert chk_old == chk_explicit_none


def test_flat_schedule_equals_constant_sigma():
    """vol_schedule of all 1.0 → RMS = 1.0 → sigma_eq = sigma. PDE
    output identical to constant-sigma call (modulo float)."""
    from src.math_utils import three_method_cross_check
    flat = np.ones(60)
    chk_with = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.05, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
        vol_schedule=flat,
    )
    chk_without = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.05, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
    )
    assert chk_with["tolerances"]["sigma_eq_pde"] == pytest.approx(0.30)
    # PDE / closed-form values identical (RMS of all-1s is 1.0).
    for row_w, row_wo in zip(chk_with["table"], chk_without["table"]):
        assert row_w[1] == pytest.approx(row_wo[1])  # MC unchanged
        assert row_w[2] == pytest.approx(row_wo[2])  # PDE side


def test_earnings_spike_raises_sigma_eq():
    """vol_schedule with a 3× spike around earnings → sigma_eq > sigma.
    The cross-check now hands PDE the elevated sigma matching what the
    MC actually saw."""
    from src.math_utils import three_method_cross_check
    sched = np.ones(60)
    sched[20:23] = 3.0  # 3-day earnings vol spike
    chk = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.05, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
        vol_schedule=sched,
    )
    sigma_eq = chk["tolerances"]["sigma_eq_pde"]
    assert sigma_eq > 0.30
    # RMS check: sqrt(((57*1 + 3*9))/60) ≈ sqrt(1.4) ≈ 1.183
    expected = 0.30 * float(np.sqrt(np.mean(sched ** 2)))
    assert sigma_eq == pytest.approx(expected)


def test_real_scenario_reduces_mc_pde_divergence():
    """End-to-end: simulate a real MC vs PDE comparison. With a
    realistic earnings spike schedule, the constant-sigma PDE under-
    estimates touch probabilities relative to MC. Passing vol_schedule
    to the cross-check should reduce the diff to within tolerance."""
    from src.math_utils import (
        analyze_joint_conditional,
        run_mc_joint_conditional,
        three_method_cross_check,
    )
    S0, sigma, mu, horizon = 100.0, 0.35, 0.10, 60

    # Build a schedule with earnings at day 21 (3-day window, 1.5× spike).
    from src.config import VOL_SCHEDULE_MULTIPLIERS  # noqa
    sched = np.ones(horizon)
    sched[20:24] = 1.5  # mild earnings region

    paths = run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu,
        horizon_days=horizon, n_paths=20000, seed=11,
        vol_schedule=sched,
    )
    mc_res = analyze_joint_conditional(
        paths, S0, dip_price=92.0, rally_price=110.0,
        horizon_days=horizon, sigma=sigma, vol_schedule=sched,
    )

    # Without vol_schedule → PDE on plain sigma → larger divergence.
    chk_unaligned = three_method_cross_check(
        S0, sigma, mu, horizon, 92.0, 110.0, mc_res,
    )
    # With vol_schedule → PDE on sigma_eq → tighter agreement.
    chk_aligned = three_method_cross_check(
        S0, sigma, mu, horizon, 92.0, 110.0, mc_res,
        vol_schedule=sched,
    )

    # Sum of absolute pp diffs across all four cross-check rows.
    def _total_div(chk):
        return sum(row[3] for row in chk["table"])

    assert _total_div(chk_aligned) < _total_div(chk_unaligned), (
        f"PR #82 expected to tighten cross-check agreement. "
        f"unaligned={_total_div(chk_unaligned):.2f}pp, "
        f"aligned={_total_div(chk_aligned):.2f}pp"
    )


def test_sigma_eq_surfaced_in_report_dict():
    """Operator needs to see in the report that sigma_eq was used —
    otherwise the cross-check appears to be using `sigma_used` directly,
    misleading."""
    from src.math_utils import three_method_cross_check
    sched = np.ones(60)
    sched[10] = 2.0
    chk = three_method_cross_check(
        S0=100.0, sigma=0.40, mu=0.0, horizon_days=60,
        dip_price=95.0, rally_price=110.0, mc_result=_mk_mc(),
        vol_schedule=sched,
    )
    assert "sigma_eq_pde" in chk["tolerances"]
    assert chk["tolerances"]["sigma_eq_pde"] != chk["tolerances"]["sigma_used"]

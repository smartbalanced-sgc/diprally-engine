"""Tests for PR #83 — adaptive first-passage tolerance.

Updated post-PR-#84 (sigma_eq formula fix). vol_schedule is in
ABSOLUTE vol per day units (`base_sigma × multipliers`), not
dimensionless multipliers.

Cycle 2 evidence: 5/26 still REFUSED-METHOD even after PR #82, AND
GHM regressed from BUY → REFUSED-METHOD because PR #82's incorrect
sigma_eq inflated certain probabilities into new disagreement. After
PR #84 fixed the formula, the cross-check agrees on marginals to
within ~1pp. PR #83's first-passage tolerance widening is now a
small (but still real) safety belt for first-passage ORDERING when
schedules have legitimate spikes.

heterogeneity := sigma_eq / sigma - 1
fp_widening   := 1 + 2 × max(0, heterogeneity)
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


def test_flat_absolute_schedule_no_widening():
    """vol_schedule at flat absolute σ → heterogeneity 0 → no widening."""
    from src.math_utils import three_method_cross_check
    sigma = 0.30
    flat = np.full(60, sigma)
    chk = three_method_cross_check(
        S0=100.0, sigma=sigma, mu=0.0, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
        vol_schedule=flat,
    )
    assert chk["tolerances"]["fp_widening_factor"] == pytest.approx(1.0)
    assert chk["tolerances"]["schedule_heterogeneity"] == pytest.approx(0.0)


def test_no_schedule_no_widening():
    """vol_schedule=None (legacy path) → no widening, backwards-compat."""
    from src.math_utils import three_method_cross_check
    chk = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.0, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
    )
    assert chk["tolerances"]["fp_widening_factor"] == pytest.approx(1.0)


def test_earnings_spike_widens_first_passage_only():
    """Earnings spike → sigma_eq > sigma → fp tolerance widens.
    Marginal tolerance UNCHANGED (PR #82+#84 made marginals agree)."""
    from src.math_utils import three_method_cross_check
    sigma = 0.30
    sched = np.full(60, sigma)
    sched[20:23] = sigma * 3.0   # 3-day 3× spike
    chk = three_method_cross_check(
        S0=100.0, sigma=sigma, mu=0.0, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
        vol_schedule=sched,
    )
    widening = chk["tolerances"]["fp_widening_factor"]
    assert widening > 1.0
    chk_flat = three_method_cross_check(
        S0=100.0, sigma=sigma, mu=0.0, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
    )
    assert chk["tolerances"]["first_passage_pp"] > chk_flat["tolerances"]["first_passage_pp"]
    assert chk["tolerances"]["refuse_first_passage_pp"] > chk_flat["tolerances"]["refuse_first_passage_pp"]
    assert chk["tolerances"]["marginal_pp"] == pytest.approx(chk_flat["tolerances"]["marginal_pp"])
    assert chk["tolerances"]["refuse_marginal_pp"] == pytest.approx(chk_flat["tolerances"]["refuse_marginal_pp"])


def test_widening_formula_matches_documented():
    """widening = 1 + 2 × (sigma_eq / sigma - 1) where sigma_eq is
    sqrt(mean(vol_schedule**2)) — post-PR-#84 formula."""
    from src.math_utils import three_method_cross_check
    sigma = 0.40
    sched = np.full(60, sigma)
    sched[10:14] = sigma * 2.0   # 4-day 2× spike
    sigma_eq = float(np.sqrt(np.mean(sched ** 2)))
    expected_heterogeneity = sigma_eq / sigma - 1.0
    expected_widening = 1.0 + 2.0 * expected_heterogeneity
    chk = three_method_cross_check(
        S0=100.0, sigma=sigma, mu=0.0, horizon_days=60,
        dip_price=95.0, rally_price=110.0, mc_result=_mk_mc(),
        vol_schedule=sched,
    )
    assert chk["tolerances"]["fp_widening_factor"] == pytest.approx(expected_widening)
    assert chk["tolerances"]["schedule_heterogeneity"] == pytest.approx(expected_heterogeneity)


def test_marginal_disagreement_still_refuses_even_with_spike():
    """PR #83 widens ONLY first-passage tolerance. A genuine marginal
    disagreement must still refuse even when schedule has spikes."""
    from src.math_utils import three_method_cross_check
    sigma = 0.30
    mc = {
        "p_round_trip": 0.30,
        "p_bag_hold": 0.10,
        "p_no_trade_rally_first": 0.20,
        "p_neither": 0.40,
        "p_dip_touched_any": 0.99,   # absurdly high vs closed-form
        "p_rally_touched_any": 0.99,
    }
    sched = np.full(60, sigma)
    sched[20:25] = sigma * 3.0
    chk = three_method_cross_check(
        S0=100.0, sigma=sigma, mu=0.0, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=mc,
        vol_schedule=sched,
    )
    assert chk["refused"]
    refusal_text = " | ".join(chk["refusals"])
    assert "touch dip" in refusal_text or "touch rally" in refusal_text


def test_extreme_schedule_documented():
    """A pathological vol_schedule at 5× the baseline σ yields large
    heterogeneity and widening. No explicit cap; this test just
    documents the behavior so future changes notice."""
    from src.math_utils import three_method_cross_check
    sigma = 0.30
    sched = np.full(60, sigma * 5.0)   # entire path at 5× baseline
    chk = three_method_cross_check(
        S0=100.0, sigma=sigma, mu=0.0, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
        vol_schedule=sched,
    )
    # sigma_eq = 5σ, heterogeneity = 4, widening = 9.
    assert chk["tolerances"]["schedule_heterogeneity"] == pytest.approx(4.0)
    assert chk["tolerances"]["fp_widening_factor"] == pytest.approx(9.0)

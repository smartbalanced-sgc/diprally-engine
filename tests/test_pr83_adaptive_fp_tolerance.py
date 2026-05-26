"""Tests for PR #83 — adaptive first-passage tolerance.

Cycle 2 follow-up to PR #82. PR #82 made the cross-check's MARGINAL
probabilities (touch-ever) agree by feeding PDE/closed-form an RMS-
equivalent constant sigma. But FIRST-PASSAGE probabilities (P(dip
first), P(rally first)) depend on WHEN the vol concentrates inside
the path, not just its average level. For names with earnings early
in the horizon, MC sees dip touched DURING the spike (early first-
passage); PDE with constant sigma_eq sees touches evenly distributed
(uniform first-passage). Same marginal totals, different ordering.

Cycle 2 evidence: 5/26 still REFUSED-METHOD even after PR #82, AND
GHM regressed from BUY → REFUSED-METHOD because PDE's inflated
sigma_eq shifted its first-passage probabilities into new
disagreement with MC.

Fix: widen ONLY the first_passage tolerance proportional to
schedule heterogeneity. Marginal tolerance unchanged.
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


def test_flat_schedule_no_widening():
    """Flat vol_schedule → heterogeneity=0 → fp tolerance unchanged."""
    from src.math_utils import three_method_cross_check
    flat = np.ones(60)
    chk = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.0, horizon_days=60,
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
    """3-day 3× spike → sigma_eq ~1.18σ → heterogeneity ~0.18 →
    fp tolerance widens ~36%. Marginal tolerance UNCHANGED."""
    from src.math_utils import three_method_cross_check
    sched = np.ones(60)
    sched[20:23] = 3.0
    chk = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.0, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
        vol_schedule=sched,
    )
    widening = chk["tolerances"]["fp_widening_factor"]
    assert widening > 1.0
    # Compare against the no-schedule baseline.
    chk_flat = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.0, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
    )
    # fp tolerance was widened.
    assert chk["tolerances"]["first_passage_pp"] > chk_flat["tolerances"]["first_passage_pp"]
    assert chk["tolerances"]["refuse_first_passage_pp"] > chk_flat["tolerances"]["refuse_first_passage_pp"]
    # Marginal tolerance UNCHANGED.
    assert chk["tolerances"]["marginal_pp"] == pytest.approx(chk_flat["tolerances"]["marginal_pp"])
    assert chk["tolerances"]["refuse_marginal_pp"] == pytest.approx(chk_flat["tolerances"]["refuse_marginal_pp"])


def test_widening_formula_matches_documented():
    """widening = 1 + 2 * (sigma_eq / sigma - 1). Verify the formula
    against a hand-computed example."""
    from src.math_utils import three_method_cross_check
    sched = np.ones(60)
    sched[10:14] = 2.0  # 4-day 2× spike
    rms = float(np.sqrt(np.mean(sched ** 2)))
    expected_heterogeneity = rms - 1.0  # since sigma_eq/sigma = rms
    expected_widening = 1.0 + 2.0 * expected_heterogeneity
    chk = three_method_cross_check(
        S0=100.0, sigma=0.40, mu=0.0, horizon_days=60,
        dip_price=95.0, rally_price=110.0, mc_result=_mk_mc(),
        vol_schedule=sched,
    )
    assert chk["tolerances"]["fp_widening_factor"] == pytest.approx(expected_widening)
    assert chk["tolerances"]["schedule_heterogeneity"] == pytest.approx(expected_heterogeneity)


def test_borderline_first_passage_disagreement_passes_with_widening():
    """A MC vs PDE first-passage diff that USED to refuse (pre-#83) now
    passes when the schedule is heterogeneous enough to justify it.

    Setup: contrive an mc_result whose first-passage breakdown differs
    from PDE's by ~5.5pp — just above the un-widened refuse threshold
    (5.4pp for sigma=0.30 MID). With a heavy earnings spike, the widened
    threshold (~7.4pp) accommodates it.
    """
    from src.math_utils import three_method_cross_check
    # Build MC result with biased dip_first.
    mc = _mk_mc(p_round_trip=0.40, p_bag_hold=0.30,
                p_no_trade_rally_first=0.10, p_neither=0.20)
    # p_dip_first_mc = 0.70 (≈70%)

    # Without schedule, PDE for these prices/sigma will give a different
    # p_dip_first. Whether refusal triggers depends on the actual numbers.
    # Just verify that the FACT of widening allows MORE leeway on first-
    # passage when schedule has spikes — i.e. refuse_fp grows.
    sched = np.ones(60)
    sched[25:30] = 3.0  # 5-day 3× spike
    chk_with = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.0, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=mc,
        vol_schedule=sched,
    )
    chk_flat = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.0, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=mc,
    )
    # First-passage refusal threshold is strictly wider with the
    # earnings-spike schedule.
    assert chk_with["tolerances"]["refuse_first_passage_pp"] > \
           chk_flat["tolerances"]["refuse_first_passage_pp"] * 1.1


def test_marginal_disagreement_still_refuses_even_with_spike():
    """PR #83 widens ONLY first-passage tolerance. A genuine marginal
    disagreement (touch-ever probabilities differ by > refuse_marg)
    must still refuse even when schedule has spikes."""
    from src.math_utils import three_method_cross_check
    # Construct mc_result whose marginal touch probabilities differ
    # widely from what PDE/closed-form will produce — engineered to
    # exceed the marginal refusal threshold (3.6pp for sigma=0.30 MID).
    # Force p_dip_touched_any much higher than closed-form expects.
    mc = {
        "p_round_trip": 0.30,
        "p_bag_hold": 0.10,
        "p_no_trade_rally_first": 0.20,
        "p_neither": 0.40,
        "p_dip_touched_any": 0.99,  # absurdly high vs closed-form
        "p_rally_touched_any": 0.99,
    }
    sched = np.ones(60)
    sched[20:25] = 3.0
    chk = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.0, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=mc,
        vol_schedule=sched,
    )
    # Refusal should fire on the MARGINAL leg (not first-passage).
    assert chk["refused"]
    # And the refusal message must cite touch dip / touch rally, not
    # first-passage.
    refusal_text = " | ".join(chk["refusals"])
    assert "touch dip" in refusal_text or "touch rally" in refusal_text


def test_extreme_schedule_heterogeneity_capped_implicitly():
    """A pathological vol_schedule (entirely 5× spike for 60 bars)
    yields RMS=5, heterogeneity=4, widening=9. That's huge — but the
    formula is bounded only by the schedule values themselves. No
    explicit cap; this test just documents the behavior so future
    changes notice."""
    from src.math_utils import three_method_cross_check
    sched = np.full(60, 5.0)
    chk = three_method_cross_check(
        S0=100.0, sigma=0.30, mu=0.0, horizon_days=60,
        dip_price=92.0, rally_price=108.0, mc_result=_mk_mc(),
        vol_schedule=sched,
    )
    # heterogeneity = 5-1 = 4; widening = 1 + 2*4 = 9.
    assert chk["tolerances"]["schedule_heterogeneity"] == pytest.approx(4.0)
    assert chk["tolerances"]["fp_widening_factor"] == pytest.approx(9.0)

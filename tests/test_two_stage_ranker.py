"""Tests for the two-stage [EV ≥ σ-class hurdle → max P(profitable)]
grid ranker that replaced max-EV ranking.

Motivation: objective-function audit (2026-05-30) found 0/10 top-10
overlap between EV ranking and P(round-trip) ranking on synth setups
across MID/HIGH/EXTREME × bull/neutral/bear μ. Max-EV always landed on
the rally grid maximum (jackpot setups, 10-21% hit rate); the mission
("lock in gains via defensible round-trips inside 20 trading days") is
hit-rate language. EV stays the FLOOR via sacred #13; P(profitable) is
the rank among setups that clear the floor.

Locks:
1. compute_dual_ev exposes p_profitable_{wait,direct} and the
   payoff-std fields the ranker reads.
2. scan_dip_rally_grid prefers the higher-P(profitable) pair among
   hurdle-clearing qualified pairs, NOT the higher-EV pair.
3. Branch selection within a pair prefers higher P(profitable) among
   branches that clear hurdle (not max-EV).
4. Legacy fallback: when no qualified pair clears the EV hurdle, the
   ranker falls back to max-EV among qualified — so REFUSED-EV
   diagnostics downstream still fire on a sensible `best` pointer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_compute_dual_ev_exposes_mission_aligned_fields():
    """The new ranker reads p_profitable_{wait,direct} and the payoff-
    std fields off compute_dual_ev's return. Lock the keys exist and
    are plausibly bounded."""
    from src.math_utils import compute_dual_ev
    S0 = 100.0
    rng = np.random.default_rng(0)
    paths = S0 * np.exp(np.cumsum(
        rng.normal(0.001, 0.04, size=(5000, 20)), axis=1,
    ))
    out = compute_dual_ev(
        paths, S0, dip_price=95.0, rally_price=110.0,
        friction_per_share=0.35, patience_window_td=10, swing_stop_pct=0.10,
    )
    for k in ("p_profitable_wait", "p_profitable_direct",
              "payoff_std_wait_pct_of_dip", "payoff_std_direct_pct_of_spot"):
        assert k in out, f"compute_dual_ev missing field {k!r}"
        assert 0.0 <= out[k] <= 5.0
    # p_profitable_wait ≤ p_dip_filled (entry constraint).
    assert out["p_profitable_wait"] <= out["p_dip_filled"] + 1e-9
    # p_profitable_direct ≤ p_rally_hit (success implies rally hit).
    assert out["p_profitable_direct"] <= out["p_rally_hit"] + 1e-9


def test_two_stage_ranker_diverges_from_max_ev_on_bullish_extreme():
    """The decisive test: on a bullish EXTREME-σ setup (the MU/MRAM
    regime that was producing 0-BUY pre-audit), the two-stage ranker
    must pick a DIFFERENT (dip, rally) pair than the old max-EV ranker.

    Specifically: max-EV pinned the rally grid maximum (jackpot, low
    hit rate); two-stage should pick a closer rally (higher hit rate)
    while still clearing the σ-class EV hurdle. If the ranker ever
    silently regresses to max-EV behavior, this test catches it.
    """
    from src.engine import scan_dip_rally_grid
    from src.math_utils import run_mc_joint_conditional
    from src.config import SIGMA_CLASSES
    S0, sigma, mu = 12.0, 1.30, 0.30
    paths = run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu, horizon_days=20,
        n_paths=30_000, distribution="student_t", df=5.0, seed=42,
    )
    best, candidates, met = scan_dip_rally_grid(
        S0=S0, sigma=sigma, mu=mu, horizon_days=20, paths=paths,
        conviction_dip=0.55, conviction_rally_cond=0.55,
        sigma_class="EXTREME",
    )
    assert met, "synth setup should produce qualified pairs"
    assert best is not None

    # Hurdle floor still respected (sacred #13).
    hurdle_pct = SIGMA_CLASSES["EXTREME"].ev_hurdle_bps / 10000.0
    assert best.ev_pct_of_dip >= hurdle_pct, (
        f"best ev_pct_of_dip {best.ev_pct_of_dip*10000:.1f}bps "
        f"under hurdle {hurdle_pct*10000:.1f}bps"
    )

    # The decisive expectation: the two-stage ranker did NOT pick the
    # max-EV pair. If it did, the audit's structural finding regressed.
    qualified = [
        c for c in candidates
        if c.p_dip_touched >= 0.55 and c.p_rally_given_dip >= 0.55
    ]
    qualified_above_hurdle = [
        c for c in qualified if c.ev_pct_of_dip >= hurdle_pct
    ]
    assert qualified_above_hurdle, (
        "synth setup should have pairs above hurdle"
    )
    ev_top = max(qualified_above_hurdle, key=lambda c: c.net_ev_per_share)
    # Same pair would mean ranker collapsed back to max-EV.
    same_pair = (
        abs(best.dip_price - ev_top.dip_price) < 1e-6
        and abs(best.rally_price - ev_top.rally_price) < 1e-6
    )
    # Either it picked a different pair, OR the same pair already has
    # the max P(profitable) among hurdle-clearers (rare ties allowed).
    if same_pair:
        max_p_prof = max(c.p_profitable for c in qualified_above_hurdle)
        assert best.p_profitable >= max_p_prof - 1e-9, (
            "ranker fell back to max-EV when a higher-P(profitable) "
            "pair existed"
        )
    # P(profitable) ranker must beat or tie the EV-top pick on the
    # mission-aligned metric.
    assert best.p_profitable >= ev_top.p_profitable - 1e-9, (
        f"two-stage best p_profitable {best.p_profitable:.3f} below "
        f"EV-top p_profitable {ev_top.p_profitable:.3f}"
    )


def test_legacy_fallback_when_no_pair_clears_hurdle():
    """If no qualified pair clears the EV hurdle, ranker must fall
    back to max-EV among qualified so the downstream REFUSED-EV gate
    has a sensible `best` to report on. Without this fallback the
    engine would have `best=None` and emit a different (wrong)
    refusal state."""
    from src.engine import scan_dip_rally_grid
    from src.math_utils import run_mc_joint_conditional
    # Bearish drift + EXTREME σ → some pairs qualify on conviction
    # but the EV distribution flattens. Force a tight hurdle by using
    # a known-marginal setup.
    S0, sigma, mu = 100.0, 1.30, -0.50
    paths = run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu, horizon_days=20,
        n_paths=20_000, distribution="student_t", df=5.0, seed=42,
    )
    best, candidates, met = scan_dip_rally_grid(
        S0=S0, sigma=sigma, mu=mu, horizon_days=20, paths=paths,
        conviction_dip=0.55, conviction_rally_cond=0.55,
        sigma_class="EXTREME",
    )
    # No guarantee about met_threshold_strict in this regime, but
    # `best` must not be None and the engine must not raise.
    assert best is not None


def test_branch_select_prefers_higher_hit_rate_when_both_clear_hurdle():
    """When both WAIT and DIRECT branches clear the EV hurdle on a
    given pair, the branch with higher P(profitable) wins. This
    locks the within-pair selection rule that mirrors the grid-level
    two-stage ranker. We can't easily construct a single pair where
    this fires deterministically across regimes, so we assert the
    weaker invariant: the chosen branch's p_profitable is at least
    the OTHER branch's p_profitable, whenever both cleared the
    hurdle and the pair is the `best`."""
    from src.engine import scan_dip_rally_grid
    from src.math_utils import run_mc_joint_conditional, compute_dual_ev
    from src.math_utils import precompute_first_touch_days
    from src.config import SIGMA_CLASSES, PATIENCE_WINDOW_TD, MIN_DIP_PROBABILITY
    S0, sigma, mu = 230.0, 0.90, 0.25
    paths = run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu, horizon_days=20,
        n_paths=20_000, distribution="student_t", df=5.0, seed=42,
    )
    best, _, met = scan_dip_rally_grid(
        S0=S0, sigma=sigma, mu=mu, horizon_days=20, paths=paths,
        conviction_dip=0.60, conviction_rally_cond=0.65,
        sigma_class="HIGH",
    )
    assert best is not None and met

    cls = SIGMA_CLASSES["HIGH"]
    hurdle_pct = cls.ev_hurdle_bps / 10000.0
    fric_bps = cls.friction_bps_round_trip
    fric_wait = (best.dip_price + best.rally_price) / 2.0 * fric_bps / 10000
    fric_direct = (S0 + best.rally_price) / 2.0 * fric_bps / 10000
    dip_first = precompute_first_touch_days(
        paths, S0, np.array([best.dip_price]), sigma, None, "down", seed=42,
    )[:, 0]
    rally_first = precompute_first_touch_days(
        paths, S0, np.array([best.rally_price]), sigma, None, "up", seed=43,
    )[:, 0]
    dual = compute_dual_ev(
        paths, S0, best.dip_price, best.rally_price, fric_wait,
        dip_first_days=dip_first, rally_first_days=rally_first,
        patience_window_td=PATIENCE_WINDOW_TD,
        swing_stop_pct=cls.swing_stop_pct,
        friction_per_share_direct=fric_direct,
    )
    wait_eligible = dual["p_dip_filled"] >= MIN_DIP_PROBABILITY
    wait_clears = wait_eligible and dual["ev_wait_pct_of_dip"] >= hurdle_pct
    direct_clears = dual["ev_direct_pct_of_spot"] >= hurdle_pct
    if wait_clears and direct_clears:
        if best.verdict_subtype == "WAIT-FOR-DIP":
            assert dual["p_profitable_wait"] >= dual["p_profitable_direct"] - 1e-9
        else:
            assert dual["p_profitable_direct"] >= dual["p_profitable_wait"] - 1e-9

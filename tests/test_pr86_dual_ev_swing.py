"""Tests for PR #86 — dual-EV swing engine re-architecture.

Retires the strict-round-trip orthodoxy. Engine now reports both
DIRECT entry (enter at spot, exit at rally) and WAIT-FOR-DIP entry
EVs, picks the higher one. Verdict subtype indicates the winning
strategy. Operator's true objective is "capture the rally"; the dip
is an optional entry-price optimization.

Also locks: horizon 60→20, min_dip_probability gate, σ-class grids
retuned for 20d, live-spot append to history_df (so mom_30d / RSI
see today's price).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# compute_dual_ev — math primitive correctness
# =============================================================================

def test_compute_dual_ev_flat_returns_zero_drift():
    """Zero drift, zero vol → all paths stay at spot. Neither dip nor
    rally touched. EV_direct = EV_wait = -friction (no fill returns 0
    payoff for wait; direct holds to T at spot = 0 gain - friction)."""
    from src.math_utils import compute_dual_ev
    S0 = 100.0
    # Build trivial 'paths' all equal to spot at every step
    paths = np.full((1000, 20), S0)
    out = compute_dual_ev(
        paths, S0, dip_price=95.0, rally_price=105.0,
        friction_per_share=0.35,  # 35bps of 100
    )
    # Neither barrier touched
    assert out["p_dip_filled"] == 0.0
    assert out["p_rally_hit"] == 0.0
    # Direct: terminal = spot, payoff = -friction
    assert out["ev_direct_per_share"] == pytest.approx(-0.35)
    # Wait: no fill → 0 payoff for all paths
    assert out["ev_wait_per_share"] == pytest.approx(0.0)


def test_compute_dual_ev_direct_wins_when_rally_likely():
    """Rally easily touched but dip rarely touched → DIRECT entry wins.
    The strategy that doesn't require waiting for a low-probability dip
    has higher unconditional EV."""
    from src.math_utils import compute_dual_ev, run_mc_joint_conditional
    S0 = 100.0
    sigma = 0.30
    mu = 0.20  # strong upward drift — dip unlikely, rally likely
    paths = run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu, horizon_days=20,
        n_paths=20_000, seed=42,
    )
    out = compute_dual_ev(
        paths, S0, dip_price=92.0, rally_price=104.0,
        friction_per_share=0.35,
    )
    # Direct should win — easy rally, low dip prob means wait often doesn't fill
    assert out["ev_direct_per_share"] > out["ev_wait_per_share"]


def test_compute_dual_ev_no_fill_path_contributes_zero():
    """Wait strategy: paths where dip is never touched MUST contribute
    zero payoff (no entry, no PnL), NOT negative friction."""
    from src.math_utils import compute_dual_ev
    # Construct paths: half touch dip, half don't.
    n_paths = 2000
    n_days = 20
    paths = np.full((n_paths, n_days), 100.0)
    # First 1000 paths: dip to 90 on day 5, recover
    paths[:1000, 5] = 90.0
    paths[:1000, 6:] = 100.0
    # Last 1000: never dip, stay flat

    out = compute_dual_ev(
        paths, S0=100.0, dip_price=92.0, rally_price=110.0,
        friction_per_share=1.0,
    )
    # p_dip_filled should be exactly 0.5
    assert out["p_dip_filled"] == pytest.approx(0.5)
    # Wait EV: 1000 paths fill (terminal=100, dip=92, payoff=100-92-1=+7),
    # 1000 paths don't fill (payoff=0). Mean = (1000*7 + 1000*0) / 2000 = 3.5.
    assert out["ev_wait_per_share"] == pytest.approx(3.5)


def test_compute_dual_ev_round_trip_payoff_uses_rally_minus_dip():
    """When path touches dip then touches rally → payoff = rally - dip
    (not rally - spot)."""
    from src.math_utils import compute_dual_ev
    n_paths = 1000
    n_days = 20
    paths = np.full((n_paths, n_days), 100.0)
    # All paths dip to 92 on day 3, rally to 110 on day 10
    paths[:, 3] = 92.0
    paths[:, 4:10] = 95.0
    paths[:, 10] = 110.0
    paths[:, 11:] = 108.0
    out = compute_dual_ev(
        paths, S0=100.0, dip_price=92.0, rally_price=110.0,
        friction_per_share=0.5,
    )
    # Wait fills every path; payoff = 110 - 92 - 0.5 = 17.5
    assert out["p_dip_filled"] == 1.0
    assert out["ev_wait_per_share"] == pytest.approx(17.5)
    # Direct: rally touched all paths; payoff = 110 - 100 - 0.5 = 9.5
    assert out["p_rally_hit"] == 1.0
    assert out["ev_direct_per_share"] == pytest.approx(9.5)


# =============================================================================
# Engine integration — scan_dip_rally_grid picks max(EV_direct, EV_wait)
# =============================================================================

def test_jc_result_carries_dual_ev_fields():
    """JointConditionalResult dataclass has the new dual-EV fields."""
    from src.engine import JointConditionalResult
    r = JointConditionalResult(
        dip_price=92.0, rally_price=110.0,
        p_dip_touched=0.5, p_rally_given_dip=0.3,
        p_round_trip=0.15, p_bag_hold=0.35,
        p_no_trade_rally_first=0.10, p_neither=0.40,
        expected_days_to_dip=5.0, expected_days_dip_to_rally=10.0,
        expected_gain_per_share=17.5, expected_bag_hold_loss=8.0,
        net_ev_per_share=2.5, ev_pct_of_dip=0.027,
    )
    # Defaults for backward compat with non-PR-#86 callers
    assert r.ev_direct_per_share == 0.0
    assert r.ev_wait_per_share == 0.0
    assert r.verdict_subtype == "DIRECT"
    assert r.p_dip_filled == 0.0
    assert r.p_rally_hit == 0.0


def test_min_dip_probability_loaded_from_config():
    """PR #86 config knob accessible and defaulted to 0.30."""
    from src.config import MIN_DIP_PROBABILITY
    assert 0.0 <= MIN_DIP_PROBABILITY <= 1.0
    assert MIN_DIP_PROBABILITY == 0.30   # locked value


def test_horizon_default_is_20():
    """PR #86 default horizon — was 60."""
    from src.config import DEFAULT_HORIZON_DAYS
    assert DEFAULT_HORIZON_DAYS == 20


def test_sigma_class_grids_scaled_for_20d():
    """σ-class grids rescaled by √(20/60) ≈ 0.577 — dip_max_depth and
    rally_max_reach should reflect this."""
    from src.config import SIGMA_CLASSES
    # MID (was -20/+30 at 60d) → ~-12/+18 at 20d
    mid = SIGMA_CLASSES["MID"].grid
    assert 0.10 <= mid.dip_max_depth_pct <= 0.15
    assert 0.15 <= mid.rally_max_reach_pct <= 0.22
    # HIGH (was -35/+50 at 60d) → ~-20/+29
    high = SIGMA_CLASSES["HIGH"].grid
    assert 0.18 <= high.dip_max_depth_pct <= 0.24
    assert 0.25 <= high.rally_max_reach_pct <= 0.33
    # EXTREME (was -50/+60 at 60d) → ~-29/+35
    ext = SIGMA_CLASSES["EXTREME"].grid
    assert 0.25 <= ext.dip_max_depth_pct <= 0.33
    assert 0.30 <= ext.rally_max_reach_pct <= 0.40


# =============================================================================
# CSV schema for PR #86 dual-EV
# =============================================================================

def test_csv_columns_include_dual_ev_fields():
    from src.engine import CSV_COLUMNS
    for col in ("verdict_subtype", "ev_direct_bps", "ev_wait_bps",
                 "p_dip_filled", "p_rally_hit"):
        assert col in CSV_COLUMNS, f"missing PR #86 CSV column: {col}"


# =============================================================================
# Orchestrator TickerDecision carries dual-EV fields
# =============================================================================

def test_ticker_decision_dataclass_has_dual_ev_fields():
    from src.orchestrator import TickerDecision
    d = TickerDecision(
        ticker="AMAT", sigma_class="MID", tier="T2",
        ambiguity=0.10, qualifies_for_t2_plus=True,
        spot=455.0, dip_target=440.0, rally_target=480.0,
        p_round_trip=0.30, ev_bps_of_dip=120.0,
        verdict="BUY", status_note="",
    )
    # Defaults present
    assert d.verdict_subtype == "DIRECT"
    assert d.ev_direct_bps is None
    assert d.ev_wait_bps is None
    assert d.p_dip_filled is None
    assert d.p_rally_hit is None
    assert d.expected_rally_date is None
    assert d.expected_dip_date is None

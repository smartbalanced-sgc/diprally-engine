"""Defect D — swing patience window in compute_dual_ev.

The legacy dual-EV model held a non-rallying position to the horizon-end
terminal and credited a rally arriving any time after entry. A real swing
trader has finite patience: if the rally doesn't materialise within
`patience_window_td` trading days of entry, the thesis is broken and the
position is time-stopped at market. This:

  - stops crediting round-trip wins for rallies a patient trader would never
    wait for (rallies arriving > window after entry), and
  - marks the no-rally exit to the entry+window price, not the horizon-end
    terminal (which let losers 'recover' and overstated EV on positive-drift
    momentum names).

patience_window_td=None preserves the legacy behaviour (backward compat for
existing callers/tests). The engine passes config PATIENCE_WINDOW_TD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import PATIENCE_WINDOW_TD
from src.math_utils import compute_dual_ev


def test_config_exposes_patience_window():
    assert isinstance(PATIENCE_WINDOW_TD, int)
    assert PATIENCE_WINDOW_TD >= 1


# ---------------------------------------------------------------------------
# Hand-built deterministic paths isolate the window mechanic exactly.
# Columns are trading-day offsets; S0 is the entry-before-day-0 spot.
# ---------------------------------------------------------------------------

def _late_rally_paths():
    """One path: dips early (day 1), rallies LATE (day 30). A patient trader
    with a short window would have time-stopped before the rally."""
    S0 = 100.0
    n_days = 40
    path = np.full(n_days, 90.0)      # sits at 90 (below dip 95, above... )
    path[0] = 90.0                    # day0: dip touched (<=95)
    path[5:] = 92.0                   # drifts to 92, still no rally
    path[30:] = 130.0                 # rallies to 130 (>= rally 120) at day 30
    return S0, path.reshape(1, n_days)


def test_late_rally_credited_without_window_dropped_with_short_window():
    """FAIL-BEFORE / PASS-AFTER core: the same late-rally path is a
    round-trip win under the legacy model (no window) but a time-stopped
    bag-hold under a short patience window."""
    S0, paths = _late_rally_paths()
    dip, rally, fr = 95.0, 120.0, 0.0

    legacy = compute_dual_ev(paths, S0, dip, rally, fr)  # patience=None
    assert legacy["p_round_trip_strict"] == pytest.approx(1.0)
    # Wait EV credits the full round trip: rally - dip = 120 - 95 = 25.
    assert legacy["ev_wait_per_share"] == pytest.approx(25.0)

    windowed = compute_dual_ev(paths, S0, dip, rally, fr, patience_window_td=10)
    # Rally at day 30 is outside the dip(day0)+10 window → NOT a round trip.
    assert windowed["p_round_trip_strict"] == pytest.approx(0.0)
    # Exit at entry(0)+10 = day 10 price (92), a small loss: 92 - 95 = -3.
    assert windowed["ev_wait_per_share"] == pytest.approx(-3.0)
    # The correction is strictly downward here (gave up an uncatchable rally).
    assert windowed["ev_wait_per_share"] < legacy["ev_wait_per_share"]


def test_rally_inside_window_still_credited():
    """A rally arriving within the window is still a round-trip win."""
    S0 = 100.0
    n_days = 40
    path = np.full(n_days, 92.0)
    path[0] = 90.0          # dip at day 0
    path[8:] = 130.0        # rally at day 8 (within a 10-day window)
    paths = path.reshape(1, n_days)
    windowed = compute_dual_ev(paths, S0, 95.0, 120.0, 0.0, patience_window_td=10)
    assert windowed["p_round_trip_strict"] == pytest.approx(1.0)
    assert windowed["ev_wait_per_share"] == pytest.approx(25.0)


def test_direct_entry_window_applies_too():
    """Patience applies to DIRECT entry as well (window from day 0), so the
    DIRECT-vs-WAIT selection isn't biased by an asymmetric exit rule."""
    S0 = 100.0
    n_days = 40
    path = np.full(n_days, 101.0)
    path[30:] = 130.0       # rally at day 30 only
    paths = path.reshape(1, n_days)
    rally, fr = 120.0, 0.0

    legacy = compute_dual_ev(paths, S0, 95.0, rally, fr)
    # Legacy credits the day-30 rally for direct entry.
    assert legacy["ev_direct_per_share"] == pytest.approx(rally - S0)  # +20

    windowed = compute_dual_ev(paths, S0, 95.0, rally, fr, patience_window_td=10)
    # Day-30 rally is outside the day0+10 window → exit at day10 price (101).
    assert windowed["ev_direct_per_share"] == pytest.approx(101.0 - S0)  # +1
    assert windowed["ev_direct_per_share"] < legacy["ev_direct_per_share"]


# ---------------------------------------------------------------------------
# Stochastic sanity: direction-of-correction matches the audit harness.
# ---------------------------------------------------------------------------

def _mc_ev(mu, W):
    from src.math_utils import run_mc_joint_conditional, precompute_first_touch_days
    S0, sigma, H = 100.0, 0.80, 60
    paths = run_mc_joint_conditional(S0, sigma, mu, H, n_paths=20000, seed=7)
    dip, rally = 80.0, 130.0
    fr = (dip + rally) / 2 * 70 / 1e4
    dft = precompute_first_touch_days(paths, S0, np.array([dip]), sigma, None, "down", seed=42)[:, 0]
    rft = precompute_first_touch_days(paths, S0, np.array([rally]), sigma, None, "up", seed=43)[:, 0]
    legacy = compute_dual_ev(paths, S0, dip, rally, fr, dft, rft)
    windowed = compute_dual_ev(paths, S0, dip, rally, fr, dft, rft, patience_window_td=W)
    return legacy["ev_wait_pct_of_dip"], windowed["ev_wait_pct_of_dip"]


def test_positive_drift_correction_is_downward():
    """On a positive-drift name the window removes overstated EV
    (uncatchable late rallies + bag-hold recovery to horizon-end)."""
    legacy, windowed = _mc_ev(mu=0.40, W=40)
    assert windowed < legacy


def test_negative_drift_correction_is_upward_or_flat():
    """On a falling name, time-stopping early avoids extra downside, so the
    windowed EV is >= legacy (never more pessimistic)."""
    legacy, windowed = _mc_ev(mu=-0.10, W=40)
    assert windowed >= legacy - 1e-6

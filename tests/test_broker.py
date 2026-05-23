"""Tests for src/broker.py — W4 PR #29 budget allocator.

Sacred constraints the broker must enforce:
  - Total spend ≤ ai_daily_budget_cap_usd ($2.00 by default)
  - T2+ tier requires qualifies_for_t2_plus = True (pre-AI net EV
    positive AND conviction met)
  - T3 requires ambiguity ≥ ai_broker.t3_min_ambiguity
  - Any AI tier requires ambiguity ≥ ai_broker.ai_min_ambiguity
  - T0 is always free; everyone defaults to T0 when budget exhausted

The allocator is pure-function: deterministic given the same inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ai_tiers import resolve_tier
from src.broker import BrokerSnapshot, allocate
from src.config import AI_BROKER, AI_DAILY_BUDGET_CAP_USD


def _snap(ticker, ambiguity, qualifies=True, sigma_class="MID"):
    return BrokerSnapshot(
        ticker=ticker, ambiguity=ambiguity,
        qualifies_for_t2_plus=qualifies, sigma_class=sigma_class,
    )


def test_default_budget_is_two_dollars():
    """Sanity: the broker pulls from config and that's the sacred $2."""
    assert AI_DAILY_BUDGET_CAP_USD == 2.00


def test_empty_input_returns_empty_allocation():
    alloc = allocate([])
    assert alloc.assignments == {}
    assert alloc.spent_usd == 0.0


def test_all_t0_when_no_qualifying_tickers():
    """No ticker qualifies for T2+; below ai_min_ambiguity → all T0."""
    snaps = [_snap("A", 0.05, qualifies=False),
             _snap("B", 0.10, qualifies=False)]
    alloc = allocate(snaps)
    assert all(v == "T0" for v in alloc.assignments.values())
    assert alloc.spent_usd == 0.0


def test_t3_goes_to_highest_ambiguity_qualified_ticker():
    """Two tickers, one high-ambiguity + qualified, one with ambiguity
    below ai_min — high gets T3, low stays T0 (math decisive)."""
    snaps = [_snap("HIGH_A", 0.90, qualifies=True),
             _snap("LOW_A",  0.10, qualifies=True)]
    alloc = allocate(snaps)
    assert alloc.assignments["HIGH_A"] == "T3"
    # LOW_A: ambiguity 0.10 is below ai_min_ambiguity (0.20) → stays T0
    # even though qualified, because math is decisive.
    assert alloc.assignments["LOW_A"] == "T0"


def test_unqualified_ticker_capped_at_t1():
    """qualifies_for_t2_plus=False blocks T2 and T3 regardless of
    ambiguity. T1 is still available (mild ambiguity gate only)."""
    snaps = [_snap("HIGH_UNQUAL", 0.95, qualifies=False)]
    alloc = allocate(snaps)
    assert alloc.assignments["HIGH_UNQUAL"] == "T1"


def test_budget_cap_strictly_respected():
    """Generate enough high-ambiguity tickers to overflow $2; spend
    must stay ≤ cap."""
    snaps = [_snap(f"T{i}", 0.99, qualifies=True) for i in range(20)]
    alloc = allocate(snaps)
    assert alloc.spent_usd <= alloc.cap_usd
    # We should have spent SOMETHING (allocator isn't broken).
    assert alloc.spent_usd > 0.0


def test_budget_cap_overrides_with_lower_cap():
    """Passing budget_usd=0.50 means fewer T3 slots (T3 costs ~$0.30)."""
    snaps = [_snap(f"T{i}", 0.99, qualifies=True) for i in range(10)]
    alloc = allocate(snaps, budget_usd=0.50)
    assert alloc.spent_usd <= 0.50
    n_t3 = sum(1 for v in alloc.assignments.values() if v == "T3")
    t3_cost = resolve_tier("T3").estimated_cost_usd
    # 0.50 / 0.30 = 1.67 → at most 1 T3 slot.
    assert n_t3 <= int(0.50 / t3_cost) + 1


def test_deterministic_ordering():
    """Identical inputs produce identical assignments — broker is pure."""
    snaps = [_snap("A", 0.7, True), _snap("B", 0.7, True),
             _snap("C", 0.3, True)]
    alloc1 = allocate(snaps)
    alloc2 = allocate(snaps)
    assert alloc1.assignments == alloc2.assignments
    assert alloc1.spent_usd == alloc2.spent_usd


def test_ties_broken_alphabetically():
    """Equal ambiguity → alphabetical ordering wins for T3 slot priority."""
    snaps = [_snap("ZEBRA", 0.80, True), _snap("ALPHA", 0.80, True)]
    # Budget that fits exactly one T3 (cost per resolve_tier).
    t3_cost = resolve_tier("T3").estimated_cost_usd
    alloc = allocate(snaps, budget_usd=t3_cost + 0.005)
    assert alloc.assignments["ALPHA"] == "T3"
    # Zebra fell back — should be T2 if it fits, else T0.
    assert alloc.assignments["ZEBRA"] in {"T0", "T2"}


def test_below_t1_threshold_stays_t0_even_with_budget():
    """Even if budget is plentiful, low-ambiguity tickers don't get
    AI tokens — math is decisive, AI input would be wasted."""
    snaps = [_snap("CLEAR", 0.05, qualifies=True)]
    alloc = allocate(snaps, budget_usd=10.0)
    assert alloc.assignments["CLEAR"] == "T0"
    assert alloc.spent_usd == 0.0


def test_three_tier_cascade():
    """Mix of high / medium / low ambiguity, all qualified, modest
    budget — should land T3 / T2 / T0 across them. Specifically:
    one high-ambig qualified gets T3, next-highest get T2 until
    budget exhausts."""
    snaps = [_snap("HI",    0.90, True),  # T3 candidate
             _snap("MID1",  0.55, True),  # T3 if budget, else T2
             _snap("MID2",  0.51, True),  # T3 if budget, else T2
             _snap("MILD",  0.25, True),  # above ai_min, qualified → T2
             _snap("CLEAR", 0.05, True)]  # below ai_min → T0 (math decisive)
    alloc = allocate(snaps, budget_usd=2.00)
    assert alloc.assignments["HI"] == "T3"
    assert alloc.assignments["CLEAR"] == "T0"
    # Spend bounded
    assert alloc.spent_usd <= 2.00


def test_ambiguity_threshold_boundaries_inclusive():
    """Inclusive >= comparison: ambiguity exactly at t3_min_ambiguity
    still qualifies for T3."""
    thresh = AI_BROKER.t3_min_ambiguity
    snaps = [_snap("BOUND", thresh, qualifies=True)]
    alloc = allocate(snaps)
    assert alloc.assignments["BOUND"] == "T3"


def test_t2_plus_disqualified_doesnt_block_t1():
    """A ticker that fails the T2+ gate (e.g. pre-EV negative) can
    still get T1 if ambiguity warrants it. T1 is "cheap diagnostic"."""
    snaps = [_snap("EDGE", 0.45, qualifies=False)]
    alloc = allocate(snaps)
    assert alloc.assignments["EDGE"] == "T1"


def test_realistic_17_ticker_universe_under_budget():
    """End-to-end sanity at full ticker count. Mix of σ-classes and
    ambiguities mirroring what the orchestrator would actually feed
    in; total spend must stay ≤ $2.00 and at least some T3 slots
    should be allocated (the broker isn't being needlessly stingy)."""
    universe = [
        ("LWLG",  0.78, True,  "EXTREME"),
        ("MRAM",  0.72, True,  "EXTREME"),
        ("ENGN",  0.66, True,  "EXTREME"),
        ("VELO3D",0.60, True,  "EXTREME"),
        ("ASTS",  0.55, True,  "HIGH"),
        ("RKLB",  0.48, True,  "HIGH"),
        ("PL",    0.41, False, "HIGH"),
        ("SATS",  0.38, True,  "HIGH"),
        ("GHM",   0.32, True,  "HIGH"),
        ("INTC",  0.28, True,  "MID"),
        ("IPGP",  0.24, True,  "MID"),
        ("LITE",  0.22, False, "MID"),
        ("MU",    0.18, True,  "MID"),
        ("STX",   0.15, True,  "MID"),
        ("AMAT",  0.12, True,  "MID"),
        ("MOG-A", 0.08, True,  "MID"),
        ("GLW",   0.05, True,  "MID"),
    ]
    snaps = [_snap(t, a, q, c) for (t, a, q, c) in universe]
    alloc = allocate(snaps)
    assert alloc.spent_usd <= 2.00
    # At least one T3 slot (we have 5 EXTREME-class qualifiers above t3_min).
    tier_counts = {"T3": 0, "T2": 0, "T1": 0, "T0": 0}
    for t in alloc.assignments.values():
        tier_counts[t] += 1
    assert tier_counts["T3"] >= 1
    # Low-ambiguity tickers (below ai_min) must stay T0.
    assert alloc.assignments["GLW"] == "T0"
    assert alloc.assignments["MOG-A"] == "T0"

"""Tests for PR #73 — BUY-quality safeguards.

Three structural additions:

1. **Forced T2 promotion for would-be-BUYs**: any T2+ qualified ticker
   gets at least T2 critique regardless of ambiguity, so every BUY
   recommendation has Pass 2 adversarial review applied. Was T0
   pre-PR-#73 when ambiguity < 0.20 — operator was acting on math-only
   verdicts.

2. **Limited-history detection**: tickers with < 250 trading days of
   price history get a reliability chip AND forced T2 minimum (Pass 2
   critique compensates for GARCH/σ-anchor instability on short
   histories).

3. **Tighter portfolio gate**: correlation threshold 0.85 → 0.75,
   window 60d → 90d. Catches more substitute-idea clusters (AI cloud,
   storage, semi capex, optical) that the looser settings missed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.broker import (
    BrokerSnapshot, LIMITED_HISTORY_THRESHOLD, allocate, resolve_tier,
)


def _snap(ticker, ambiguity, qualifies=True, sigma_class="MID",
          limited_history=False):
    return BrokerSnapshot(
        ticker=ticker, ambiguity=ambiguity,
        qualifies_for_t2_plus=qualifies, sigma_class=sigma_class,
        limited_history=limited_history,
    )


# =============================================================================
# 1. Forced T2 promotion for would-be-BUYs
# =============================================================================

def test_low_ambiguity_qualified_promoted_to_t2():
    """The core PR #73 rule: math-confident BUY candidate → T2 critique."""
    snaps = [_snap("CONFIDENT_BUY", ambiguity=0.05, qualifies=True)]
    alloc = allocate(snaps, budget_usd=10.0)
    assert alloc.assignments["CONFIDENT_BUY"] == "T2"
    # Cost is t2_cost, not zero
    assert alloc.spent_usd == resolve_tier("T2").estimated_cost_usd


def test_low_ambiguity_unqualified_stays_t0():
    """Math-confident refusal doesn't need AI critique — engine is
    saying don't trade, no BUY signal to vet."""
    snaps = [_snap("REFUSE", ambiguity=0.05, qualifies=False)]
    alloc = allocate(snaps, budget_usd=10.0)
    assert alloc.assignments["REFUSE"] == "T0"
    assert alloc.spent_usd == 0.0


def test_buy_promotion_falls_back_to_t1_when_budget_exhausted():
    """If T2 cost would exceed cap, fall back to T1 (still gets
    Pass 1 critique). Better some adversarial review than none."""
    t2_cost = resolve_tier("T2").estimated_cost_usd
    t1_cost = resolve_tier("T1").estimated_cost_usd
    snaps = [
        _snap("HIGH_AMB", ambiguity=0.90, qualifies=True),     # T3 wins
        _snap("LOW_AMB_QUAL", ambiguity=0.05, qualifies=True),  # T2 candidate
    ]
    # Budget that fits T3 + T1 but NOT T3 + T2
    t3_cost = resolve_tier("T3").estimated_cost_usd
    budget = t3_cost + t1_cost + 0.001
    alloc = allocate(snaps, budget_usd=budget)
    assert alloc.assignments["HIGH_AMB"] == "T3"
    # T2 promotion fell back to T1
    assert alloc.assignments["LOW_AMB_QUAL"] == "T1"


def test_buy_promotion_cost_projection():
    """Sanity: 11 low-ambiguity qualified tickers all promoted to T2 →
    cost roughly = 11 × $0.08 = $0.88 from promotion alone, well
    within $2/day cap."""
    snaps = [_snap(f"T{i}", ambiguity=0.05, qualifies=True) for i in range(11)]
    alloc = allocate(snaps, budget_usd=2.0)
    t2_count = sum(1 for v in alloc.assignments.values() if v == "T2")
    assert t2_count == 11
    assert alloc.spent_usd == pytest.approx(11 * resolve_tier("T2").estimated_cost_usd)
    assert alloc.spent_usd <= 2.0


# =============================================================================
# 2. Limited-history forced T2 minimum
# =============================================================================

def test_limited_history_forces_t2_even_unqualified():
    """Limited-history names get forced T2 critique regardless of
    ambiguity OR qualification. GARCH unstable on short histories,
    so AI critique compensates."""
    snaps = [_snap("NEW_IPO", ambiguity=0.05, qualifies=False,
                    limited_history=True)]
    alloc = allocate(snaps, budget_usd=10.0)
    assert alloc.assignments["NEW_IPO"] == "T2"


def test_full_history_stays_t0_when_low_ambig_unqualified():
    """Long-history non-qualifying refusal correctly stays T0 — only
    limited-history OR qualified tickers get the forced critique."""
    snaps = [_snap("ESTABLISHED_REFUSED", ambiguity=0.05,
                    qualifies=False, limited_history=False)]
    alloc = allocate(snaps, budget_usd=10.0)
    assert alloc.assignments["ESTABLISHED_REFUSED"] == "T0"
    assert alloc.spent_usd == 0.0


def test_limited_history_threshold_is_250():
    """The threshold constant is exposed for the engine to use."""
    assert LIMITED_HISTORY_THRESHOLD == 250


# =============================================================================
# 3. Tighter portfolio gate config
# =============================================================================

def test_portfolio_gate_threshold_tightened_to_75():
    from src.config import PORTFOLIO_GATE
    assert PORTFOLIO_GATE.correlation_threshold == pytest.approx(0.75)


def test_portfolio_gate_window_extended_to_90():
    from src.config import PORTFOLIO_GATE
    assert PORTFOLIO_GATE.correlation_window_days == 90


# =============================================================================
# 4. BrokerSnapshot extension is backward-compatible
# =============================================================================

def test_snapshot_limited_history_defaults_false():
    """Old code paths constructing snapshots without limited_history
    arg should still work — field defaults to False."""
    s = BrokerSnapshot(
        ticker="X", ambiguity=0.3,
        qualifies_for_t2_plus=True, sigma_class="MID",
    )
    assert s.limited_history is False


def test_orchestrator_snapshot_parse_accepts_old_json():
    """Old subprocess output without limited_history field should
    parse correctly (graceful degradation, defaults to False)."""
    from src import orchestrator as orch
    stdout = (
        'BROKER_SNAPSHOT_JSON={"ticker":"X","ambiguity":0.3,'
        '"qualifies_for_t2_plus":true,"sigma_class":"MID"}'
    )
    snap = orch.parse_snapshot(stdout)
    assert snap is not None
    assert snap.limited_history is False


def test_orchestrator_snapshot_parse_reads_new_field():
    from src import orchestrator as orch
    stdout = (
        'BROKER_SNAPSHOT_JSON={"ticker":"NEW","ambiguity":0.3,'
        '"qualifies_for_t2_plus":true,"sigma_class":"HIGH",'
        '"limited_history":true}'
    )
    snap = orch.parse_snapshot(stdout)
    assert snap is not None
    assert snap.limited_history is True


# =============================================================================
# 5. Reliability chip surfaces limited-history
# =============================================================================

def test_reliability_chip_limited_history_triggers_below_250():
    from src.reporter import _reliability_chips
    vp = SimpleNamespace(garch_alpha_plus_beta=0.7, divergence_pp=5.0)
    chips = _reliability_chips(
        vol_profile=vp, base_signals=[],
        sigma_class_mismatch=None, history_bars=200,
    )
    labels = [label for label, _ in chips]
    assert any("limited history" in lbl and "200" in lbl for lbl in labels)


def test_reliability_chip_limited_history_not_triggered_at_threshold():
    """Exactly 250 bars is the threshold — at or above doesn't fire."""
    from src.reporter import _reliability_chips
    vp = SimpleNamespace(garch_alpha_plus_beta=0.7, divergence_pp=5.0)
    chips = _reliability_chips(
        vol_profile=vp, base_signals=[],
        sigma_class_mismatch=None, history_bars=250,
    )
    labels = [label for label, _ in chips]
    assert not any("limited history" in lbl for lbl in labels)


def test_reliability_chip_history_bars_optional_arg():
    """When history_bars not provided (legacy call sites), chip just
    skips the check — doesn't crash."""
    from src.reporter import _reliability_chips
    vp = SimpleNamespace(garch_alpha_plus_beta=0.7, divergence_pp=5.0)
    chips = _reliability_chips(
        vol_profile=vp, base_signals=[], sigma_class_mismatch=None,
    )
    # Should not crash; should not include the limited-history chip
    labels = [label for label, _ in chips]
    assert not any("limited history" in lbl for lbl in labels)


def test_reliability_chip_limited_history_severity_is_red():
    """LIMITED HISTORY is a math-instability flag, surfaced in red
    severity. Comparable to near-IGARCH — math layer can't trust
    itself, only operator awareness + AI critique mitigates."""
    from src.reporter import _reliability_chips
    vp = SimpleNamespace(garch_alpha_plus_beta=0.7, divergence_pp=5.0)
    chips = _reliability_chips(
        vol_profile=vp, base_signals=[],
        sigma_class_mismatch=None, history_bars=180,
    )
    for label, severity in chips:
        if "limited history" in label:
            assert severity == "red"
            return
    pytest.fail("limited-history chip not generated")

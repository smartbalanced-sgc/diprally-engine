"""Budget broker — multi-ticker AI tier allocator (W4 PR #29).

Takes a per-ticker math snapshot (T0 outputs only — no AI yet) and
returns the AI tier each ticker should run at, under the $2/day hard
cap. Pure function: deterministic given inputs, no I/O, no model calls.

Allocation algorithm (greedy, walks T3→T2→T1):

  1. All tickers start at T0 (free).
  2. Sort by ambiguity descending, ticker name alphabetical as tie-break.
  3. T3 pass — for each ticker in priority order:
       - skip if ambiguity < ai_broker.t3_min_ambiguity
       - skip if doesn't qualify for T2+ gate (sacred: pre-AI net EV
         positive AND conviction met; broker reads this from the
         snapshot, doesn't recompute)
       - if remaining budget covers T3 cost → assign T3
  4. T2 pass — fill T2+ qualified tickers with at least mild ambiguity:
       - skip if ambiguity < ai_broker.ai_min_ambiguity (below this
         the math is decisive enough that AI tokens are wasted)
       - skip if not qualified for T2+
       - if remaining budget covers T2 cost → assign T2
  5. T1 pass — catches (a) unqualified tickers above ai_min_ambiguity
     and (b) qualified tickers that lost their T2 slot to the cap. T1
     itself doesn't need pre-EV qualification (cheap diagnostic).
  6. Everything left stays T0.

Sacred CLAUDE.md:
  T2: ~$0.10 — "Pre-AI net EV positive AND conviction met"
  T3: ~$0.30 — "T2 critique passed + budget allows"

The "T2 critique passed" check is a RUNTIME upgrade decision, not a
pre-allocation one — we can't know until T2 runs. The broker reserves
budget assuming T3 will go through; if T2 critique fails at runtime,
the engine demotes to T2 and refunds the difference (handled in
engine integration, not here).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.ai_tiers import resolve_tier
from src.config import AI_BROKER, AI_DAILY_BUDGET_CAP_USD


@dataclass(frozen=True)
class BrokerSnapshot:
    """One ticker's T0 math state — broker input.

    qualifies_for_t2_plus encodes the sacred CLAUDE.md T2 gate:
        pre-AI net EV positive AND conviction met.
    The engine computes this once during the T0 math layer (cheap)
    and hands it to the broker.

    limited_history (PR #73): set when fetch_history returns fewer than
    LIMITED_HISTORY_THRESHOLD bars (default 250 trading days ≈ 12 months).
    GARCH and σ-triangulation are structurally unreliable on shorter
    histories. Broker forces T2 minimum for these so Pass 2 critique
    is always applied — institutional safeguard, not a refusal.
    """
    ticker: str
    ambiguity: float                       # ∈ [0, 1] from compute_ambiguity()
    qualifies_for_t2_plus: bool
    sigma_class: str                       # for logging / future per-class biasing
    limited_history: bool = False          # PR #73 — < 250 trading days available


# PR #73: number of bars required for stable GARCH fit + σ-anchor agreement.
# Below this, broker forces T2 critique to compensate for math-layer
# structural uncertainty on newly-public / recently-restructured tickers.
LIMITED_HISTORY_THRESHOLD = 250


@dataclass(frozen=True)
class BrokerAllocation:
    """Broker output. assignments maps ticker → tier name. spent_usd
    is the total estimated cost across all assigned tickers (T0 = $0,
    not summed)."""
    assignments: dict[str, str]
    spent_usd: float
    cap_usd: float
    notes: list[str] = field(default_factory=list)


def allocate(
    snapshots: list[BrokerSnapshot],
    budget_usd: Optional[float] = None,
) -> BrokerAllocation:
    """Allocate AI tiers across a ticker list under a budget cap.

    budget_usd defaults to AI_DAILY_BUDGET_CAP_USD (sacred $2/day).
    Passing an explicit value (e.g. remaining intraday budget for a
    re-run) is supported but the cap is still enforced as a hard
    ceiling — broker never exceeds it.
    """
    if budget_usd is None:
        budget_usd = AI_DAILY_BUDGET_CAP_USD
    cap = float(budget_usd)
    notes: list[str] = []

    # Resolve tier costs once.
    t1_cost = resolve_tier("T1").estimated_cost_usd
    t2_cost = resolve_tier("T2").estimated_cost_usd
    t3_cost = resolve_tier("T3").estimated_cost_usd

    # Deterministic order: ambiguity desc, then ticker alpha.
    ranked = sorted(snapshots, key=lambda s: (-s.ambiguity, s.ticker))

    # Default everyone to T0.
    assignments: dict[str, str] = {s.ticker: "T0" for s in snapshots}
    spent = 0.0

    # 3. T3 pass — high-ambiguity + T2+ qualified, in priority order.
    for s in ranked:
        if s.ambiguity < AI_BROKER.t3_min_ambiguity:
            continue
        if not s.qualifies_for_t2_plus:
            continue
        if spent + t3_cost > cap:
            notes.append(
                f"{s.ticker}: T3 candidate (ambiguity {s.ambiguity:.2f}) "
                f"but budget exhausted (${spent:.2f} + ${t3_cost:.2f} > ${cap:.2f})"
            )
            continue
        assignments[s.ticker] = "T3"
        spent += t3_cost

    # 4. T2 pass — PR #87 broker fix: T2 eligibility relaxed.
    # Old logic: required qualifies_for_t2_plus (pre-AI net EV positive).
    # That created a cascade: short-horizon math can't find positive-EV
    # setups, qualifies_for_t2_plus is False everywhere, AI Pass 2 never
    # runs, math layer never gets the catalyst-driven mu kicker, EV stays
    # negative. Self-reinforcing dead loop.
    #
    # New logic: T2 eligible if EITHER qualifies_for_t2_plus OR ambiguity
    # is non-trivial (math uncertain enough to benefit from AI critique).
    # Lets AI Pass 2 actually engage on borderline tickers where the
    # math layer says "uncertain" — the EXACT cases that need AI's
    # catalyst overlay to render a verdict.
    for s in ranked:
        if assignments[s.ticker] != "T0":
            continue
        if s.ambiguity < AI_BROKER.ai_min_ambiguity:
            continue
        # PR #87: drop the strict qualifies_for_t2_plus gate. Ambiguity
        # is now the primary signal — uncertain math = let AI weigh in.
        # qualifies_for_t2_plus is retained as a TIE-BREAKER in ranking
        # (already done at line 91-97 — ranked sort).
        if spent + t2_cost > cap:
            notes.append(
                f"{s.ticker}: T2 candidate but budget exhausted"
            )
            continue
        assignments[s.ticker] = "T2"
        spent += t2_cost

    # PR #73 — Forced T2 promotion for institutional rigor:
    # Any T2+ qualified ticker AND any limited-history ticker gets
    # at least T2 Pass-2 adversarial critique, even if ambiguity is low.
    #
    # Rationale: prior to this rule, broker assigned T0 to confident-math
    # qualifying tickers — meaning likely BUYs got NO AI critique.
    # Operator was acting on math-only recommendations on names like
    # MRVL, NBIS, CRWV, ANAB with zero adversarial second opinion.
    # Sacred decision #7 says Pass 2 wins — but Pass 2 must run to
    # win. This pass guarantees Pass 2 reviews every potential BUY.
    #
    # Limited-history names (< 250 trading days) ALSO get forced T2
    # because GARCH and σ-triangulation are structurally unreliable
    # at short histories — AI critique adds the second opinion the
    # math layer can't trust itself for.
    for s in ranked:
        if assignments[s.ticker] != "T0":
            continue
        needs_review = s.qualifies_for_t2_plus or s.limited_history
        if not needs_review:
            continue
        if spent + t2_cost > cap:
            # Fall back to T1 (cheaper but still gives Pass 1 critique)
            # rather than dropping to T0. Better some review than none.
            if spent + t1_cost <= cap:
                assignments[s.ticker] = "T1"
                spent += t1_cost
                notes.append(
                    f"{s.ticker}: forced T2 (BUY-quality safeguard) → "
                    f"T1 fallback due to cap"
                )
            else:
                notes.append(
                    f"{s.ticker}: needs review (qualifies={s.qualifies_for_t2_plus}, "
                    f"limited_history={s.limited_history}) but budget exhausted"
                )
            continue
        assignments[s.ticker] = "T2"
        spent += t2_cost

    # 5. T1 pass — mild-ambiguity tickers, no pre-EV gate required.
    # This catches both (a) unqualified tickers above ai_min_ambiguity
    # and (b) qualified tickers that lost their T2 slot to the cap.
    for s in ranked:
        if assignments[s.ticker] != "T0":
            continue
        if s.ambiguity < AI_BROKER.ai_min_ambiguity:
            continue
        if spent + t1_cost > cap:
            notes.append(
                f"{s.ticker}: T1 candidate but budget exhausted"
            )
            continue
        assignments[s.ticker] = "T1"
        spent += t1_cost

    return BrokerAllocation(
        assignments=assignments,
        spent_usd=spent,
        cap_usd=cap,
        notes=notes,
    )


def format_allocation(allocation: BrokerAllocation,
                       snapshots: list[BrokerSnapshot]) -> str:
    """Human-readable allocation summary — used by the broker_preview CLI
    and any future orchestrator log line. Tickers in allocation order
    (ambiguity desc), grouped by tier."""
    snap_by_ticker = {s.ticker: s for s in snapshots}
    by_tier: dict[str, list[str]] = {"T3": [], "T2": [], "T1": [], "T0": []}
    for ticker, tier in allocation.assignments.items():
        by_tier[tier].append(ticker)
    # Sort each tier's tickers by ambiguity desc.
    for tier_name in by_tier:
        by_tier[tier_name].sort(
            key=lambda t: -snap_by_ticker[t].ambiguity
        )

    lines = []
    lines.append("=" * 78)
    lines.append(
        f"BUDGET BROKER — proposed allocation across {len(snapshots)} tickers"
    )
    lines.append(
        f"Spend ${allocation.spent_usd:.2f} of ${allocation.cap_usd:.2f} cap "
        f"({allocation.spent_usd / allocation.cap_usd * 100:.0f}%)"
    )
    lines.append("=" * 78)
    for tier_name in ("T3", "T2", "T1", "T0"):
        tickers = by_tier[tier_name]
        if not tickers:
            continue
        cost_each = resolve_tier(tier_name).estimated_cost_usd
        lines.append(
            f"  {tier_name}  (${cost_each:.2f} each × {len(tickers)} = "
            f"${cost_each * len(tickers):.2f})"
        )
        for t in tickers:
            s = snap_by_ticker[t]
            qual = "✓" if s.qualifies_for_t2_plus else " "
            lines.append(
                f"    {t:<8} ambiguity={s.ambiguity:.2f}  "
                f"σ-class={s.sigma_class:<7}  T2+qual={qual}"
            )
    if allocation.notes:
        lines.append("")
        lines.append("Budget-exhaustion notes:")
        for n in allocation.notes:
            lines.append(f"  - {n}")
    return "\n".join(lines)

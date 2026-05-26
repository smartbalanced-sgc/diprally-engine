# DIPRALLY-ENGINE — Claude Code session contract

Multi-ticker swing decision engine. Daily quant + AI analysis of 17 volatile
stocks to identify defensible dip-and-rally round-trip setups within 60 trading
days. Refuses negative-EV setups.

## Working style (Jesse)
- Caveman: terse, surgical, no fluff
- Restate task before acting
- Wait for explicit "go" before commits/pushes
- One change at a time
- No "yes-man" — push back honestly
- End every response with `#End`
- **Basis-point clarity**: whenever you state a value in bps (basis points) — in
  chat or in engine output / reports — ALSO express it as a percentage in
  parentheses. Examples: "EV hurdle +25 bps (0.25%)", "friction 35 bps (0.35%)",
  "+182 bps EV (1.82%)". 1 bp = 0.01% always. Never drop the percentage gloss.

## Asking questions
- ALWAYS explain each question in plain English first (what the question is
  actually about, what's at stake, what the choice means in real-world
  trading terms). No jargon-only or naked-options-list questions.
- When asked any question, ELEVATE to an institution-grade super-intelligent
  stock trader and provide the highest-conviction correct recommendation.
  No "your call", "both have merit", "depends on preference", or other lazy
  middle-ground punts. Pick the right answer with reasoning. If the right
  answer is genuinely ambiguous, say so explicitly and explain why; do not
  use false ambiguity to avoid commitment.

## Approval gates
- NEVER commit without explicit "go"
- NEVER push without explicit "go"
- NEVER PR/merge to main without explicit "merge to main"
- Each meaningful step = its own commit (don't pile work into one giant commit)

## Diagnostics + FMP data fetches (CANON — never violate)
Operator's time is the bottleneck. Speculative round-trips wasting his time
are forbidden. Whenever you need to diagnose engine output OR fetch FMP data
to validate a hypothesis, in a SINGLE response:

  1. State the decision being made.
  2. State the data needed to make it (precise — endpoint, field, ticker, time).
  3. Provide ONE exact bash command for the operator to copy-paste-run.
     - Always print HTTP status + raw response body up to 1000 chars,
       NOT a derived pretty-print that can fail on unexpected payload shape.
  4. If the endpoint may be plan-tier restricted, also provide a verbatim
     question for FMP support — copy-pasteable, no editing required.
  5. NEVER use iterative "run this, then I'll figure out next step" patterns.

Math-layer diagnostics follow the same rule, plus: numerical harness in the
sandbox FIRST (synthesize realistic inputs, decompose by axis, identify root
cause with hard numbers), THEN propose a single surgical fix with a regression
test that fails before and passes after.

## Hard constraints
- AI cost cap: **$2/day across all tickers** (HARD)
- Token discipline: AI allocated by budget broker, never sprayed
- Same-day re-runs must not corrupt CSV or double-charge AI
- No "capital" concept — recommendation tool, user sizes externally
- Ticker universe is CONFIG (YAML), not code. Current roster is 17 names
  but adding/removing is a YAML edit, not a code change. Engine must
  handle any universe size without modification.

## Sacred decisions — NEVER violate
 1. No block bootstrap
 2. No multi-step vol forecast
 3. No synthesized reliability score (components shown separately)
 4. No SNDK-specific hardcodes anywhere in code or display
 5. No imports from sgc-dip-engine
 6. No "capital" / position-sizing concept
 7. Pass 2 wins — its drift REPLACES Pass 1's in signal slot before MC
 8. Three-method math cross-check on every run (MC + PDE + closed-form)
 9. Brownian bridge correction on barrier MC
10. AI outputs are arithmetic inputs, not display prose
11. Same-day CSV dedup — one canonical row per (ticker, date)
12. Bayesian prior across days with same-day artifact guard
13. EV-hurdle gate — refuse to recommend if EV < σ-class threshold of dip
    after friction. σ-class thresholds (PR #70): MID 50 bps (0.50%);
    HIGH and EXTREME 25 bps (0.25%) — the lower hurdle on HIGH/EXTREME
    acknowledges that the engine's "blind execution" EV estimate
    understates realized EV for active swing traders who time entry/exit
    discretionarily (sacred #6 — trader sizes / manages externally)
14. Trend filter — refuse dip if 30d momentum < -25% AND no fundamental catalyst
15. Insider signal dropped (Form 4 lag + noise)
16. Method-disagreement refusal — MC vs PDE diverge >5pp on marginal = no recommendation
17. **All configurable values live in `config/diprally.yaml`. `src/` holds code only.**
    Tickers, σ-class table, peer mappings, blend weights, vol schedule
    multipliers, AI pricing, conviction thresholds, method tolerance formula
    coefficients, Bayesian parameters, panic floor, friction bps, signal-level
    thresholds (analyst spread brackets, sector regime caps, macro drift
    levels, insider scaling, short-interest brackets, peer RS caps, regime
    detection triggers, catalyst magnitude map), data-fetch thresholds (VIX
    macro brackets, options liquidity cutoff), PDE grid resolution, GARCH
    fit parameters, sensitivity scenario list — none of these belong as
    Python constants. `src/config.py` is a YAML loader with schema validation
    that exposes typed constants for import convenience. Changing a threshold
    must NEVER require a code edit, a PR, or a deploy.
18. **Parabola filter** — refuse dip-buy when `mom_30d ≥` σ-class threshold
    AND no in-horizon bearish-direction catalyst surfaced by AI Pass 1/Pass 2.
    σ-class thresholds (PR #70): MID +50%, HIGH +80%, EXTREME +100% —
    calibrated to the actual vol regime of each class (a +50% monthly
    move is exceptional for AMAT-class names but baseline for the AI/
    semi/momentum names this engine targets). Mirror of sacred #14
    (falling-knife trend filter) for blow-off tops. Asymmetric exception:
    requires SPECIFICALLY bearish catalyst — generic "two-sided earnings"
    is the math layer's default, not a de-rating thesis. Codified by
    PRs #41 / #44 / #45 / #46 / #51 / #70.

## Ticker universe (current roster — adjust via YAML)
- **EXTREME (11)**: LWLG, MRAM, ENGN, VELO, SNDK, ARM, CRWV, NBIS, INOD, CRDO, ANAB
  - VELO replaces VELO3D (delisted 2024, relisted Aug 2025 as VELO)
  - SNDK = WDC Flash spinoff (Feb 2025 IPO); ARM IPO Sep 2023; CRWV IPO Mar 2025;
    NBIS = post-Yandex restructure. Limited history names — auto-detector may
    flag class shifts in early cycles.
- **HIGH (6)**: ASTS, RKLB, PL, SATS, GHM, MRVL
- **MID (9)**: INTC, IPGP, LITE, MU, STX, AMAT, MOG-A, GLW, LRCX

> **Ticker convention**: canonical form across this repo uses dashes for
> class shares (MOG-A, BRK-B, BF-B) — the Yahoo Finance / industry-standard
> form. **FMP empirically accepts the dash form** (verified 2026-05-22 via
> MOG-A smoke). FMP also has MOG.A in its database but our Starter plan
> tier returns 402 on the dot form. Conclusion: standardize on the dash
> form, use it as-is when calling FMP. If a future provider requires a
> different separator, the W2 registry adds per-provider translation.

## σ-class defaults
| Class    | Conv-dip | Conv-rally | Grid dip | Grid rally | Panic floor | Friction bps RT | AI vol_mult (H/M/L) |
|----------|----------|-----------|----------|-----------|-------------|-----------------|---------------------|
| EXTREME  | 0.60     | 0.75      | -50%     | +60%      | -35%        | 70              | 1.10 / 1.00 / 0.90  |
| HIGH     | 0.65     | 0.75      | -35%     | +50%      | -25%        | 35              | 1.15 / 1.00 / 0.90  |
| MID      | 0.65     | 0.70      | -20%     | +30%      | -18%        | 18              | 1.25 / 1.00 / 0.80  |

## AI tier system (broker, $2/day hard cap)
- **T0** ($0)    — math only. Default; every ticker daily.
- **T1** (~$0.02) — Haiku Pass 1 only (web_search cap 1). Mild trigger.
- **T2** (~$0.10) — Sonnet Pass 1 + Pass 2. Pre-AI net EV positive AND conviction met.
- **T3** (~$0.30) — Opus Pass 1 + Sonnet Pass 2 + Haiku stress. T2 critique passed + budget allows.

Broker: T0 all 17 first, sort by ambiguity, greedy allocate T3→T2→T1 within $2.

## Wave plan (11 waves, sequential, approval-gated)
W0 scaffolding + v2 migration → W1 AI efficiency → W2 multi-ticker + registry →
W3 σ-class auto-detection → W4 budget broker + ambiguity → W5 orchestrator + cron
+ dashboard → W6 institutional signals → W7 execution realism → W8 risk mgmt
→ W9 fat-tail MC → W10 calibration.

## Pointers
- Full session-start context: seed handover (Jesse's clipboard) or
  `docs/handover/01_SESSION_CONTEXT.md` (populated at W2+).
- **Deferred fixes from earlier waves**: `docs/handover/_DEFERRED.md`. Scan at
  the start of each wave; clear items whose target wave is now active.
- v1 seed (`tools/_seed_v1.py`) and v2 seed (`tools/_seed_v2.py`) are deleted at
  end of W0. After W0, source of truth is `src/`.

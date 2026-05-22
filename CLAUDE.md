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

## Approval gates
- NEVER commit without explicit "go"
- NEVER push without explicit "go"
- NEVER PR/merge to main without explicit "merge to main"
- Each meaningful step = its own commit (don't pile work into one giant commit)

## Hard constraints
- AI cost cap: **$2/day across all tickers** (HARD)
- Token discipline: AI allocated by budget broker, never sprayed
- Same-day re-runs must not corrupt CSV or double-charge AI
- No "capital" concept — recommendation tool, user sizes externally
- 17-ticker universe (fixed)

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
13. EV-hurdle gate — refuse to recommend if EV < +50 bps of dip after friction
14. Trend filter — refuse dip if 30d momentum < -25% AND no fundamental catalyst
15. Insider signal dropped (Form 4 lag + noise)
16. Method-disagreement refusal — MC vs PDE diverge >5pp on marginal = no recommendation

## Ticker universe (17, fixed)
- **EXTREME (4)**: LWLG, MRAM, ENGN, VELO3D
- **HIGH (5)**: ASTS, RKLB, PL, SATS, GHM
- **MID (8)**: INTC, IPGP, LITE, MU, STX, AMAT, MOG.A, GLW

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
- v1 seed (`tools/_seed_v1.py`) and v2 seed (`tools/_seed_v2.py`) are deleted at
  end of W0. After W0, source of truth is `src/`.

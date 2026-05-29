# DIPRALLY-ENGINE — Claude Code session contract

Multi-ticker swing decision engine. Daily quant + AI analysis of 26 volatile
stocks (default roster; +5 large-caps in YAML `tickers_scratch`, run only on
explicit `--tickers`) to identify defensible dip-and-rally round-trip setups
within 20 trading days. Refuses negative-EV setups.

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

## End-to-end audit protocol (CANON — MANDATORY when asked to audit/evaluate, OR when any output looks structurally extreme)
Added 2026-05-28 after repeated failures where the operator (a self-described
novice) out-diagnosed the engine's root causes — EV payoff asymmetry, catalyst
blindness, invisible analyst signals — that a shallow "the code does what it
says / the math checks out" pass missed. The cardinal sin: confirming
self-consistency instead of fitness-for-purpose. This protocol forces
falsification and input-auditing by default. Run ALL steps in ONE pass; the
deliverable is a ranked defect list, never reassurance.

  1. **Anomaly-first / falsify.** An extreme output (0 BUYs in N runs,
     all-identical verdicts, every value at a cap) is a STRUCTURAL red flag,
     not data coincidence. State the null — "the system CANNOT produce
     outcome X" — and try to FALSIFY it with a numerical harness on realistic
     inputs BEFORE touching anything else.
  2. **Two separate questions, always:** (a) does the code do what it
     documents? (b) is what it documents CORRECT for the goal? Passing (a)
     while never asking (b) is the cardinal sin.
  3. **Trace the full pipeline, stage by stage:** data source → fetch →
     signal → AI catalyst → drift → MC → EV → gate → verdict → display. One
     line per stage: what flows in, what flows out, what can be silently
     wrong/empty here.
  4. **Audit inputs before math.** Sample the ACTUAL AI catalysts / signals
     for ≥1 ticker against an INDEPENDENT ground-truth web search. Garbage-in
     invalidates any downstream correctness.
  5. **Enumerate every silent-failure path:** try/except swallows,
     `_none_signal` fallbacks, plan-restricted endpoints returning `[]`,
     "degrade gracefully" branches, hard-coded block/allow-lists. Verify each
     is firing as intended, not masking a problem.
  6. **Interrogate the objective function directly.** Harness + decompose by
     axis (upside cap vs downside tail, win-prob vs loss-magnitude) to expose
     structural bias in the metric itself.
  7. **Adversarial, not confirmatory.** The deliverable is a ranked list of
     what's WRONG or FRAGILE by impact. "I found nothing / the math checks
     out" means the audit wasn't deep enough — and that claim is NEVER made
     without a harness reproducing the observed behavior on real inputs.

## Hard constraints
- AI cost cap: **$2/day across all tickers** (HARD)
- Token discipline: AI allocated by budget broker, never sprayed
- Same-day re-runs must not corrupt CSV or double-charge AI
- No "capital" concept — recommendation tool, user sizes externally
- Ticker universe is CONFIG (YAML), not code. Default roster is 26 names
  (`tickers`); 5 more in `tickers_scratch` run only on explicit `--tickers`.
  Adding/removing/promoting is a YAML edit, not a code change. Engine must
  handle any universe size without modification.
- Mean reversion: the engine supports a mean-reversion drift term
  (`run.py --mean-reversion`, anchor in YAML `mean_reversion.anchor_pct_below_spot`)
  but it DEFAULTS TO 0.0 (OFF) and the orchestrator does NOT pass it. Pure
  GBM is the live default. Treat this as a known structural lever, not an
  accident — enabling it is a deliberate calibration decision, not a bugfix.

## Sacred decisions
**Two tiers of "sacred":**
- **Architecture** (truly NEVER violate — changing these would corrupt the
  model's mathematical or statistical integrity): #1-12, 15-17.
- **Calibration thresholds** (these LOOK like sacred rules but are YAML
  values that must be recalibrated as market regimes shift): #13 EV hurdle,
  #14 trend filter threshold, #18 parabola thresholds. When the engine
  produces structurally extreme output (0 BUYs, all-refuse), interrogate
  these first. Recalibrating a YAML threshold is NOT a sacred violation —
  it is the correct response to a miscalibrated engine.
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
    discretionarily (sacred #6 — trader sizes / manages externally).
    **Calibration note**: EXTREME friction is 70 bps (0.70%) RT, so
    EXTREME names need 95 bps (0.95%) gross EV just to clear the gate.
    With pure GBM (mean reversion OFF by default), this is achievable but
    tight. If 0-BUY persists, interrogate friction bps and EV hurdle as a
    combined threshold — they are YAML values, not architectural constants.
14. Trend filter — refuse dip if 30d momentum < -25% AND no fundamental
    catalyst. **Calibration note**: this threshold is GLOBAL (not σ-class
    adjusted). At EXTREME σ≈130%, a 30-day 1σ move is ~36%, so a -25%
    drop is LESS than 1σ — normal variance, not a falling knife. The filter
    is miscalibrated for high-vol classes and will block legitimate dip-buy
    setups on EXTREME/HIGH names where the dip IS the opportunity. The
    -0.25 threshold lives in YAML (`trend_filter.mom_30d_threshold`) and
    can be recalibrated per-class. Recommended direction: -0.40 or -0.50
    for EXTREME, -0.35 for HIGH, leave -0.25 for MID.
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
    σ-class thresholds (PR #89, supersedes PR #70's +50/+80/+100): MID +80%,
    HIGH +150%, EXTREME +200% — recalibrated for the AI bull cycle where
    EXTREME/HIGH names (MRAM/LWLG/VELO/INOD; MU/ARM/NBIS/INTC/MRVL) routinely
    ran 80-180% mom_30d WITHOUT parabolic reversal, so the old thresholds
    were refusing legitimate continuation setups. Only TRUE blow-offs trip
    it now. Mirror of sacred #14 (falling-knife trend filter) for blow-off
    tops. Asymmetric exception: requires SPECIFICALLY bearish catalyst —
    generic "two-sided earnings" is the math layer's default, not a
    de-rating thesis. Codified by PRs #41 / #44 / #45 / #46 / #51 / #70 / #89.

## Ticker universe — lives in YAML, NOT here (sacred #17)
The roster, σ-class assignment, stock/ETF peers, and every per-ticker setting
live in the `config/diprally.yaml` registry block (each entry keyed by symbol,
carrying `sigma_class`, `stock_peers`, `etf_peer`). The YAML is the SINGLE
source of truth — do NOT maintain a ticker list here (it drifts and contradicts
the config, which is exactly what sacred #17 forbids). To see the current
universe grouped by class:

    python3 -c "import yaml,collections; c=yaml.safe_load(open('config/diprally.yaml')); \
    g=collections.defaultdict(list); \
    [g[v['sigma_class']].append(k) for blk in ('tickers','tickers_scratch') \
    for k,v in c.get(blk,{}).items()]; print('DEFAULT roster = tickers block; scratch runs only on --tickers'); \
    [print(f'{cl} ({len(g[cl])}): {sorted(g[cl])}') for cl in ('EXTREME','HIGH','MID')]"

As of 2026-05-29 the DEFAULT daily roster (`tickers` block) is 26 names
(11 EXTREME / 7 HIGH / 8 MID). A separate `tickers_scratch` block holds 5
large-caps (ADBE, AVGO, DE, IBM, ORCL) that are ad-hoc only: default
orchestrator runs (no `--tickers` flag) iterate `tickers` ONLY, so scratch
names get full registry support when explicitly requested but do NOT run in
the daily cycle. **Watch-out**: if the intent was for those large-cap
diversifiers to lift the universe BUY hit rate, leaving them in scratch means
they never run daily — promoting them into `tickers` is a YAML move, not code.
Notable registry facts that aren't obvious from the symbol alone:
- VELO replaces VELO3D (delisted 2024, relisted Aug 2025 as VELO).
- Limited-history names (SNDK Feb-2025 spinoff, ARM Sep-2023 IPO, CRWV Mar-2025
  IPO, NBIS post-Yandex restructure) — σ auto-detector may flag class shifts
  in early cycles; broker forces ≥T2 on limited-history tickers.
- MU is HIGH (realized vol 70-90% annualised in the AI-cycle semi regime).

> **Ticker convention**: canonical form across this repo uses dashes for
> class shares (MOG-A, BRK-B, BF-B) — the Yahoo Finance / industry-standard
> form. **FMP empirically accepts the dash form** (verified 2026-05-22 via
> MOG-A smoke). FMP also has MOG.A in its database but our Starter plan
> tier returns 402 on the dot form. Conclusion: standardize on the dash
> form, use it as-is when calling FMP. If a future provider requires a
> different separator, the W2 registry adds per-provider translation.

## σ-class defaults — lives in YAML, NOT here (sacred #17)
The per-class conviction / grid / panic / friction / EV-hurdle / parabola /
vol-mult values are the `sigma_classes` block in `config/diprally.yaml`. That
YAML is authoritative — read it directly rather than trusting a transcribed
table here (the previous hardcoded table drifted to stale 60d/0.75 values and
caused 0-BUY misdiagnosis). Quick dump:

    python3 -c "import yaml,json; c=yaml.safe_load(open('config/diprally.yaml')); \
    print(json.dumps(c['sigma_classes'], indent=2))"

Orientation only (verify against YAML before acting): conviction is far lower
than a naive reader expects — EXTREME ≈0.55/0.55, HIGH ≈0.60/0.65, MID
≈0.65/0.70 — because the old 0.75 rally-conditional was mathematically
unachievable at σ>60% and guaranteed 0 BUYs. EV hurdle is per-class
(EXTREME/HIGH 25 bps = 0.25%, MID 50 bps = 0.50%). Grid/panic are 20d-horizon
values (PR #86 rescaled from the legacy 60d grid by √(20/60)).

## AI tier system (broker, $2/day hard cap)
- **T0** ($0)    — math only. Default; every ticker daily.
- **T1** (~$0.02) — Haiku Pass 1 only (web_search cap 1). Mild trigger.
- **T2** (~$0.10) — Sonnet Pass 1 + Pass 2. Gated on ambiguity ≥ `ai_min_ambiguity`
  (PR #87 dropped the old "pre-AI net EV positive AND conviction met" gate — it
  was screening out exactly the ambiguous names AI exists to resolve).
- **T3** (~$0.30) — Opus Pass 1 + Sonnet Pass 2 + Haiku stress. Ambiguity ≥
  `t3_min_ambiguity` AND `qualifies_for_t2_plus` AND budget allows.

Broker: T0 all 26 (default roster) first, sort by ambiguity, greedy allocate
T3→T2→T1 within $2.

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

# NEW SESSION BRIEF — diagnose & fix the 0-BUY engine (read this first)

You are picking up a multi-ticker swing-trading decision engine that has a
**0-BUY problem**: it runs the full roster and recommends nothing, repeatedly.
A recommendation tool that never recommends has failed at its one job. Your
mandate is to find WHY — from the code and the data, not from anyone's prior
conclusions — and fix it so the system becomes a genuinely superior,
AI-leveraged, institutional-grade swing engine that still costs ≤ $2 per
complete run.

This brief is deliberately light on answers. It tells you the operator's
concerns, the rules of engagement, and where the bodies are likely buried. It
does NOT hand you a ranked defect list — deriving that yourself, from scratch,
is the job. Where a prior session reached a conclusion, it is quarantined at
the very bottom under "PRIOR-SESSION CLAIMS (unverified — do not anchor)."

---

## 1. The operator's problem statement (his words, distilled)

1. **The engine is engineered to refuse, not to recommend.** Every gate biases
   toward "not confident enough." It's an expensive machine for saying no.
2. **It's calibrated for a trader who isn't him.** Friction models a blind
   market-order bot; mean reversion (which IS the dip-and-rally thesis) is OFF,
   so the math runs the OPPOSITE hypothesis from the strategy; the trend filter
   refuses the very oversold dips he wants to buy.
3. **A novice keeps out-diagnosing the machine.** The sophisticated quant +
   AI stack misses root causes a human catches by intuition, because it
   confirms its own self-consistency instead of asking if it's fit for purpose.
4. **He pays for intelligence that gets thrown away.** AI searches, collects,
   reasons — and the pipeline silently drops it. Wasted tokens AND a blinder
   engine.
5. **His time is the bottleneck.** Speculative round-trips, re-derivation, and
   re-walked dead ends all cost his day.

The deepest pain is BOTH the 0-BUY outcome AND the meta-problem: he cannot
trust a session to diagnose honestly without anchoring, drifting, or
reassuring him. Your job is to break that pattern.

---

## 2. Non-negotiable mindset (this is why you exist)

- **Nothing in the code or YAML is sacred-by-default.** Numbers in
  `config/diprally.yaml` and constants in `src/` are CALIBRATION, not law. A
  threshold that causes 0-BUY is a bug to interrogate, not a commandment to
  obey. The ONLY truly inviolable items are the architectural sacred decisions
  (#1-12, 15-17 in CLAUDE.md) — and even those you should understand before you
  trust. Calibration thresholds (#13 EV hurdle, #14 trend filter, #18 parabola)
  are explicitly recalibratable.
- **Assume nothing. Verify everything against code AND live/real data.** "The
  comment says X" is a hypothesis, not a fact. "The YAML has value Y" tells you
  what's loaded, not whether Y is correct for the goal.
- **Be honest, be adversarial, never reassure.** "I found nothing / the math
  checks out" is a failure state — it means you didn't dig. The deliverable is
  always a ranked list of what's WRONG or FRAGILE, reproduced on real inputs.
- **You may read and interrogate the ENTIRE codebase.** Do it if that's what
  honest diagnosis requires. Don't shortcut to a plausible story.
- **Volatility is the friend, not the risk.** See Mission in CLAUDE.md. Any
  instinct to "de-risk by refusing volatile names" is exactly the failure mode.
  The edge IS the volatility, harnessed.

---

## 3. Rules of engagement (from CLAUDE.md — honor them)

- **Approval gates**: NEVER commit/push without an explicit "go"; NEVER
  PR/merge to main without "merge to main". Each meaningful step = its own
  commit.
- **Diagnostics CANON (operator's time is the bottleneck)**: when you need a
  data fetch or a diagnostic, deliver it in ONE response — state the decision,
  state the exact data needed, give ONE copy-paste bash command (print HTTP
  status + raw body ≤1000 chars), and if an FMP endpoint may be plan-restricted,
  include a verbatim copy-pasteable FMP-support question. NO iterative
  "run this, then I'll tell you the next step."
- **Math-layer diagnostics**: numerical harness in the sandbox FIRST
  (synthesize realistic inputs, decompose by axis, find root cause with hard
  numbers), THEN one surgical fix + a regression test that fails-before /
  passes-after.
- **Audit protocol (MANDATORY here)**: falsify-first. State the null — "the
  engine CANNOT produce a BUY" — and try to FALSIFY it with a harness on
  realistic inputs before touching anything. Then: two questions (does code do
  what it documents? is what it documents correct for the goal?); trace the
  pipeline stage by stage; audit AI inputs vs an independent web-search ground
  truth; enumerate every silent-failure path; interrogate the objective
  function (EV) by decomposing upside vs downside.
- **Basis-point clarity**: every bps figure gets its % gloss, e.g. 100 bps (1%).
- **AI output must never be silently dropped** (hard constraint, recently
  added): every datum the AI searches/collects/generates must flow to the next
  stage AND persist to artifacts. A parsed-but-unused AI field is a defect.
- **$2 hard cap per complete run.** Don't propose a fix that blows the budget.
  More intelligence is good ONLY if the broker can still fit it under $2.

---

## 4. What this system IS (so you don't mis-target the fix)

- **Goal**: lock in gains from expected short-horizon RALLIES via dip→rally
  round-trips within **20 trading days**. NOT long-term investing. Enter on a
  defensible dip, exit into the rally, bank it.
- **Universe**: the `tickers` block in `config/diprally.yaml` (default daily
  roster) + `tickers_scratch` (runs only on explicit `--tickers`). Never assume
  a count — dump it (command in CLAUDE.md "Ticker universe" section).
- **Tiers**: T0 (math only, free, every ticker) → T1 Haiku → T2 Sonnet P1+P2
  → T3 Opus+Sonnet+Haiku. Broker greedy-allocates by ambiguity under $2.
- **Verdict waterfall** (priority order, in `src/engine.py`
  `_compute_verdict_state`): REFUSED-TREND → REFUSED-PARABOLA → REFUSED-METHOD
  → REFUSED-EV → WAIT → BELOW-THRESHOLD → NEGATIVE-EV → BUY. `verdict_state`
  IS the binding-gate label per ticker.

---

## 5. Repo map (where to look)

```
src/
  engine.py          orchestration of one ticker; verdict waterfall; gates live here
  math_utils.py      MC + PDE + closed-form; GARCH; enrichment_drift; EV; mean-reversion term
  signals.py         all signal_from_* (analyst, sector, peer_rs, fundamentals, catalyst, …); blend
  ai_layer.py        Pass 1 / Pass 2 / verification / stress; JSON parse; client init
  ai_tiers.py        tier→model resolution
  broker.py          $2 greedy allocation; forced-T2 safeguards
  ambiguity.py       5-component ambiguity score that gates AI access
  data_fetch.py      FMP + yfinance fallback; macro/sector/options fetches
  config.py          pydantic YAML loader → typed constants (NOT a place for values)
  orchestrator.py    two-phase batch driver; portfolio correlation gate
  reporter.py        text + HTML report; dashboard
config/diprally.yaml  ALL tunable values (sacred #17). Single source of truth.
tools/
  run.py             single-ticker CLI (python tools/run.py TICKER --tier T0|T1|T2|T3)
  orchestrate.py     batch CLI (python tools/orchestrate.py [--tickers …] [--dry-run] [--budget N])
  falsify_buy.py     EXISTING 0-BUY falsification harness — Pass A (free, T0 baseline) +
                     Pass B (real AI). START HERE to reproduce the anomaly.
tests/               ~60 test files — your regression net; respect it, extend it
docs/handover/_DEFERRED.md   compact backlog: 7 OPEN items + 1 REJECTED (don't re-litigate)
```

---

## 6. How to reproduce the anomaly (do this FIRST, before any fix)

- Free baseline (no AI spend), confirms the structural floor:
  `python tools/falsify_buy.py --baseline-only`
- Real pass (spends ≤ $2), tests whether AI catalysts unblock BUYs:
  `python tools/falsify_buy.py`
- Single ticker, math-only, to read the gate trace:
  `python tools/run.py LWLG --tier T0`
- Dry-run broker allocation across the roster (free):
  `python tools/orchestrate.py --dry-run`

Both require `FMP_API_KEY`; the real pass also needs `ANTHROPIC_API_KEY`.

---

## 7. Environment realities (verify, don't assume)

- **This container is freshly cloned and ephemeral.** As of writing,
  `output/` contains NO `round_trip_history_*.csv` — only
  `catalyst_audit_ledger.csv`. Several `_DEFERRED.md` items are gated on
  "~30 days of realized-outcome CSV"; that data is NOT present here. Do not
  assume historical outcome data exists — check, and if it doesn't, the
  data-gated items can't be closed empirically this session (say so plainly).
- **Anything worth keeping must be committed and pushed.** The container is
  reclaimed after inactivity.
- **Network access is governed by the environment policy.** FMP/Anthropic may
  or may not be reachable; if a fetch fails, that's a finding, not a dead end —
  fall back to a numerical harness on synthesized-but-realistic inputs.
- **Develop on the designated feature branch** the harness assigns; create it
  locally if needed; push only there; never to main without "merge to main".

---

## 8. Suggested first-pass plan (a scaffold, not a script — adapt it)

1. Reproduce: run `falsify_buy.py --baseline-only`, capture the gate histogram.
   Which gate is the binding constraint per ticker? Measure, don't guess.
2. For the dominant gate(s), open a numerical harness: synthesize realistic
   inputs for an EXTREME and a HIGH name, decompose the EV / conviction math by
   axis (upside cap vs downside tail; P(dip) vs P(rally|dip)), and find whether
   the refusal is correct or an artifact.
3. Audit one ticker's ACTUAL AI catalysts against an independent web search —
   is the AI input even right? Garbage-in invalidates the downstream.
4. Trace the AI-persistence path end to end: does every field the AI is prompted
   for actually reach the blend AND the CSV/cache? (Known suspect: Pass 2 vs
   factor_bias — see _DEFERRED D-W2-19.)
5. Interrogate the calibration the operator flagged: friction bps vs his real
   (discretionary, limit-order) execution; mean-reversion default OFF vs the
   dip-rally thesis; the GLOBAL (non-σ-class) trend filter at high vol.
6. Produce a RANKED defect list by impact on 0-BUY, each with a harness that
   reproduces it on real inputs. THEN propose surgical fixes + regression tests.
   Get a "go" before committing.

---

## 9. Discrepancies already caught (start trusting nothing — these are examples)

- `tools/orchestrate.py` `--max-parallel` default is **4** (PR #89), but
  `_DEFERRED.md` D-OPS-1 still describes it as 2 and asks to bump to 3. The
  deferred note is stale on this point. (Illustrates: even the backlog drifts —
  verify against code.)
- CLAUDE.md previously carried stale conviction (0.75), parabola (+100%), and a
  60-day horizon — all corrected this session against the live YAML. Treat any
  remaining prose number as suspect until you've matched it to the YAML/code.

---

## 10. PRIOR-SESSION CLAIMS (UNVERIFIED — do NOT anchor; falsify independently)

A previous audit produced the hypotheses below. They are recorded ONLY so you
don't waste time re-discovering them blind — but they may be WRONG, partial, or
mis-prioritized. Treat each as a null to falsify with your own harness. Do not
cite them as established. If your independent diagnosis disagrees, your
harness-backed result wins.

- Claimed structural BUY-suppressors: (a) mean reversion never passed by the
  orchestrator (pure GBM only); (b) `enrichment_drift` momentum term scaled by
  `/1000` making it inert; (c) GARCH fallback uses a spike-sensitive 90-bar
  variance that inflates σ on dip candidates.
- Claimed calibration conflicts with the operator's profile: friction bps too
  high for a discretionary limit-order trader; trend filter -25% is GLOBAL, not
  σ-class-adjusted, so it's <1σ on EXTREME names and refuses normal dips.
- Claimed correct-as-designed (the prior session said these are NOT bugs):
  parabola filter direction; three-method cross-check; method-disagreement
  refusal; two-sided catalyst drift = 0.0 (see _DEFERRED D-2026-1, REJECTED).
- The prior session ALSO produced several confident findings that turned out
  WRONG on inspection (e.g., "trend filter not implemented" — it is, in
  engine.py; "EXTREME 0.55 conviction impossible" — that was the OLD 0.75).
  This is your cautionary tale: confident ≠ correct. Reproduce or discard.

Your north star: read the code, interrogate the data, assume nothing, stay
honest, keep it ≤ $2/run, and make the engine actually buy the dips it was
built to catch.

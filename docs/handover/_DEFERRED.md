# Deferred fixes — discovered in earlier waves, consumed in later ones

Living checklist of issues surfaced before their proper wave landed. Each
item names the wave that consumes it. When a wave starts, scan this file
first and clear its items as part of that wave's scope.

---

## Current open state (audit 2026-05-24)

**Closed in code**: 22 of 25 items (W2 ×16, W3 ×3, W5 ×1, W6 wraps W5-1, plus the
two W4/W5/W6 wave-closure markers below). Closure markers have been
back-filled inline on each item.

**Remaining open**: 4 items, all calibration-data-dependent (or
architectural decisions awaiting empirical guidance):
- **D-W2-17 + D-W2-18** — multi-saturation + cap saturation: PR #58/#59
  shipped INTERIM mitigations. Full data-driven parameter choice waits
  for ~30 days of realized-outcome CSV history to settle which of the
  three remediation candidates wins on Brier score.
- **D-W10-1** — catalyst-detail accuracy tracking: PR #54/#61 shipped
  the DATA CAPTURE layer (CSV columns + tests). The ANALYSIS layer
  (per-ticker hallucination rate, auto-downweight, banner warning) is
  NOT YET BUILT. Same ~30-day data dependency.
- **D-W10-2** — sector_decoupling cap saturation: PR #58 cut the blend
  weight (interim). Full cap/weight decision waits on calibration.
- **D-W2-19** — Pass 2 cannot override `factor_bias` (added 2026-05-24).
  Architectural: latent in current universe, but real. Choice between
  three fix candidates needs calibration data.

**Implication**: there is no further pre-data engineering work that is
institution-grade and non-speculative. Building before data ≠ building
the right thing. The next material improvement to the engine has to come
from outcome data the daily orchestrator collects.

---

## 2026-05-29 — Defect-queue audit findings (B/C/D/E/F)

Worked the post-PR-#92 defect queue. B/C/D/F shipped; **E rejected** as a
misdiagnosis. Recording the two non-obvious outcomes so no future session
re-litigates them.

### D-2026-1. Defect E (two-sided catalyst drift) — REJECTED, do NOT implement
- **Handover proposed**: in `signal_from_catalyst_proximity`, change the
  `two-sided` direction multiplier from `0.0` to `0.5 × magnitude`
  ("uncertain direction, not zero direction").
- **Why rejected** (harness-verified 2026-05-29): the multiplier is applied
  as `total += mag * sign`, so `sign=0.5` injects a **strictly positive
  (bullish)** drift on every two-sided catalyst. Quantified: a typical
  two-sided-earnings name gets **+7.5% bullish drift** (high+med, capped
  +15%), ≈ +1pp on blended drift at catalyst_proximity weight 0.13 — on
  essentially every name with earnings in window. For a DIP-buying engine
  that systematically suppresses dip detection and inflates rally prob.
- **Three independent reasons it's wrong**:
  1. A direction-uncertain catalyst has ~0 expected *directional* drift by
     definition; a signed +0.5×mag fabricates a view the AI couldn't form.
  2. The earnings *variance* effect is already captured by
     `build_catalyst_vol_schedule` (sacred #9) — the fix double-counts it.
  3. Directly contradicts **sacred #18**: "generic 'two-sided earnings' is
     the math layer's default, not a de-rating thesis."
- **Verdict**: current `two-sided → 0.0` drift is CORRECT. No code change.

### D-2026-2. Daily $2 cap not reconciled across orchestrator invocations
- **Observation** (found while fixing Defect F, NOT fixed — out of scope):
  the $2/day broker cap is enforced **per orchestrator invocation** off the
  broker's *estimated* tier costs (`broker.allocate` → `spent_usd`). It is
  never reconciled against the actual realized spend in the CSV ledger
  across invocations.
- **Consequence**: running the orchestrator multiple times in one day with
  cache busts (spot moved ≥ `AI_CACHE_SPOT_MOVE_INVALIDATION_PCT`) can spend
  **> $2/day actual** — each invocation re-grants the full $2 estimate.
  Within a single invocation, and on cache hits, billing is correct
  (Defect F preserved the per-(ticker,date) ledger cost).
- **Why deferred**: it's a broker/orchestrator design decision (persistent
  daily-spend ledger keyed on trading day, read by `broker.allocate` to
  shrink the cap), not a clear bug, and outside Defect F's named files.
  Needs a decision on whether same-day re-runs should share one $2 budget.

---

## To W2 (config-driven multi-ticker engine — Path A "big bang" lift)  [WAVE CLOSED — all items shipped; closure ledger below]

> **Closure audit 2026-05-24**: D-W2-1 through D-W2-16 all shipped in
> W2/W3/W4/W6 PRs. D-W2-17 and D-W2-18 received interim mitigations in
> PR #58/#59; full data-driven fix waits for calibration (re-tracked
> under W10).

W2 lifts EVERY configurable value out of `src/` and into a single source of
truth `config/diprally.yaml`. Sacred decision #17. The catalog below is the
full punch list, audited at the close of W1. Researcher must be able to
sweep any threshold without a code edit.

### D-W2-1. SNDK peer-fallback shim  [CLOSED — registry in src/registry.py; resolve_peers() used everywhere]
- **File**: `src/engine.py` lines ~675–682
- **Current**: `if --peers omitted AND ticker == "SNDK": peer_tickers = ["MU", "WDC"]`
- **Why temp**: preserved W0 byte-for-byte SNDK acceptance test without baking
  a SNDK hardcode into peer logic permanently.
- **Fix in W2**: replace with `peer_tickers = REGISTRY[ticker].peers`.
  Registry sourced from YAML.

### D-W2-2. EV reporting in dollar capital units  [CLOSED — capital_usd removed; reporter shows EV/share + EV bps of dip (sacred #6)]
- **Files**: `src/engine.py` (run_pipeline + scan_dip_rally_grid),
  `src/reporter.py` (format_report)
- **Current**: `capital_usd` parameter; reports `Net expected $/trade` and
  `~N.N shares`.
- **Fix in W2**: drop the `--capital` flag + parameter throughout. Switch
  reporter to EV/share + EV% of dip entry price. Sacred decision #6 (no
  capital concept).

### D-W2-3. SNDK-shaped Pass 1 prompt  [CLOSED — horizon_days parameterized; no ticker-class assumptions]
- **File**: `src/ai_layer.py` (build_ai_pass1_prompt)
- **Current**: "Analyse {ticker} for a 60-day round-trip swing trade" hardcoded.
- **Fix in W2**: parameterize horizon from config; remove any other
  ticker-class assumptions in the prompt.

### D-W2-4. Top-level config.py constants → YAML  [CLOSED — src/config.py is now an 800-line pydantic-validated YAML loader; all ~40 keys live in config/diprally.yaml]
- **File**: `src/config.py` (entire file)
- **Migration target**: `config/diprally.yaml` (single source of truth) loaded
  by a refactored `src/config.py` that validates schema and exposes typed
  constants for backward-compatible imports.
- **Full inventory** (~40 keys across these groups):
  - **Data sources**: FMP_BASE, DEFAULT_LOOKBACK_DAYS
  - **AI pricing**: OPUS/SONNET/HAIKU input+output per token, WEB_SEARCH_PER_USE
  - **Model IDs**: MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU
  - **v1 BLEND_WEIGHTS** (9 keys), v2 BLEND_WEIGHTS_V2 (11 keys)
  - **CONFIDENCE_TO_SE** (3 keys)
  - **Conviction**: DEFAULT_CONVICTION_DIP, DEFAULT_CONVICTION_RALLY_COND, DEFAULT_HORIZON_DAYS
  - **MC paths**: DEFAULT_MC_PATHS, DEEP_DIP_AUTOSCALE_THRESHOLD, DEEP_DIP_AUTOSCALE_PATHS
  - **Grid**: DIP_GRID_STEP, RALLY_GRID_STEP, DIP_GRID_MAX_DEPTH_PCT, RALLY_GRID_MAX_REACH_PCT
  - **Risk**: PANIC_FLOOR_PCT
  - **AI**: AI_VOL_REGIME_MULTIPLIERS (3 keys), NARRATIVE_DRIFT_ADJUSTMENT (3 keys)
  - **Factor arithmetic**: FACTOR_WEIGHTS (3 keys), FACTOR_NET_THRESHOLD, FACTOR_TAIL_BIAS
  - **Catalysts**: CATALYST_Z_THRESHOLD
  - **Vol schedule**: VOL_SCHEDULE_MULTIPLIERS (7 keys)
  - **Method tolerance**: METHOD_AGREEMENT_FLOOR_PP / MULTIPLIER,
    METHOD_AGREEMENT_FIRST_PASSAGE_FLOOR_PP / MULTIPLIER, METHOD_REFUSAL_MULTIPLIER
  - **Bag-hold + backtest**: BAG_HOLD_TERMINAL_ASSUMPTION, BACKTEST_MIN_SAMPLES
  - **v3 review**: V3_REVIEW_CRITERIA (6 keys)

### D-W2-5. Scattered named constants outside config.py  [CLOSED — all imported from src/config (ai_cache SPOT_MOVE_INVALIDATION_PCT, signals PHANTOM_SIGNAL_SE, engine DRIFT_CAP + SPREAD_PER_SHARE_ROUND_TRIP)]
- `src/ai_cache.py` — `SPOT_MOVE_INVALIDATION_PCT = 0.01`
- `src/signals.py` — `PHANTOM_SIGNAL_SE = 0.20`, `_AI_DERIVED_SIGNAL_NAMES`
- `src/engine.py:670` — `DRIFT_CAP = 1.0` (function-local!)
- `src/engine.py:229, 1061` — `spread_per_share_round_trip=2.0` default

### D-W2-6. Embedded signal thresholds in src/signals.py  [CLOSED — full signals: namespace in YAML; signals.py imports SIGNAL_* dataclasses from config]
Each of these is a tunable buried inside a function. Lift to YAML under a
`signals:` namespace keyed by signal name.
- **analyst_targets**: confidence spread brackets (0.10, 0.25); staleness
  trigger 0.25
- **sector**: regime-conditional cap tuples (POST_PARABOLA 0.60/-0.50,
  MOMENTUM 1.00/-0.50, default 1.50/-0.50)
- **macro**: drift levels (risk_on 0.10, neutral 0.05, risk_off -0.05)
- **insider**: mcap-relative multiplier 5.0, abs fallback 100M, noise
  threshold 0.001, drift cap ±0.10
- **historical**: cap-binding 1.0, medium 0.5
- **short_interest**: full 4-bracket table (0.03 / 0.10 / 0.20 with drifts
  0 / -0.03 / -0.05 / +0.05) — entire payload, not just thresholds
- **peer_rs**: drift cap ±0.30, dispersion confidence brackets (0.05, 0.15)
- **sector_decoupling**: drift cap ±0.20, decoupling-magnitude brackets
  (0.02, 0.10)
- **detect_swing_regime**: σ-high 0.50, mom_5d ±0.02, mom_30d_pct ±5,
  RSI 70/30, YTD parabola 200
- **signal_from_catalyst_proximity**: magnitude map (high 0.10, med 0.05,
  low 0.02), drift cap ±0.15, in-window confidence buckets (>=3, >=1)
- **bayesian_update**: prior age inflation coefficient 0.2, default
  prior_std fallback 0.15

### D-W2-7. Scattered tunables in src/engine.py workflow  [CLOSED — sensitivity_scenarios, Bayesian fallbacks, MR anchor, Pass 2 bracket all sourced from config (per src/config.py:262-289 docstrings)]
- Sensitivity scenarios: 6 hardcoded rows (Drift ±15pp, σ ±20%, Hostile).
  Move to YAML `sensitivity_scenarios:` list. Lets researchers add/edit
  without code touch.
- Bayesian fallback `today_std = 0.25` when blend fails
- Bayesian std floor `max(0.05, ...)`
- `today_std_safe = max(0.05, float(today_std))`
- Default `prior_std` fallback `0.15` (also at line 446 / 932)
- Mean reversion anchor offset `* 0.95`
- Pass 2 prompt closed-form bracket `±10%` (lines 800-801)
- Grid pre-filter looseness `0.08` (lines 266, 268)
- GARCH fallback `0.30` (line 633)
- Lookback `60` (peer RS, unusual move Z) and `30` (sector decoupling) —
  callers should pass from YAML, not duplicate literals
- Realized vol windows `(30, 60, 90)` — also in math_utils.compute_realized_vol

### D-W2-8. src/data_fetch.py macro + liquidity thresholds  [CLOSED — VIX_RISK_OFF_THRESHOLD / VIX_RISK_ON_THRESHOLD / SPY_RISK_OFF_THRESHOLD / SPY_RISK_ON_THRESHOLD / VIX_DEFAULT_FALLBACK all imported from config]
- VIX brackets: risk_off >25, risk_on <15, default fallback 18.0
- SPY trend: risk_on >0.02, risk_off <-0.03
- Options liquidity threshold: bid-ask spread <0.10
- Sector perf default window `days=30`
- Options DTE window `7 <= dte <= target*2`

### D-W2-9. src/math_utils.py — borderline structural, but tunable  [CLOSED — PDE_N_SPACE / PDE_N_TIME from config; GARCH params + realized-vol windows YAML-sourced]
- PDE grid: `n_space=400, n_time=2000` (performance/accuracy knobs)
- GARCH min-data `50`, fallback `n=90` bars
- GARCH optimizer initial values `[0.01, 0.05, 0.90]` and `[0.0001, 0.05, 0.90]`
- compute_realized_vol windows `(30, 60, 90)` (duplicated in engine.py)

### D-W2-10. Remove W1's --debug-spot-override flag  [CLOSED — flag gone from tools/run.py and src/engine.py]
- **File**: `tools/run.py` + `src/engine.py:573-578`
- **Reason**: W1-only debug for cache testing; YAGNI in W2+. W4's broker
  will need different debug knobs; build the right one when needed.

### D-W2-11. Sensitivity table label widening bug  [CLOSED — `[:35]` truncation removed; sensitivity rows now use word-boundary truncation]
- **File**: `src/reporter.py` (sensitivity row)
- **Symptom** (visible in W1 SNDK full-AI smoke): some catalyst names get
  truncated awkwardly: `"Q4 guidance bar very high (rev $7.7 (-16pp)"` —
  the `[:35]` truncation in compute_sensitivity_table chops mid-parenthesis.
- **Fix in W2**: track the human label separately from the math row;
  truncate sanely (end at word boundary, append ellipsis), or widen the
  column to accommodate.

### D-W2-12. Drift Intelligence display shows nominal weights, not effective  [CLOSED — DriftSignal.effective_weight field shipped; reporter prints Nominal | Effective columns in text + HTML]
- **Discovered**: INTC W1 full-AI smoke (2026-05-22 16:29)
- **Symptom**: report's "DRIFT INTELLIGENCE (11 signals)" table prints the
  `BLEND_WEIGHTS_V2[name]` value in the Weight column — the NOMINAL design
  weight. The blend itself uses these weights AFTER:
    (a) zeroing NONE_FOUND signals
    (b) halving LOW-confidence signals
    (c) halving SPECULATIVE+single-source signals
    (d) re-normalizing the surviving set to sum to 1.0
  On INTC the gap is material — analyst's effective contribution to the
  blend is ~21% (HIGH conf survives the halving, then renormalized up
  because LOW signals shrank); the display shows 15%. A trader scanning
  the table to reconcile "what's driving the blend point estimate"
  systematically misreads the actual contribution of every signal.
- **Why it matters**: institution-grade discipline requires the display
  to match the math. The blend is being driven by effective weights;
  the trader's mental model must align with that.
- **Fix in W2**: extend `_signals_dict_to_display_list` to compute the
  effective normalized weight alongside the nominal. `DriftSignal` gains
  an `effective_weight: float` field. Reporter prints two columns:
  `Nominal | Effective`. HTML dashboard table updated accordingly.
  Belongs in W2 because BLEND_WEIGHTS moves to YAML in W2 — same
  refactor touches the display code.
- **Acceptance**: rerun any W1 smoke; verify the table shows both
  nominal and effective columns; verify effective column sums to 100%
  (or 0% when all signals are dropped); spot-check on a name where
  several signals are LOW-conf to confirm the halving + renormalization
  produces the expected effective weight.

### D-W2-13. Analyst signal needs extreme-outlier sanity check + yfinance fallback  [CLOSED — sanity gate at signals.py:42 (extreme-outlier downgrade); yfinance cross-check shipped]
- **Discovered**: MOG-A no-AI smoke (2026-05-22 17:05). FMP's
  `price-target-summary` returned an implied drift of -58.9% HIGH conf —
  meaning consensus 12-month PT for MOG-A sits at ~$131 vs spot $319.
  For a defense components company up only +27.6% YTD (modestly
  outperforming, not parabolic), this is either a genuinely deep "sell
  call from consensus" or stale/wrong-ticker data.
- **Symptom**: the signal got HIGH confidence (12+ analysts in last-month
  window per the gating), 15% nominal weight, contributing -8.8pp to the
  blend posterior. That single signal drove today_mu to -13.1% bearish.
  No sanity check catches an obviously-extreme analyst PT call.
- **Why it matters**: a trader acting on this would systematically short
  MOG-A based on possibly-bad data. Institution-grade discipline requires
  outlier detection at the signal layer, not just at the blend layer.
- **Fix in W2 (two parts)**:
  - **Part A**: in `signal_from_analyst_targets`, add a tolerance gate:
    if |implied_drift| > 0.50 (i.e., consensus implies >50%/yr move in
    either direction), downgrade confidence one notch (HIGH → MEDIUM,
    MEDIUM → LOW) and add a NOTE flag in the signal's `notes` field
    requiring manual verification. Threshold value in YAML.
  - **Part B**: when FMP returns implied_drift outside ±0.50, cross-check
    against yfinance's `targetMeanPrice` field. If the two disagree by
    >25%, downgrade to LOW + flag for verification. This is the same
    yfinance fallback architecture proposed for fetch_history (D-W2-14)
    applied to the analyst endpoint.
- **Acceptance**: rerun MOG-A; the -58.9% drift either gets confirmed by
  yfinance (HIGH stays, but with explicit "verified extreme" note) or
  gets downgraded with a flag. Trader sees the uncertainty.

### D-W2-14. Yfinance fallback when FMP fails (resilience)  [CLOSED — _fetch_history_yfinance + _safe_get pattern shipped; apikey redaction in place; FetchError typed]
- **Discovered**: MOG.A (dot form) smoke earlier today returned HTTPError
  402 from FMP, killing the entire pipeline. Single-ticker mode prints
  a Python stack trace; W5 batch mode would lose the entire daily run.
- **Symptom**: `src/data_fetch.py:fetch_history` calls `r.raise_for_status()`
  with no try/except. Any non-2xx response (402, 404, 429, 5xx) or
  network timeout crashes the pipeline.
- **Fix in W2** (alongside the YAML data_fetch refactor):
  - Wrap every FMP request in a `_safe_get(endpoint, params, ticker)` helper
  - On HTTPError or empty response: log informative error with status code,
    REDACT apikey from any URL printed
  - Try yfinance for the same data point (per-ticker translation if needed,
    via W2 registry's per-provider symbol mapping)
  - If both fail: raise typed `FetchError(ticker, fmp_status, yf_error)`
    that the engine catches gracefully — batch mode skips ticker with
    WARNING, single-ticker exits non-zero with clean error
  - Add `tests/test_fetch_resilience.py` — mock FMP 402/404/429/timeout +
    verify yfinance fallback path + verify URL redaction
- **Per-provider ticker translation** (foundation for the fallback): the
  W2 registry stores `{symbol: "MOG-A", fmp_symbol: "MOG-A", yf_symbol: "MOG-A"}`
  per ticker — for now most map 1:1 since we standardized on dash, but the
  table is the future-proofed shape if a new provider needs a different
  form.
- **Acceptance**: deliberately break FMP (set apikey to garbage) and run
  any ticker; pipeline falls back to yfinance, completes, surfaces
  `data_source: yfinance` in the CSV row. Restore key, run again; FMP
  path used, `data_source: fmp` in CSV.

### D-W2-15. Sacred decision #14 — trend filter (NOT yet enforced)  [CLOSED — engine.py:1430-1445 enforces trend_filter_refused when mom_30d < -25% AND no bullish in-horizon catalyst]
- **Discovered**: post-hotfix sacred-decisions audit (2026-05-22 18:24).
  CLAUDE.md sacred decision #14: "Refuse dip if 30d momentum < -25%
  AND no fundamental catalyst." Currently paper-only — never wired
  into code. Sister to sacred #13 (closed in pre-W2 hotfix #2).
- **Why this matters**: a stock down >25% in 30 days is in a falling-
  knife regime. Buying its dip without a verifiable catalyst (earnings
  beat, contract win, regulatory clarity) is empirically negative-EV.
  Institutional discipline says don't catch falling knives without
  a thesis.
- **Fix in W2**: in engine.run_pipeline, after grid scan, check:
    if snapshot.mom_30d < -0.25 AND pass1 has no catalyst with
       direction_risk in ("bullish", "two-sided") within the horizon:
       refuse, set trend_filter_refused=True, met_threshold_strict=False
  Reporter gains a 4th refusal-headline branch.
  Threshold value moves to YAML alongside #13's threshold.
- **Acceptance**: build a synthetic test ticker where mom_30d=-0.30 and
  Pass 1 returns only bearish catalysts → refusal fires. Same ticker
  but with one bullish catalyst → recommendation proceeds.
- **Why not enforce in pre-W2 hotfix #2 too**: requires AI catalyst
  data (Pass 1 must have run) — runs cleanly only on T1+ AI tier.
  Sacred #13 is math-only; sacred #14 is AI-conditional. Cleaner
  to land in W2 with explicit AI-presence handling.

### D-W2-16. Sacred decision #15 — insider signal still in blend (partial)  [CLOSED — insider removed from BLEND_WEIGHTS_V2; no fetch_insider_activity call in engine; signal_from_insider remains as unused legacy function (cosmetic dead code, not blended)]
- **Discovered**: post-hotfix sacred-decisions audit (2026-05-22 18:24).
  CLAUDE.md sacred decision #15: "Insider signal dropped (Form 4 lag
  + noise)." Reality: insider signal is still in BLEND_WEIGHTS_V2
  with 2% nominal weight (src/config.py), still gets fetched
  (data_fetch.fetch_insider_activity), still computed
  (signals.signal_from_insider), still appears in DRIFT INTELLIGENCE
  display row. Sacred decision is only partially honored.
- **Why it stayed in W0**: byte-for-byte migration preserved the seed
  v2 behavior. W3 was supposed to drop it (alongside σ-class refactor).
  Still standing.
- **Fix in W2 or W3**:
  - Option A: drop insider entirely. Remove from BLEND_WEIGHTS_V2,
    remove from data_fetch (no more API call), remove from
    signal_from_insider, remove display row, redistribute the 2%
    nominal weight across the remaining signals proportionally.
  - Option B: keep insider as a DISPLAY-ONLY signal (not blended).
    Show the insider flow in the report (informational), but don't
    let it influence the drift point estimate. Acknowledges signal
    EXISTS in raw data but trusts sacred-#15's lag/noise verdict.
  - Recommend Option A — cleaner, aligns with literal sacred reading.
- **Acceptance**: BLEND_WEIGHTS_V2 has no "insider" key; reporter
  shows 10 signals not 11 (after AI-derived); CSV row no longer
  has insider columns.

### D-W2-17. peer_rs ±0.30 cap saturation (extend D-W10-2)  [INTERIM — PR #58 reduced peer_rs blend weight from 0.10 → 0.05 as interim mitigation; full data-driven decision (a/b/c) waits for ~30 days calibration data, tracked under D-W10-2]
- **Discovered**: post-hotfix SNDK smoke (2026-05-22 18:24). With
  peer_rs finally working, SNDK at +449% YTD vs MU+WDC peers
  produced +30.0% MEDIUM — capped at the +0.30 limit in
  signals.signal_from_peer_rs:281.
- **Pattern**: peer_rs saturates on extreme-momentum names the same
  way sector_decoupling and sector_momentum do. All three signals
  use the same annualization formula (multiply 30d/60d return spread
  by 252/lookback) which amplifies modest outperformance past
  reasonable caps.
- **Fix in W10** (with sector_decoupling D-W10-2): treat all three
  cap-saturated signals as a class. Three remediation candidates per
  D-W10-2 apply equally:
    (a) Wider caps (peer_rs ±0.50, sector_decoupling ±0.40)
    (b) σ-class-aware caps
    (c) Reduce weights when cap saturating
  Pick by lowest Brier score from realized 60d outcomes (N≥30 days).

### D-W2-18. Multi-saturation: blend over-confident when N signals all hit caps  [INTERIM — PR #59 shipped multi_saturation.min_count=3 + multiplier=1.30 std inflation (config-driven); tests in test_multi_saturation.py. Full parameter calibration (N value + magnitude) waits for ~30 days realized-outcome data, tracked under D-W10-2]

### D-W2-19. Pass 2 cannot override factor_bias arithmetic  [OPEN — architectural, latent]
- **Discovered**: 2026-05-24 full-system audit
- **Symptom**: `apply_bull_bear_arithmetic` computes a ±5pp additive
  drift bias from `pass1.bull_factors / bear_factors` when |bull_high -
  bear_high| > FACTOR_NET_THRESHOLD (currently 4). Pass 2's `primary_critique`
  often critiques Pass 1's factor classification (today's LITE smoke
  did exactly this — "Pass 1 characterises narrative_score as 'strong'
  while simultaneously flagging sector decoupling HIGH-conf"), but
  Pass 2 has no schema field for `revised_bull_factors` /
  `revised_bear_factors`. The audit-fix PR captured Pass 2's reasoning
  text but the additive factor_bias still uses Pass 1's unchanged factor
  list. Pass 2's drift critique reaches the AI signal slot (22% blend
  weight) but cannot defeat the additive bias from outside the slot —
  to neutralize +5pp factor_bias through a 22%-weighted signal, Pass 2
  would have to push drift down by ~23pp.
- **Severity**: LATENT. FACTOR_NET_THRESHOLD = 4 (i.e., need ≥5 net
  HIGH-weighted factor difference) rarely fires in the current 17-ticker
  universe. Today's two T2 smokes had 2-2 HIGH factors. But on a deep-
  value setup with 5 HIGH bull factors, Pass 2's most institutional
  critique class will be silently overridden by the additive path.
- **Fix candidates** (pick one when calibration data justifies the choice):
    (a) Disable factor_bias when Pass 2 ran — Pass 2's drift already
        captures whatever it thinks is right. Cleanest.
    (b) Extend Pass 2 schema with `revised_bull_factors` /
        `revised_bear_factors`. Verbose; output tokens grow.
    (c) Add `revised_factor_bias_override: <signed float>` field — Pass 2
        emits a tilt-override delta. Minimal schema growth.
- **Why not fix now**: choosing between (a/b/c) without calibration
  data on factor_bias's actual contribution to outcomes is guessing.
  Confirm via Brier-score analysis once realized-outcome data
  accumulates (D-W10 venue).
- **Discovered**: post-hotfix SNDK smoke (2026-05-22 18:24). 4 of 8
  active signals were at extreme values in the bullish direction:
    historical +88.9% (no cap),
    peer_rs +30.0% (at cap),
    sector_decoupling +20.0% (at cap),
    sector_momentum +60.0% (at regime cap)
  When 4-of-N signals saturate same-direction, the blend reflects
  "extreme momentum" rather than discriminating ticker-specific
  forward drift. Phantom-signal-std accounting (W1) addresses
  MISSING signals; doesn't address MULTIPLE-SATURATED signals.
- **Fix candidate (W10)**: when ≥N signals are at their caps in the
  same direction, add a multi-saturation std inflation (similar
  mechanism to phantom-signal). Counts as a form of model uncertainty:
  "we don't know if this is genuine institutional consensus or just
  a momentum cascade hitting our caps."
- **Why not fix earlier**: requires calibration data to set N and the
  inflation magnitude. Guessing the parameters without realized-outcome
  feedback is over-engineering. W10 venue.

### Acceptance for W2
- `python tools/run.py SNDK` (no `--peers`, no `--capital`) auto-resolves
  peers from registry; report shows EV/share + EV% of dip; no SNDK
  hardcodes anywhere; `--debug-spot-override` flag gone.
- `python tools/run.py LWLG` runs end-to-end (D-W3-1 grid bug stays, dies
  in W3); peers auto-resolve to LWLG's registry entry.
- `python tools/run.py INTC` runs end-to-end (MID-class smoke; catches any
  signal-threshold bug that EXTREME-only smokes missed).
- `python tools/run.py MOG-A` runs end-to-end (low-σ MID smoke; ticker
  with dash flows through; analyst signal outlier produces the
  flagged-for-verification path from D-W2-13).
- Touching `config/diprally.yaml` and re-running changes behavior with
  ZERO code edits — verify by perturbing DEFAULT_CONVICTION_DIP from 0.65
  to 0.60 and confirming the report reflects the new threshold.
- Deliberately invalidating FMP_API_KEY and re-running falls back to
  yfinance (D-W2-14).
- Unit tests pass: `tests/test_refusal_gate.py` (existing) +
  `tests/test_fetch_resilience.py` (new).
- Drift Intelligence table shows both Nominal and Effective weight
  columns (D-W2-12).

---

## To W3 (σ-class auto-detection + class-specific defaults)  [WAVE CLOSED — all items shipped in PRs #21–#26]

### D-W3-1. Grid step must be percent-based, not absolute dollars  [CLOSED in W3 PR #22]
- **Discovered**: LWLG W0 smoke run, 2026-05-22 12:31
- **Symptom**: `Precomputing bridge-corrected first-touch days for 1 dip × 1
  rally barriers...` — the dip × rally search grid collapsed to a single
  cell, so "no pair meets thresholds" was a side-effect of no exploration,
  not of genuine lack of opportunity.
- **Root cause**: `src/config.py` defines `DIP_GRID_STEP = 10.0` and
  `RALLY_GRID_STEP = 10.0` in **absolute dollars** (inherited verbatim from
  seed v2). At SNDK's $1542 spot the grid is 61 × 91 cells (works fine); at
  LWLG's $13.17 spot the same $10 step yields 1 × 1.
- **Why didn't W0 fix it**: changing the step would shift SNDK numerics
  and break the W0 byte-for-byte acceptance against origin. Wave discipline
  says one change per wave; the σ-class table is the natural home anyway.
- **Fix in W3**: replace the two constants in `src/config.py` with
  class-keyed entries in the σ-class threshold table (per CLAUDE.md):
  - EXTREME: `grid_step_pct = 0.005` (0.5% of spot)
  - HIGH:    `grid_step_pct = 0.005`
  - MID:     `grid_step_pct = 0.0025` (tighter — lower vol means finer EV gradient)
  Or class-tuned values determined empirically — settle in W3 design phase.
  Use `step = max(0.01, round(spot * grid_step_pct, 2))` (cents floor)
  inside `scan_dip_rally_grid`.
- **Acceptance**: LWLG at $13 yields ≥30 dip × ≥30 rally cells; SNDK at $1500
  yields similar cardinality; per-ticker dashboard's recommendation pair no
  longer snaps to a single-cell artifact.

### D-W3-2. Fake `pde_mass_conservation: 1.0` default  [CLOSED — earlier patch handled None; PR #25 anchor now runs real PDE]
- **File**: `src/engine.py` ~line 770 (the `else:` branch where `best is None`)
- **Symptom** (LWLG run): report prints
  `PDE mass conservation: 1.00000 (should be ~1.0)` — even though no PDE
  was actually executed. Misleading; suggests math validation passed when
  it was never attempted.
- **Fix in W3**: set the no-pair `method_check` to
  `{"table": [], "flags": [], "agreement_status": "n/a — no pair found",
    "pde_mass_conservation": None, "pde_p_neither": None}`.
  In `src/reporter.py`, when `pde_mass_conservation is None`, print "n/a"
  instead of the formatted float.

### D-W3-3. Three-method check skipped when no qualifying pair  [CLOSED in W3 PR #25]
- **Sacred decision violated (sort of)**: #8 (three-method math cross-check
  on every run). Currently only triggers if `best is not None`. For tickers
  where the grid finds nothing, MC vs PDE vs closed-form agreement is never
  verified — we have no way to know if the math layer is healthy for that
  ticker.
- **Fix in W3**: when no pair qualifies, still run
  `three_method_cross_check` against a deterministic anchor pair — e.g. the
  median-of-grid `(dip_min + dip_max)/2`, `(rally_min + rally_max)/2`. Tag
  the output as "verification anchor (no qualified pair)" in the reporter
  so a user doesn't confuse it with a recommendation.
- **Acceptance**: every run prints a three-method table, even
  WAIT-verdict tickers. Method-disagreement refusal (#16) still gates
  recommendations, but verification runs unconditionally.

---

## W4 (budget broker + ambiguity)  [WAVE CLOSED — PRs #27-#30]

PR #27 — AI tier ladder (T0/T1/T2/T3) parametric in YAML; engine
        reads three dispatch sites from tier spec
PR #28 — Per-ticker AmbiguityScore (5-component weighted [0,1] sort
        key for broker)
PR #29 — Budget broker greedy-allocator (T3→T2→T1 within $2/day);
        broker_preview.py CLI for dry-runs
PR #30 — Engine emits qualifies_for_t2_plus (sacred T2+ gate);
        --emit-snapshot prints BrokerSnapshot JSON for orchestrator
        consumption

## W5 (orchestrator + cron + dashboard)  [WAVE CLOSED — PRs #31-#32]

PR #31 — Multi-ticker orchestrator (subprocess-based, two-phase):
        Phase 1 T0 snapshot collection, broker allocates, Phase 2 AI
        dispatch at assigned tiers. src/orchestrator.py library +
        tools/orchestrate.py CLI shim. Per-ticker logs in
        output/orchestrator_<ts>/. Summary table at completion.
PR #32 — Aggregate dashboard: output/index.html (stable bookmark) +
        output/orchestrator_<ts>/index.html (audit copy). Sortable
        per-ticker table: ticker / σ-class / tier / ambiguity /
        verdict / spot / dip / rally / P(RT) / EV bps / status.
        Cron docs in docs/cron.md with sample crontab + log
        rotation + flock budget-guard pattern.

## W6 (institutional signals)  [WAVE CLOSED — PRs #33-#36]

PR #33 — D-W5-1 catalyst verification (Haiku-constrained primary-
        source check at T2 + T3; UNVERIFIED→magnitude=low,
        REFUTED→drop). Closes the RKLB-convertible hallucination
        class.
PR #34 — Fundamentals signal (TTM FCF yield + ND/EBITDA leverage +
        operating margin trend). Blend weight 0.08; FMP key-metrics-
        ttm + income-statement endpoints. Graceful degradation for
        pre-revenue names (LOW confidence on 1-2 sub-components).
PR #35 — Analyst revision momentum (time-decay-weighted net
        upgrades/downgrades over 90d). Blend weight 0.04; FMP
        upgrades-downgrades endpoint. 1.0 / 0.6 / 0.3 weighting by
        30d / 30-60d / 60-90d age buckets.
PR #36 — Integration tests + W6 close-out (this).

Net W6 blend impact:
  v2 signals: 10 → 12 (added fundamentals + revision_momentum)
  ai weight: 0.26 → 0.17 (AI's outsized voice trimmed for institutional
                          structural signals to occupy)

## To W5 / W6 (AI quality — catalyst verification)

### D-W5-1. AI catalyst-detail hallucination layer  [CLOSED in W6 PR #33]
- **Discovered**: RKLB W1 full-AI smoke (2026-05-22 15:47)
- **Symptom**: Pass 1 produced "Convertible note conversion window
  (2026-04-01/2026-06-30, bearish, magnitude med)" and Pass 2 carried the
  same framing forward. RKLB's convertibles (2029 maturity, ~$11 conversion
  price, deep in-the-money) do NOT operate on a calendar conversion window —
  voluntary exchange at holder's option whenever the 20-of-30-day premium
  trigger fires, or under specific issuer-call mandatory-conversion clauses.
  Pass 1's "window" framing is fabricated structure around a real
  underlying risk (the dilution overhang). Pass 2's adversarial layer
  caught Pass 1's REASONING errors but did not verify factual details.
  Same run also surfaced "Motiv Space Systems acquisition close (Q2/Q3)"
  which couldn't be independently verified by the human reviewer.
- **Root cause**: Pass 1 + web_search produces high-fluency catalyst
  summaries that can fabricate specifics (dates, deal counterparties,
  structure mechanics) around real underlying themes. The 5-min Pass 2
  Sonnet critique focuses on internal consistency and reasoning, not on
  primary-source verification of factual claims.
- **Fix in W5 or W6**: add a "catalyst verification" pass after Pass 2.
  For top-3 catalysts by magnitude weight, run a Haiku call constrained to
  one-question-per-catalyst lookups:
    - Earnings dates → verify via FMP earnings-calendar (already in code)
    - Convertible note terms → 10-Q footnote search (web_search constrained
      to SEC.gov + the company's IR site)
    - M&A close dates → 8-K / press release search (web_search constrained
      to wsj.com / reuters.com / company-IR)
    - Government contract awards → SAM.gov / DoD comptroller release search
  Each verification returns a confidence rating (verified / unverified /
  refuted). Catalysts that come back UNVERIFIED get magnitude downgraded
  to "low" before they hit signal_from_catalyst_proximity. Catalysts that
  come back REFUTED are dropped entirely.
- **Acceptance**: re-run RKLB; the convertible-note "window" gets flagged
  UNVERIFIED (no SEC footnote supports a Q2 window) and the catalyst
  contribution shrinks. If the Motiv deal is real, it stays VERIFIED;
  if not, it drops.
- **Cost estimate**: 3 catalysts × 1 Haiku call × ~$0.005 = ~$0.015/run.
  Negligible vs Pass 1's $0.50-0.60, and saves the entire run from
  publishing a hallucinated catalyst signal to a real trader.

---

## To W10 (calibration)

### D-W10-1. Pass 1 catalyst-detail accuracy tracking  [PARTIAL — PR #54 added catalyst capture to CSV (date/type/magnitude/direction columns); PR #61 added catalyst-stress capture; tests in test_catalyst_capture.py + test_catalyst_stress_capture.py. ANALYSIS LAYER (per-ticker hallucination rate, auto-downweight, banner warning) NOT BUILT — needs N≥30 days outcome data first]
- **Discovered**: RKLB W1 full-AI smoke (2026-05-22 15:47) — same root
  hallucination event as D-W5-1.
- **Symptom**: Pass 1 catalyst details can be fabricated even when the
  underlying theme is real. Without a verification layer (D-W5-1) or
  calibration tracking, the engine has no feedback loop to know when
  Pass 1 is producing reliable vs unreliable catalyst lists.
- **Fix in W10**: alongside the Brier-score calibration on dip/rally
  hit rates, add a catalyst-accuracy track:
    - For each catalyst emitted by Pass 1, store the date / type /
      magnitude / direction in CSV (extend the round_trip_history schema).
    - At calibration time (N ≥ 30 days), for each catalyst whose date
      has now passed, check via primary sources whether it actually
      occurred as Pass 1 described.
    - Compute a per-ticker and per-catalyst-type hallucination rate.
    - If hallucination rate > 20%, automatically downweight
      signal_from_catalyst_proximity for that ticker/type.
    - If overall hallucination rate > 30% across the universe, raise
      a banner-level warning in the daily report ("Pass 1 catalyst
      accuracy degraded — apply skepticism").
- **Acceptance**: after 30+ days of runtime, the dashboard shows
  per-ticker catalyst accuracy. Tickers with high hallucination rates
  trigger automatic downweighting without code edits.
- **Interaction with D-W5-1**: D-W5-1 (W5/W6) is preventative — block
  hallucinated catalysts from entering the blend in the first place.
  D-W10-1 is calibrative — measure how well preventative + Pass 2 are
  working, adjust weights accordingly. Both are needed.

### D-W10-2. Sector-decoupling signal saturates on every momentum name  [INTERIM — PR #58 cut sector_decoupling weight from 0.10 → 0.05 (config/diprally.yaml:166). This is the interim mitigation noted in the W2 punch list (D-W2-12's display fix made it visible; PR #58 partially addressed magnitude). Full data-driven choice between (a) wider cap / (b) σ-class-aware cap / (c) weight reduction still pending Brier-score validation from N≥30 days realized outcomes]
- **Discovered**: pattern across SNDK / LWLG / RKLB / INTC / MOG-A W1
  smokes (2026-05-22). All FIVE tickers reported `Sector decoupling
  +20.0%` (HIGH conf on the parabolic names, MEDIUM on MOG-A which is
  only mildly outperforming +27% YTD) — the signal hit its `±0.20` cap
  (signals.signal_from_sector_decoupling line 319) on every name tested.
- **MOG-A extends the diagnosis**: it's not just a "high-momentum
  outliers cap out" issue. MOG-A has 30d momentum of only +2.2% vs sector
  -7.8% — a 9.8pp 30d spread. The annualization formula
  (`drift = spread × 252/30`) amplifies this to +82.3% annualized,
  capped at +20%. So **the cap binds even on barely-outperforming names**.
  The annualization-by-252/30 multiplier is too aggressive; one mild
  outperformer month produces a maxed-out signal.
- **Symptom**: a signal that saturates on every observation isn't a
  discriminator — it's contributing a near-constant `+0.20 × 10% = +2pp`
  to the blend regardless of ticker. The signal's design weight (10%)
  is allocated to information value that isn't being delivered for this
  engine's target universe.
- **Root cause**: the cap was inherited from the seed (W0 migration).
  The seed engine ran only on SNDK at SNDK-like vol. The cap was
  conservative for that single-ticker context. On a 17-ticker universe
  of high-momentum names (where YTD returns are routinely +80% to +460%
  vs sector returns of -5% to +5%), the raw decoupling is regularly
  +50% to +300%, all clipped to the same +20% value.
- **Fix in W10 (data-driven decision)**:
    - After N≥30 days of runtime, regress signal_from_sector_decoupling's
      historical drift contribution against realized 60-day returns
      across the universe.
    - Three candidate corrections (test each):
      (a) Widen the cap to `±0.50` (preserve ranking discrimination)
      (b) Replace the cap with a σ-class-aware cap (EXTREME wider, MID tighter)
      (c) Reduce the signal's blend weight from 10% → 5% (acknowledge
          information-value degradation in this universe)
    - Pick the candidate with the lowest realized Brier score against
      held-out validation.
- **Why not fix earlier than W10**: changing the cap or weight without
  data is guessing. The seed value was conservative; W10 calibration is
  the right venue to revise it from realized outcomes.
- **Interim mitigation**: the W2 effective-weight display fix (D-W2-12)
  at least makes this visible to the trader — they can see that
  sector_decoupling is contributing a constant value across every ticker
  and discount it mentally.

---

## Capture protocol

When you discover a bug or sharp edge that doesn't belong in the current
wave, add it here. Format:
```
### D-W<N>-<seq>. <one-line headline>
- **Discovered**: <ticker/context/date>
- **Symptom**: …
- **Root cause**: …
- **Why didn't W<current> fix it**: …
- **Fix in W<N>**: …
- **Acceptance**: …
```

---

## Open / pending operational items

### D-OPS-1. Bump --max-parallel default from 2 → 3 (if FMP plan tier allows)
- **Discovered**: 2026-05-27, post-PR-#87 — Phase 1 wall-time is ~45-75s/ticker × 26 / parallel 2 ≈ 10-15 min. Could be ~7-10 min at parallel-3.
- **Symptom**: Phase 1 cycle takes 10-15 min wall-clock; operator wants faster.
- **Root cause**: max-parallel default set to 2 (PR #72, conservative). Each ticker does ~15 FMP supplementary calls + peer fetches.
- **Why didn't immediate fix**: needs FMP Starter plan rate-limit verification first. Bumping blindly risks triggering 429s mid-cycle (silent partial failures).
- **Fix when next system change is needed**: bump default to 3 in tools/orchestrate.py. Verify no FMP 429 errors over 3 consecutive cycles.
- **Acceptance**: cycle wall-time drops materially; no FMP rate-limit errors in subprocess logs.
- **Verification command + FMP support question**: see Diagnostic D-OPS-1 below in this file.

### Diagnostic D-OPS-1 — FMP Starter plan rate limits

**Decision**: whether parallel-3 (~30-45 FMP req/sec at peak) is safe on Starter plan.

**Data needed**: FMP Starter plan's documented request-per-minute (or per-second) cap, plus rate-limit response-header behavior.

**ONE exact command — measures actual rate-limit headers FMP returns**:

```bash
python3 -c "
import requests, os, time
key = os.environ['FMP_API_KEY']
# Hit the most-frequently-called endpoint in a tight loop and read
# rate-limit response headers (FMP uses X-Plan-Limit and similar).
print('--- Burst test: 15 requests as fast as possible ---')
for i in range(15):
    r = requests.get(
        f'https://financialmodelingprep.com/stable/quote?symbol=AMAT&apikey={key}',
        timeout=10,
    )
    headers_of_interest = {k: v for k, v in r.headers.items()
                            if 'rate' in k.lower() or 'limit' in k.lower()
                            or 'remaining' in k.lower() or 'reset' in k.lower()
                            or 'plan' in k.lower()}
    print(f'  req {i+1:2d}: HTTP {r.status_code}  '
          f'body[:60]={r.text[:60]!r}  '
          f'headers={headers_of_interest}')
print('--- Done. If any HTTP != 200 appeared, rate limit was hit. ---')
"
```

**Verbatim question for FMP support** (if any 429 appears OR headers are
unclear):

> Hello, my account is on the FMP Starter plan. Could you confirm in
> writing:
>
> 1. What is my exact rate limit on the Starter plan — requests per
>    second, per minute, per day?
> 2. Which endpoints share that quota — i.e. does `/stable/quote`,
>    `/stable/historical-price-eod`, `/stable/historical-chart/5min`,
>    `/stable/profile`, `/stable/analyst-estimates`, etc. all draw
>    from the same bucket, or separate buckets per endpoint family?
> 3. What HTTP status and headers do you return when the rate limit is
>    exceeded (429? what's the `Retry-After` header value)?
> 4. If I run 3 parallel subprocesses each issuing ~10-15 FMP calls
>    over ~30 seconds (peak ~1-1.5 req/sec sustained, brief bursts
>    of 3 req/sec), will that trigger any throttling?
> 5. If Starter throttles parallel-3, which next plan tier would
>    eliminate that constraint, and what's the monthly price?

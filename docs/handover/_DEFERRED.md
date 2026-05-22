# Deferred fixes — discovered in earlier waves, consumed in later ones

Living checklist of issues surfaced before their proper wave landed. Each
item names the wave that consumes it. When a wave starts, scan this file
first and clear its items as part of that wave's scope.

---

## To W2 (config-driven multi-ticker engine — Path A "big bang" lift)

W2 lifts EVERY configurable value out of `src/` and into a single source of
truth `config/diprally.yaml`. Sacred decision #17. The catalog below is the
full punch list, audited at the close of W1. Researcher must be able to
sweep any threshold without a code edit.

### D-W2-1. SNDK peer-fallback shim
- **File**: `src/engine.py` lines ~675–682
- **Current**: `if --peers omitted AND ticker == "SNDK": peer_tickers = ["MU", "WDC"]`
- **Why temp**: preserved W0 byte-for-byte SNDK acceptance test without baking
  a SNDK hardcode into peer logic permanently.
- **Fix in W2**: replace with `peer_tickers = REGISTRY[ticker].peers`.
  Registry sourced from YAML.

### D-W2-2. EV reporting in dollar capital units
- **Files**: `src/engine.py` (run_pipeline + scan_dip_rally_grid),
  `src/reporter.py` (format_report)
- **Current**: `capital_usd` parameter; reports `Net expected $/trade` and
  `~N.N shares`.
- **Fix in W2**: drop the `--capital` flag + parameter throughout. Switch
  reporter to EV/share + EV% of dip entry price. Sacred decision #6 (no
  capital concept).

### D-W2-3. SNDK-shaped Pass 1 prompt
- **File**: `src/ai_layer.py` (build_ai_pass1_prompt)
- **Current**: "Analyse {ticker} for a 60-day round-trip swing trade" hardcoded.
- **Fix in W2**: parameterize horizon from config; remove any other
  ticker-class assumptions in the prompt.

### D-W2-4. Top-level config.py constants → YAML
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

### D-W2-5. Scattered named constants outside config.py
- `src/ai_cache.py` — `SPOT_MOVE_INVALIDATION_PCT = 0.01`
- `src/signals.py` — `PHANTOM_SIGNAL_SE = 0.20`, `_AI_DERIVED_SIGNAL_NAMES`
- `src/engine.py:670` — `DRIFT_CAP = 1.0` (function-local!)
- `src/engine.py:229, 1061` — `spread_per_share_round_trip=2.0` default

### D-W2-6. Embedded signal thresholds in src/signals.py
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

### D-W2-7. Scattered tunables in src/engine.py workflow
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

### D-W2-8. src/data_fetch.py macro + liquidity thresholds
- VIX brackets: risk_off >25, risk_on <15, default fallback 18.0
- SPY trend: risk_on >0.02, risk_off <-0.03
- Options liquidity threshold: bid-ask spread <0.10
- Sector perf default window `days=30`
- Options DTE window `7 <= dte <= target*2`

### D-W2-9. src/math_utils.py — borderline structural, but tunable
- PDE grid: `n_space=400, n_time=2000` (performance/accuracy knobs)
- GARCH min-data `50`, fallback `n=90` bars
- GARCH optimizer initial values `[0.01, 0.05, 0.90]` and `[0.0001, 0.05, 0.90]`
- compute_realized_vol windows `(30, 60, 90)` (duplicated in engine.py)

### D-W2-10. Remove W1's --debug-spot-override flag
- **File**: `tools/run.py` + `src/engine.py:573-578`
- **Reason**: W1-only debug for cache testing; YAGNI in W2+. W4's broker
  will need different debug knobs; build the right one when needed.

### D-W2-11. Sensitivity table label widening bug
- **File**: `src/reporter.py` (sensitivity row)
- **Symptom** (visible in W1 SNDK full-AI smoke): some catalyst names get
  truncated awkwardly: `"Q4 guidance bar very high (rev $7.7 (-16pp)"` —
  the `[:35]` truncation in compute_sensitivity_table chops mid-parenthesis.
- **Fix in W2**: track the human label separately from the math row;
  truncate sanely (end at word boundary, append ellipsis), or widen the
  column to accommodate.

### Acceptance for W2
- `python tools/run.py SNDK` (no `--peers`, no `--capital`) auto-resolves
  peers from registry; report shows EV/share + EV% of dip; no SNDK
  hardcodes anywhere; `--debug-spot-override` flag gone.
- `python tools/run.py LWLG` runs end-to-end (D-W3-1 grid bug stays, dies
  in W3); peers auto-resolve to LWLG's registry entry.
- `python tools/run.py INTC` runs end-to-end (MID-class smoke; catches any
  signal-threshold bug that EXTREME-only smokes missed).
- Touching `config/diprally.yaml` and re-running changes behavior with
  ZERO code edits — verify by perturbing DEFAULT_CONVICTION_DIP from 0.65
  to 0.60 and confirming the report reflects the new threshold.
- The unit test from W1 (`tests/test_refusal_gate.py`) still passes — no
  regression in the σ-scaled tolerance + refusal gate.

---

## To W3 (σ-class auto-detection + class-specific defaults)

### D-W3-1. Grid step must be percent-based, not absolute dollars
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

### D-W3-2. Fake `pde_mass_conservation: 1.0` default
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

### D-W3-3. Three-method check skipped when no qualifying pair
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

## To W5 / W6 (AI quality — catalyst verification)

### D-W5-1. AI catalyst-detail hallucination layer
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

### D-W10-1. Pass 1 catalyst-detail accuracy tracking
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

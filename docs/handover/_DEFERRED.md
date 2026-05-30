# Deferred fixes — canonical backlog

Living checklist of issues surfaced before their fix was actionable. Scan this
file at the start of substantive work. **Open items are at the top, full
detail. Closed items are a one-line ledger at the bottom.**

Capture protocol (append new finds under "Open items"):
```
### D-W<N>-<seq>. <one-line headline>
- **Discovered**: <ticker/context/date>
- **Symptom / Root cause / Fix / Acceptance**: …
```

---

# OPEN ITEMS

## Operational

### D-2026-3. Stale `output/round_trip_history_*.csv` from sidelined tickers pollute the falsify harness
- **Discovered**: 2026-05-30 falsify_buy.py harness output. After the 5-name
  cull (commit 14d59e1), the harness still reads CSVs for the 21 sidelined
  tickers (ADBE/ANAB/ASTS/CRDO/...) — frozen at their pre-fix values, never
  re-run. Histogram shows "3 BUYs / 36 total" when reality is "3 BUYs / 5
  active fresh rows."
- **Severity**: COSMETIC but actively MISLEADING when reading the harness.
  Hides the real signal.
- **Fix candidates**: (a) one-time `rm output/round_trip_history_<sidelined>.csv`
  before next Pass B run; (b) recency filter in `falsify_buy.py` so it only
  counts CSVs touched within the last 24h. (a) is simpler; (b) is more
  durable across future cull/expand cycles.
- **Timing**: ACTION NOW — bundle with the next falsify_buy.py run so the
  histogram reads cleanly on Pass B.

### D-2026-4. Text report does not give DIRECT and WAIT-FOR-DIP equal billing
- **Discovered**: 2026-05-30 LWLG interrogation. The `verdict_subtype` column
  surfaces which strategy WON (max-EV), but the text report headline may
  bury the losing branch's EV. CSV has both ev_direct_bps + ev_wait_bps;
  reporter.format_report likely surfaces only the chosen one prominently.
- **Operator intent**: see BOTH numbers so the trader picks the strategy
  themselves ("I'm willing to wait" vs "enter now"). CLAUDE.md sacred #6:
  trader sizes externally, makes strategy choice externally.
- **Fix**: audit `src/reporter.py` `format_report` and `_headline_card` —
  ensure both ev_direct_bps and ev_wait_bps are surfaced equally with
  recommended dip/spot prices for each. Add a "if entering now: …" vs
  "if waiting for dip: …" section side-by-side.
- **Timing**: AFTER the engine is producing reliable BUYs on the full
  cohort (post Pass B + 26-name re-expansion). The data is already there;
  this is purely presentation.

### D-2026-5. Sacred #16 method-disagreement audit on MID names
- **Discovered**: 2026-05-30 falsify baseline runs. 6 of 6 REFUSED-METHOD
  refusals are MID-class (AVGO/CEG/LRCX/PLTR/SATS/STX). All show
  `| METHOD,EV` in the joint-refusal column → would refuse on EV too if
  method didn't fire first. Sacred #16 tolerance is σ-scaled; at MID
  σ=0.40-0.50 the tolerance is tight.
- **Question**: is sacred #16's MID tolerance miscalibrated post-stop-layer +
  post-MR-re-aim? MR changes the MC paths (more restoration) without
  changing the PDE math (no MR term in `pde_two_barrier`). Could widen MC
  vs PDE divergence on MID names that the PDE didn't account for.
- **Fix candidates**: (a) audit `three_method_cross_check` tolerance formula
  with MR-enabled MC paths; (b) extend `pde_two_barrier` to model MR (PR
  scope creep — only if (a) shows real divergence); (c) per-σ-class
  tolerance multipliers in YAML.
- **Timing**: AFTER 26-name re-expansion (D-2026-7) — need to see how
  many active MID names hit REFUSED-METHOD on fresh runs, not stale CSVs.

### D-2026-6. ARM/AMAT marginal-setup interrogation
- **Discovered**: 2026-05-30 post-MR-re-aim falsify baseline. ARM
  (+14 bps vs 25 bps hurdle, short by 11 bps) and AMAT (+29 bps vs 50 bps
  hurdle, short by 21 bps) are near-miss after both math fixes. Math says
  "real but insufficient edge" at T0.
- **Question**: do AI catalysts close the gap via Bayesian μ revision
  (the expected behavior), OR is there a structural issue with the dip/rally
  grid placement for low-σ HIGH and MID names? AMAT specifically may be
  facing the joint-conviction-unreachability my synth flagged earlier.
- **Fix candidates**: defer until Pass B data arrives. If Pass B's T2
  catalyst overlay lifts both over the hurdle → no defect, math working as
  designed. If T2 doesn't help → interrogate grid placement, MID conviction
  thresholds (0.65 dip / 0.70 rally|dip), and whether MID's swing_stop_pct=5%
  is too tight at σ=0.45.
- **Timing**: IMMEDIATELY AFTER Pass B (D-2026-7 trigger).

### D-2026-7. Re-expand 5-name cohort back to 26-name institutional roster
- **Discovered**: 2026-05-30 cull commit (14d59e1). Cull was a temporary
  iteration measure; 21 names sit in `tickers_scratch:` with full metadata.
- **Prerequisite**: Pass B confirms LWLG/MU/RKLB BUYs survive AI catalyst
  review (no regime-change refusal). ARM/AMAT interrogation complete.
- **Risk**: wider universe may surface edge cases the 5-name cohort
  didn't (limited-history names like VELO/CRWV/NBIS/INOD; biotech binary
  catalyst names like ENGN/ANAB; method-disagreement candidates like
  LRCX/STX). Each could become a follow-up D-item.
- **Timing**: AFTER D-2026-6 (ARM/AMAT interrogation). YAML edit only,
  reversible — move the 21 entries back from `tickers_scratch:` to
  `tickers:`.

### D-2026-8. patience_window_td=40 silently inert at horizon=20
- **Discovered**: 2026-05-30 stop-layer audit. `wait_exit_idx = min(dip_first
  + patience_window_td, n_days - 1)` clamps to 19 every time when
  patience_window_td (40) ≥ n_days (20). The YAML key documents a 40-td
  hold window but the math behaves like "hold to horizon end." Probe at
  patience=10 shows EV does change with smaller patience.
- **Severity**: LATENT — doesn't break the engine, but the configured value
  is silently meaningless. Operator may tune it expecting an effect.
- **Fix candidates**: (a) align patience_window_td to horizon_days
  (set default to 15 td or so, < horizon); (b) clarify YAML comment + log a
  warning when patience ≥ horizon; (c) remove the field entirely and use
  horizon - some_buffer.
- **Timing**: PARALLEL to D-2026-5 audit. Low priority.

### D-2026-9. Stop slippage at high σ (instant-fill assumption gap)
- **Discovered**: 2026-05-30 stop-layer design. `compute_dual_ev` assumes
  the stop fills at exactly `stop_level - friction`. At EXTREME σ=148%
  daily σ ≈ 9.4%, overnight gaps could blow through the stop by 5-10%
  before fill. Slippage modeled in friction term collectively but not
  per-leg.
- **Severity**: REAL — under-states left-tail loss on EXTREME names. The
  engine's EV is therefore optimistic by ~30-50 bps (0.30-0.50%) on
  EXTREME names where gap-down stops are common.
- **Fix candidates**: (a) per-leg friction (entry vs exit) with exit
  friction scaled to volatility; (b) explicit gap-slippage term in YAML
  per σ-class (e.g. EXTREME 0.005 = 50 bps additional slippage); (c)
  worst-of-day fill model in compute_dual_ev (path low instead of
  stop_level on stop-hit days).
- **Timing**: AFTER 26-name re-expansion (D-2026-7) shows accumulating
  realized stop-out data. Calibrate against actual fill prices, not synth.

## Data-gated (need ~30 days of realized-outcome CSV to pick the final fix)

> All four below shipped INTERIM mitigations and are waiting on Brier-score
> validation from accumulated daily outcomes. **If ≥30 days of
> `output/round_trip_history_*.csv` now exist, these are READY TO ACTION — not
> defer.** Check first.

### D-W2-17. peer_rs ±0.30 cap saturates on momentum names
- **Discovered**: SNDK smoke 2026-05-22. SNDK +449% YTD vs MU+WDC peers
  produced +30.0% MEDIUM — clipped at the +0.30 cap (`signal_from_peer_rs`).
- **Pattern**: peer_rs, sector_decoupling, sector_momentum all use the same
  `spread × 252/lookback` annualization that amplifies modest outperformance
  past their caps. Treat as one class with D-W10-2.
- **Interim**: PR #58 cut peer_rs blend weight 0.10 → 0.05.
- **Final fix candidates** (pick by lowest Brier, N≥30): (a) wider caps
  (peer_rs ±0.50); (b) σ-class-aware caps; (c) reduce weight when saturating.

### D-W2-18. Multi-saturation: blend over-confident when N signals all hit caps
- **Discovered**: SNDK smoke 2026-05-22. 4 of 8 active signals at extreme
  bullish values simultaneously (historical +88.9%, peer_rs +30% capped,
  sector_decoupling +20% capped, sector_momentum +60% regime-capped). When
  ≥N signals saturate same-direction, the blend reflects "extreme momentum"
  rather than ticker-specific forward drift. Phantom-signal std (W1) handles
  MISSING signals, not MULTIPLE-SATURATED ones.
- **Interim**: PR #59 shipped `multi_saturation.min_count=3` + ×1.30 std
  inflation (config-driven); tests in `test_multi_saturation.py`.
- **Final fix**: calibrate N and inflation magnitude from realized data.

### D-W10-2. sector_decoupling saturates on EVERY name (even mild outperformers)
- **Discovered**: across SNDK/LWLG/RKLB/INTC/MOG-A smokes 2026-05-22. All five
  hit the ±0.20 cap. **MOG-A is the key tell**: only +2.2% mom_30d vs sector
  -7.8% — a 9.8pp spread — yet `spread × 252/30` amplifies to +82.3% annualized,
  clipped to +20%. **The cap binds even on barely-outperforming names**, so the
  signal is a near-constant +2pp contributor, not a discriminator. Root cause:
  the `252/30` multiplier is too aggressive; cap inherited from single-ticker
  SNDK seed.
- **Interim**: PR #58 cut weight 0.10 → 0.05; D-W2-12 effective-weight display
  makes the constant contribution visible to the trader.
- **Final fix candidates** (lowest Brier, N≥30): (a) widen cap ±0.50;
  (b) σ-class-aware cap; (c) weight 10%→5%. NOTE: the real defect may be the
  annualization multiplier itself, not the cap — worth testing a gentler
  spread→drift transform as a 4th candidate.

### D-W10-1. Catalyst hallucination-rate tracking (analysis layer)
- **Discovered**: RKLB smoke 2026-05-22 (same event as D-W5-1, now closed).
  Pass 1 catalyst details can be fabricated even when the underlying theme is
  real. Without a feedback loop the engine can't tell reliable from unreliable
  catalyst lists per ticker.
- **Shipped (data capture)**: PR #54/#61 added catalyst + catalyst-stress
  columns to CSV; tests in `test_catalyst_capture.py` /
  `test_catalyst_stress_capture.py`.
- **NOT BUILT (analysis)**: per-ticker/per-type hallucination rate;
  auto-downweight `signal_from_catalyst_proximity` when rate >20%; universe
  banner warning when rate >30%. Needs N≥30 days of passed-date catalysts to
  check against primary sources.

## Architectural / design (need a decision, not just data)

### D-W2-19. Pass 2 cannot override factor_bias arithmetic  [latent but real]
- **Discovered**: 2026-05-24 full-system audit.
- **Symptom**: `apply_bull_bear_arithmetic` adds a ±5pp drift bias from
  Pass 1's `bull_factors / bear_factors` when |bull_high − bear_high| >
  `FACTOR_NET_THRESHOLD` (=4). Pass 2 routinely critiques Pass 1's factor
  classification but has no schema field to revise it — its drift critique
  reaches the AI signal slot (~22% blend weight) but cannot defeat the additive
  bias from OUTSIDE the slot. To neutralize +5pp it would have to push the
  22%-weighted signal down ~23pp. **This is a concrete AI-output-dropped case
  (CLAUDE.md hard constraint): Pass 2's reasoning is paid for but cannot act on
  the factor path.**
- **Severity**: LATENT — threshold rarely fires in the current universe, but on
  a deep-value setup with ≥5 HIGH bull factors, Pass 2's most institutional
  critique class is silently overridden.
- **Fix candidates**: (a) disable factor_bias when Pass 2 ran (cleanest —
  Pass 2's drift already captures its view); (b) add `revised_bull/bear_factors`
  to Pass 2 schema (verbose); (c) add `revised_factor_bias_override: <float>`
  (minimal schema growth). Pick (a) unless data shows factor_bias adds value.

### D-2026-2. Daily $2 cap not reconciled across orchestrator invocations
- **Discovered**: 2026-05-29 while fixing Defect F (out of its scope).
- **Symptom**: the $2/day cap is enforced PER orchestrator invocation off the
  broker's ESTIMATED tier costs, never reconciled against actual realized spend
  in the CSV ledger across invocations. Running multiple times in one day with
  cache busts (spot moved ≥ `AI_CACHE_SPOT_MOVE_INVALIDATION_PCT`) can spend
  **>$2/day actual** — each invocation re-grants the full $2 estimate. Within a
  single invocation and on cache hits, billing is correct.
- **Fix**: persistent daily-spend ledger keyed on trading day, read by
  `broker.allocate` to shrink the cap. Needs a decision on whether same-day
  re-runs should share one $2 budget.

## Operational

### D-OPS-1. Bump --max-parallel default 2 → 3 (if FMP plan tier allows)
- **Discovered**: 2026-05-27 post-PR-#87. Phase 1 ~45-75s/ticker; full roster
  at parallel-2 ≈ 10-15 min wall-clock. Parallel-3 ≈ 7-10 min.
- **Blocker**: needs FMP Starter rate-limit verification — bumping blindly
  risks 429s mid-cycle (silent partial failures).
- **Fix**: bump default in `tools/orchestrate.py`; verify no 429 over 3 cycles.

**Diagnostic command** (measures actual FMP rate-limit headers):
```bash
python3 -c "
import requests, os
key = os.environ['FMP_API_KEY']
print('--- Burst test: 15 requests as fast as possible ---')
for i in range(15):
    r = requests.get(
        f'https://financialmodelingprep.com/stable/quote?symbol=AMAT&apikey={key}',
        timeout=10,
    )
    hoi = {k: v for k, v in r.headers.items()
           if any(s in k.lower() for s in ('rate','limit','remaining','reset','plan'))}
    print(f'  req {i+1:2d}: HTTP {r.status_code}  body[:60]={r.text[:60]!r}  headers={hoi}')
print('--- Done. Any HTTP != 200 = rate limit hit. ---')
"
```
**Verbatim FMP support question** (if any 429 appears or headers unclear):
> Hello, my account is on the FMP Starter plan. Could you confirm in writing:
> 1. My exact rate limit — requests per second, per minute, per day?
> 2. Which endpoints share that quota (do `/stable/quote`,
>    `/stable/historical-price-eod`, `/stable/historical-chart/5min`,
>    `/stable/profile`, `/stable/analyst-estimates` draw from one bucket or
>    separate buckets per endpoint family)?
> 3. What HTTP status and headers do you return when the limit is exceeded
>    (429? what `Retry-After` value)?
> 4. If I run 3 parallel subprocesses each issuing ~10-15 calls over ~30s
>    (peak ~1-1.5 req/s sustained, brief 3 req/s bursts), will that throttle?
> 5. If Starter throttles parallel-3, which next tier removes the constraint,
>    and what's the monthly price?

---

# REJECTED — do NOT re-litigate

### D-2026-1. Defect E (two-sided catalyst drift → 0.5×magnitude) — REJECTED
- **Proposed**: change the `two-sided` direction multiplier in
  `signal_from_catalyst_proximity` from `0.0` to `0.5 × magnitude`
  ("uncertain direction, not zero direction").
- **Why rejected** (harness-verified 2026-05-29): the multiplier is applied as
  `total += mag * sign`, so `sign=0.5` injects a **strictly bullish** drift on
  every two-sided catalyst — a typical two-sided-earnings name gets +7.5%
  bullish drift (≈ +1pp on blended drift at catalyst_proximity weight 0.13) on
  essentially every name with earnings in window. For a DIP-buying engine that
  systematically suppresses dip detection and inflates rally prob.
- **Three independent reasons it's wrong**: (1) a direction-uncertain catalyst
  has ~0 expected *directional* drift by definition; (2) the earnings *variance*
  effect is already captured by `build_catalyst_vol_schedule` (sacred #9) — the
  fix double-counts; (3) contradicts **sacred #18** ("generic 'two-sided
  earnings' is the math layer's default, not a de-rating thesis").
- **Verdict**: current `two-sided → 0.0` drift is CORRECT. No code change.

---

# CLOSED — ledger (no action; here for PR archaeology)

**Closed 2026-05-24 audit**: 22 of 25 original items shipped.

| ID | Headline | Closed in |
|----|----------|-----------|
| D-W2-1 | SNDK peer-fallback shim → registry `resolve_peers()` | W2 |
| D-W2-2 | EV in $ capital units → EV/share + EV bps of dip (sacred #6) | W2 |
| D-W2-3 | SNDK-shaped Pass 1 prompt → horizon parameterized | W2 |
| D-W2-4 | ~40 config.py constants → pydantic YAML loader | W2 |
| D-W2-5 | Scattered named constants → imported from config | W2 |
| D-W2-6 | Embedded signal thresholds → YAML `signals:` namespace | W2 |
| D-W2-7 | Scattered engine.py tunables → config | W2 |
| D-W2-8 | data_fetch macro/liquidity thresholds → config | W2 |
| D-W2-9 | math_utils PDE/GARCH/vol-window knobs → config | W2 |
| D-W2-10 | Removed W1 `--debug-spot-override` flag | W2 |
| D-W2-11 | Sensitivity label mid-parenthesis truncation bug | W2 |
| D-W2-12 | Drift display: added Nominal \| Effective weight columns | W2 |
| D-W2-13 | Analyst extreme-outlier sanity gate + yfinance cross-check | W2 |
| D-W2-14 | yfinance fallback on FMP failure + apikey redaction + typed FetchError | W2 |
| D-W2-15 | Sacred #14 trend filter wired into engine | W2 |
| D-W2-16 | Sacred #15 insider signal removed from blend | W2/W3 |
| D-W3-1 | Grid step %-of-spot, not absolute $ (LWLG 1×1 collapse) | W3 PR #22 |
| D-W3-2 | Fake `pde_mass_conservation: 1.0` on no-pair → None/"n/a" | W3 PR #25 |
| D-W3-3 | Three-method check runs on anchor even with no qualifying pair | W3 PR #25 |
| D-W5-1 | AI catalyst-detail hallucination → verification pass (RKLB convertible) | W6 PR #33 |
| (W4) | Tier ladder, ambiguity, broker, snapshot emit | PRs #27-#30 |
| (W5) | Orchestrator + cron + aggregate dashboard | PRs #31-#32 |
| (W6) | Catalyst verification, fundamentals, analyst-revision signals | PRs #33-#36 |

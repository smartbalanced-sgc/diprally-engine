# Operator runbook (PR #60)

Use this when reading the daily report and the engine has REFUSED
recommendations across the universe — current frothy-market state.

## Universe shows mostly REFUSED — is the engine broken?

**No, this is institution-grade discipline working correctly.** In a
parabolic / overheated market, the engine SHOULD refuse most setups.
The current 17-ticker universe is up 60-250% YTD across the board;
mathematically there are very few positive-EV dip-buy opportunities.

The engine has 8 independent refusal gates running in parallel
(sacred decisions #13, #14, #16, #18 plus their lexical variants).
Refusing 14-17 of 17 tickers in this environment means the gates
are doing their job, not that the math is broken.

**Action**: continue running daily, wait for setups. Re-evaluate
when ≥3 tickers consistently produce ✅ BUY headlines across
consecutive runs — that's the signal that market conditions have
shifted from "everything overheated" to "real dip opportunities".

## What each refusal headline means and what to do

### `⛔ REFUSED · trend filter · mom_30d -XX%` (sacred #14)
- **What**: 30-day momentum below -25% AND no bullish/two-sided
  catalyst in horizon → falling knife without thesis
- **Operator action**: WAIT. Re-check when either (a) momentum
  stabilizes above -25% OR (b) AI surfaces a concrete catalyst
  (earnings, regulatory, M&A) that could re-rate the name
- **Override**: not recommended. Sacred #14 is empirically validated
  as negative-EV; the engine refuses for your protection

### `⛔ REFUSED · parabola filter · mom_30d +XX%` (sacred #18)
- **What**: 30-day momentum above +50% AND no bearish catalyst in
  horizon → blow-off top without de-rating thesis
- **Operator action**: WAIT. Re-check when either (a) momentum
  cools below +50% OR (b) AI surfaces a concrete bearish catalyst
  (secondary offering, regulatory action, peer disappointment,
  insider selling, debt overhang)
- **Override**: not recommended. Buying parabolic moves without a
  de-rating thesis is gravity-betting on a swing horizon

### `⛔ REFUSED · math methods disagree` (sacred #16)
- **What**: Monte Carlo, PDE, and closed-form first-passage
  probabilities diverge beyond the σ-scaled refusal threshold
- **Operator action**: investigate the math layer. Check the
  σ-CLASS PROFILE block in the report for unusual values
  (divergence > 30pp across anchors is suspicious). Re-run after
  next close when fresh data is in
- **Override**: never. Publishing under method disagreement means
  publishing a number the engine itself can't verify

### `⛔ REFUSED · EV/dip -XX.Xbps below 50bps hurdle` (sacred #13)
- **What**: best-EV pair found but post-friction expected return
  is too marginal to survive realistic execution costs
- **Operator action**: WAIT for higher-EV setup. The math is sound
  but the edge is too thin — institutional discipline says pass
- **Override**: discouraged. Sub-50bps trades empirically don't
  survive slippage + spread + fees + opportunity cost

### `⚠ BELOW-THRESHOLD · best-EV fallback` (no sacred decision)
- **What**: no pair cleared conviction strictly; engine showing
  best-by-EV fallback for context
- **Operator action**: DO NOT TRADE the fallback pair. Wait for a
  higher-conviction setup. The fallback is shown for diagnostic
  purposes only

### `⚠ NEGATIVE-EV` (no sacred decision)
- **What**: conviction thresholds met but `net_ev_per_share < 0`
  (average outcome loses money on the recommended pair)
- **Operator action**: SKIP. The math says you'll lose on average
  even though the pair "qualifies"

### `✅ BUY $X → SELL $Y` (clean recommendation)
- **What**: all gates cleared, math is sound, EV is institutional
- **Operator action**: this is the actionable recommendation.
  Trader sizes externally per sacred #6 (engine never recommends
  position size)

### `⚠ WAIT · no qualifying pair` (math-decisive no-trade)
- **What**: no dip × rally pair survived the conviction prefilter
- **Operator action**: re-run after next close

### `DELISTED — remove from universe` (PR #57)
- **What**: ticker is no longer available from data provider
  (bankrupt, acquired, delisted)
- **Operator action**: remove the ticker from `config/diprally.yaml`
  under `tickers:` to stop the daily failed-fetch attempts

## When to suspect the engine is actually broken

Real bugs look different from disciplined refusals:

1. **Same ticker, identical report two days in a row** with no spot
   movement → cache hit corruption. Run with `--bust-cache` to verify
2. **AI cost line shows $0 on a T2/T3 run with `--bust-cache`** →
   Anthropic credit issue or model dispatch broken
3. **σ-class auto-detected as MID for a name showing 100%+ vol** →
   GARCH fit corrupted; investigate σ TRIANGULATION block
4. **Three-method check shows MC and PDE differ by 15pp+** →
   bridge-correction edge case; flag for math review
5. **Backtest layer says "N=0 days tracked" after 30+ days of cron
   runs** → calibration harness broken; check
   `output/round_trip_history_<TICKER>.csv` for outcome columns
6. **Verification line missing entirely on T2/T3 runs** →
   Anthropic API key or domain configuration changed; check
   `tools/run.py` stderr for the catalyst-verification block

## Daily 30-second triage

1. Open `output/index.html` — count the verdict tiles
2. If `BUY ≥ 1`: read the headline card of each BUY ticker. Decision
   takes one line. Size externally per sacred #6
3. If `BUY = 0` and `REFUSED ≥ 10`: market is overheated. Skip the
   day, re-check tomorrow
4. If `FAIL ≥ 1`: investigate the specific ticker's
   `output/orchestrator_<ts>/<TICKER>.phase1.log`. Fix or skip
5. If `DELISTED ≥ 1`: edit `config/diprally.yaml`, remove the
   ticker, commit

## Reading BUY-verdict shape in different market regimes

The engine targets dip-and-rally setups. In a **strongly trending
regime** (e.g. mid-2026 AI/semi rally) the conviction threshold
(P(touch dip) ≥ 65%) biases the grid scan toward **shallow dip
targets** in trending names — bigger dip levels have lower touch
probability and don't clear conviction. Empirical pattern from the
2026-05-24 smoke:

  AMAT BUY  dip target -1.5% from spot  (spot $432, dip $426)
  LRCX BUY  dip target -1.0% from spot  (spot $305, dip $302)
  SATS BUY  dip target -1.5% from spot  (spot $124, dip $122)

These are **trend-continuation buys on small pullbacks**, NOT
contrarian deep-dip buys. The engine is being honest about what's
available — in a parabolic market, deep dips don't happen often
enough to clear the 65% probability gate. In a correction regime
the same engine will surface deeper, higher-EV dip buys because
the conviction math swings the other way.

Operator implication: a BUY in a frothy regime carries less margin
of safety than a BUY in a correction regime, even at identical
P(RT) values. Account for the regime in your external sizing
(sacred #6 — engine doesn't size).

## Reliability-warning chip line under the headline card

Audit 2026-05-24 round 2 added a unified reliability-warning line
directly under the verdict headline. Surfaces math-layer warnings
that exist but were previously buried mid-report:

  ⚠ RELIABILITY: near-IGARCH (α+β=0.9999) · σ divergence 19.3pp · ...

Read the chips in this order of severity:

1. **near-IGARCH** (GARCH α+β > 0.98): vol forecast is at the
   boundary of stationarity — σ estimate is structurally
   untradeable. The engine still computes everything but operator
   should NOT auto-execute on this name. Investigate options chain
   for an upcoming event the IV is pricing.
2. **σ divergence > 15pp**: blended σ disagrees with anchors by
   more than 15 percentage points. Usually a pre-event signal —
   options market is pricing vol the realized data hasn't seen yet.
   Check earnings calendar / catalysts.
3. **N/M signals active**: blend is weakly aggregated. Drift
   estimate carries more uncertainty than the band shows.
4. **σ-class registry hint stale**: auto-detector and YAML hint
   disagree. PR #62 advisor will emit a YAML patch after 5+ cycles
   of consistent mismatch — until then, the engine uses the
   detector value (sacred #1, data wins).

Empty line = no flags trip = clean run. No noise on healthy
tickers.

## Calibration timeline

- **Day 0**: harness shipped (W10 PR #47, #54); no data yet
- **Days 1-30**: cron / manual runs accumulate prediction history
- **Day 30+**: W10 analysis layer unlocks. Per-ticker Brier scores,
  per-σ-class P(RT) calibration, Pass 1 hallucination rates
- **Day 60+**: PR #53 Sonnet-vs-Opus A/B for T3 Pass 1
- **Day 90+**: auto-tune blend weights via Brier-optimal regression
  (reverses PR #58 interim mitigations once data says the right caps)

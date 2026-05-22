# Deferred fixes — discovered in earlier waves, consumed in later ones

Living checklist of issues surfaced before their proper wave landed. Each
item names the wave that consumes it. When a wave starts, scan this file
first and clear its items as part of that wave's scope.

---

## To W2 (multi-ticker generalisation + ticker registry)

### D-W2-1. SNDK peer-fallback shim
- **File**: `src/engine.py` lines ~675–682
- **Current**: `if --peers omitted AND ticker == "SNDK": peer_tickers = ["MU", "WDC"]`
- **Why temp**: preserved W0 byte-for-byte SNDK acceptance test without baking a
  SNDK hardcode into peer logic permanently.
- **Fix in W2**: replace with `peer_tickers = REGISTRY[ticker].peers`.
  Registry holds the canonical peer mapping per the seed context (LWLG/MRAM/
  ENGN/VELO3D → ETF fallback; ASTS → RKLB, IRDM; RKLB → ASTS, LMT; etc.).
- **Acceptance**: `python tools/run.py LWLG` (no `--peers`) auto-resolves
  LWLG's peers from the registry; SNDK still resolves to MU+WDC; the
  `elif ticker == "SNDK"` branch is gone.

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

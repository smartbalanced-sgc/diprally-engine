# Cron-driven daily runs (W5 PR #32)

The orchestrator (`tools/orchestrate.py`) is the canonical entry point
for a daily automated batch. It runs the full ticker universe under
the W4 budget broker, writes per-ticker logs + a summary file + an
aggregate HTML dashboard, and exits cleanly. It's designed to be
invoked by `cron` with zero hand-holding.

## Recommended crontab

US market closes at 16:00 ET (≈21:00 UTC during standard time, 20:00
UTC during daylight saving). Run a few minutes after close so the FMP
end-of-day prints are populated.

```cron
# Daily diprally-engine run — 16:30 ET, Mon-Fri.
# Adjust the TZ line for your operator account.
TZ=America/New_York
FMP_API_KEY=...your-FMP-key...
ANTHROPIC_API_KEY=...your-Anthropic-key...

30 16 * * 1-5  cd /path/to/diprally-engine && /usr/bin/env python3 tools/orchestrate.py >> /var/log/diprally/orchestrator.log 2>&1
```

Notes:
- `cd` first so relative paths inside the engine (config, output) resolve.
- Stdout + stderr both into the same log so subprocess error
  diagnostics survive.
- Pin the absolute Python interpreter path if the cron environment's
  `PATH` is sparse.
- The orchestrator writes per-ticker logs into
  `output/orchestrator_<timestamp>/` — that's the audit trail, not the
  stdout-redirected log. Don't expect the redirected file to have
  every detail.

## Log retention

Each run creates a new `output/orchestrator_<timestamp>/` directory.
Per-ticker `.log` files inside are small (~10-50 KB each). For a
17-ticker universe that's roughly 1-2 MB per run × 252 trading days
≈ 500 MB / year. Rotate or prune as needed:

```cron
# Weekly cleanup — keep last 30 orchestrator runs.
0 3 * * 0  cd /path/to/diprally-engine/output && ls -t orchestrator_* 2>/dev/null | tail -n +31 | xargs -r rm -rf
```

## Budget guard

The broker enforces `ai_daily_budget_cap_usd: 2.00` (sacred). For a
double-run safety net (e.g. a manual re-run on the same day), the
same-day AI cache makes the second invocation cheap — but if you want
a hard ceiling at the OS level, wrap the cron entry with a per-day
lock file:

```cron
30 16 * * 1-5  cd /path/to/diprally-engine && flock -n /tmp/diprally.lock python3 tools/orchestrate.py >> /var/log/diprally/orchestrator.log 2>&1
```

`flock -n` exits non-zero (and skips the run) if the lock is held —
prevents concurrent invocations from racing.

## Tightening the budget mid-day

If a partial run already burned some of the daily budget and you want
to re-run for the remaining tickers, pass `--budget` with the
remaining amount:

```bash
python tools/orchestrate.py --budget 0.50 --tickers LWLG MRAM
```

The broker still enforces strict ≤ on this lower cap.

## Verifying a cron run

After a scheduled execution, the operator's daily check:

1. Open `output/index.html` — the stable-URL bookmark.
2. Verify the timestamp matches the expected cron slot.
3. Skim the four-tile summary (BUY/WAIT/REFUSED/FAIL counts).
4. Drill into any FAIL row by clicking its ticker → opens the
   per-ticker dashboard. If the dashboard is also missing, check the
   matching `output/orchestrator_<ts>/<TICKER>.phase{1,2}.log`.

## What `output/index.html` looks like

Single sortable HTML table with one row per ticker:
- Ticker (linked to per-ticker dashboard)
- σ-class (EXTREME / HIGH / MID)
- Tier (T0 / T1 / T2 / T3 — broker-assigned)
- Ambiguity score
- Verdict pill (BUY / WAIT / REFUSED-EV / REFUSED-TREND / REFUSED-METHOD / FAIL)
- Spot, Dip target, Rally target, P(round-trip), EV bps of dip
- Status note (refusal reason, dip/rally summary, etc.)

Click column headers to sort. Default sort: ambiguity descending,
FAILs at the bottom.

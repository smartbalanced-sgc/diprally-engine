"""Multi-ticker orchestrator CLI (W5 PR #31).

Two-phase batch driver. See src/orchestrator.py for the library
implementation.

  Phase 1 — collect T0 snapshots from each ticker (subprocess)
  Broker  — src.broker.allocate() under $2/day cap
  Phase 2 — AI dispatch at broker-assigned tiers (subprocess)

Per-ticker logs land in output/orchestrator_<timestamp>/<TICKER>.log.
The summary is printed to stdout AND saved to SUMMARY.txt.

Usage:
    python tools/orchestrate.py                          # full universe (parallel=2 default)
    python tools/orchestrate.py --tickers LWLG INTC MU   # subset
    python tools/orchestrate.py --budget 1.00            # tighter cap
    python tools/orchestrate.py --dry-run                # Phase 1+broker only
    python tools/orchestrate.py --max-parallel 4         # more parallelism (faster)
    python tools/orchestrate.py --max-parallel 1         # sequential (slowest, safest)
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.broker import allocate, format_allocation
from src.orchestrator import (
    format_summary,
    generate_aggregate_dashboard,
    run_phase1,
    run_phase2,
)
from src.registry import list_universe


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Explicit ticker list. Defaults to the full YAML universe.")
    p.add_argument("--budget", type=float, default=None,
                   help="Override the $2/day cap (USD). Broker enforces strict ≤.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run Phase 1 + broker allocation, skip Phase 2 (AI dispatch).")
    p.add_argument("--max-parallel", type=int, default=4,
                   help="Concurrent subprocesses per phase (PR #89: default "
                        "4 — empirically safe on FMP Starter plan per the "
                        "diagnostic burst test 2026-05-27, 3 req/sec sustained "
                        "with no 429s. Bump to 6-8 if you upgrade FMP tier. "
                        "Drop to 2 for slower machines.")
    p.add_argument("--run-id", default=None,
                   help="Override the output/<run_id>/ directory name.")
    p.add_argument("--bust-cache", action="store_true",
                   help="Bypass the same-day AI cache for every ticker "
                        "in Phase 2 — forces fresh Pass 1/2/verify/stress "
                        "calls. Use to re-validate AI behavior after a "
                        "code change to the prompt or signal pipeline.")
    args = p.parse_args()

    tickers = args.tickers if args.tickers else list_universe()
    tickers = [t.upper() for t in tickers]
    if not tickers:
        print("ERROR: no tickers to run (universe empty + no --tickers)")
        return 1

    run_id = args.run_id or f"orchestrator_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = _REPO_ROOT / "output" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    if not os.getenv("FMP_API_KEY"):
        print("ERROR: FMP_API_KEY not set", file=sys.stderr)
        return 1

    # PR #76: market-state pre-flight. Banner if today is closed; loud
    # failure if NYSE library and FMP runtime status DISAGREE (one of
    # them is stale → operator decides whether to proceed).
    from src.market_calendar import (
        is_trading_day, last_trading_day, holiday_name,
        verify_market_state_via_fmp,
    )
    _today = datetime.now().date()
    if not is_trading_day(_today):
        last_open = last_trading_day(_today)
        h = holiday_name(_today) or "non-trading day"
        print()
        print("=" * 72)
        print(f"  ⚠  NYSE CLOSED TODAY  ({h})")
        print(f"  Last trading day: {last_open:%a %Y-%m-%d}")
        print("  Quotes will be that session's last trade. Engine analysis")
        print("  is correct for THAT data, not 'as of now'. Operator decides")
        print("  whether to act before next open. AI cache keys on the")
        print("  last trading day to avoid stale-data pollution.")
        print("=" * 72)
        print()
    fmp_check = verify_market_state_via_fmp(os.getenv("FMP_API_KEY"))
    if fmp_check is not None and not fmp_check["agree"]:
        print()
        print("=" * 72)
        print("  ⚠  MARKET-STATE DISAGREEMENT (NYSE library vs FMP)")
        print(f"  library says trading day = {fmp_check['library_open']}")
        print(f"  FMP runtime says open NOW = {fmp_check['fmp_open_now']}")
        print("  One source is stale. Investigate before trusting verdicts.")
        print("=" * 72)
        print()

    # Phase 1.
    results = run_phase1(tickers, run_dir, max_parallel=args.max_parallel)

    # Broker.
    valid_snapshots = [r.snapshot for r in results if r.snapshot is not None]
    if not valid_snapshots:
        print("\nNo valid snapshots — skipping broker / Phase 2.")
        allocation = None
    else:
        allocation = allocate(valid_snapshots, budget_usd=args.budget)
        print()
        print(format_allocation(allocation, valid_snapshots))

    # Phase 2.
    if args.dry_run:
        print("\n--dry-run: skipping AI dispatch.")
    elif allocation is not None:
        run_phase2(allocation, results, run_dir,
                    max_parallel=args.max_parallel,
                    bust_cache=args.bust_cache)

    # Summary.
    summary = format_summary(results, allocation)
    print()
    print(summary)
    summary_path = run_dir / "SUMMARY.txt"
    summary_path.write_text(summary + "\n")

    # Aggregate dashboard (W5 PR #32). Writes both a run-dir audit copy
    # and a stable output/index.html the operator bookmarks.
    dashboard_path = generate_aggregate_dashboard(results, allocation, run_dir)
    print(f"\nAggregate dashboard: {dashboard_path}")
    print(f"Stable bookmark URL: {run_dir.parent / 'index.html'}")
    print(f"Full logs + summary saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

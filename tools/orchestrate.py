"""Multi-ticker orchestrator CLI (W5 PR #31).

Two-phase batch driver. See src/orchestrator.py for the library
implementation.

  Phase 1 — collect T0 snapshots from each ticker (subprocess)
  Broker  — src.broker.allocate() under $2/day cap
  Phase 2 — AI dispatch at broker-assigned tiers (subprocess)

Per-ticker logs land in output/orchestrator_<timestamp>/<TICKER>.log.
The summary is printed to stdout AND saved to SUMMARY.txt.

Usage:
    python tools/orchestrate.py                          # full universe
    python tools/orchestrate.py --tickers LWLG INTC MU   # subset
    python tools/orchestrate.py --budget 1.00            # tighter cap
    python tools/orchestrate.py --dry-run                # Phase 1+broker only
    python tools/orchestrate.py --max-parallel 4         # subprocess fan-out
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
from src.orchestrator import format_summary, run_phase1, run_phase2
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
    p.add_argument("--max-parallel", type=int, default=1,
                   help="Concurrent subprocesses per phase (default 1 — sequential).")
    p.add_argument("--run-id", default=None,
                   help="Override the output/<run_id>/ directory name.")
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
        run_phase2(allocation, results, run_dir, max_parallel=args.max_parallel)

    # Summary.
    summary = format_summary(results, allocation)
    print()
    print(summary)
    summary_path = run_dir / "SUMMARY.txt"
    summary_path.write_text(summary + "\n")
    print(f"\nFull logs + summary saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Broker dry-run CLI (W4 PR #29).

Reads a JSON list of broker snapshots and prints the proposed tier
allocation. Used to validate broker behavior end-to-end without
needing the full T0 math pipeline to run on every ticker (which
arrives via the W5 orchestrator).

Snapshot JSON schema (one object per ticker):
    {
      "ticker": "INTC",
      "ambiguity": 0.31,
      "qualifies_for_t2_plus": true,
      "sigma_class": "MID"
    }

Usage:
    python tools/broker_preview.py < snapshots.json
    python tools/broker_preview.py --budget 1.50 < snapshots.json
    python tools/broker_preview.py --file snapshots.json

The orchestrator (W5) will call src.broker.allocate() directly with
real BrokerSnapshot dataclasses; this CLI is a debugging shim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.broker import BrokerSnapshot, allocate, format_allocation


def _load_snapshots(source) -> list[BrokerSnapshot]:
    raw = json.load(source)
    if not isinstance(raw, list):
        raise ValueError("snapshot JSON must be a top-level list")
    snapshots = []
    for entry in raw:
        snapshots.append(BrokerSnapshot(
            ticker=str(entry["ticker"]).upper(),
            ambiguity=float(entry["ambiguity"]),
            qualifies_for_t2_plus=bool(entry["qualifies_for_t2_plus"]),
            sigma_class=str(entry["sigma_class"]),
        ))
    return snapshots


def main():
    p = argparse.ArgumentParser(
        description="Broker dry-run: print proposed tier allocation for a "
                    "list of ticker snapshots under the $2/day cap.",
    )
    p.add_argument("--file", type=Path, default=None,
                   help="JSON file of snapshots. Defaults to stdin.")
    p.add_argument("--budget", type=float, default=None,
                   help="Override the daily budget cap (USD). "
                        "Defaults to config/diprally.yaml's ai_daily_budget_cap_usd.")
    args = p.parse_args()

    if args.file:
        with open(args.file) as f:
            snapshots = _load_snapshots(f)
    else:
        snapshots = _load_snapshots(sys.stdin)

    if not snapshots:
        print("No snapshots provided — nothing to allocate.")
        return 0

    allocation = allocate(snapshots, budget_usd=args.budget)
    print(format_allocation(allocation, snapshots))
    return 0


if __name__ == "__main__":
    sys.exit(main())

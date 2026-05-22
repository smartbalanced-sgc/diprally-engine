"""Single-ticker CLI entry. W0: thin dispatch to src.engine.run_pipeline.

W2 introduces multi-ticker batch via src.orchestrator alongside this.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python tools/run.py …` from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import (
    DEFAULT_CONVICTION_DIP,
    DEFAULT_CONVICTION_RALLY_COND,
    DEFAULT_HORIZON_DAYS,
)
from src.engine import run_pipeline


def main():
    p = argparse.ArgumentParser(
        description="diprally-engine (W0 single-ticker) — round-trip dip-and-rally framework",
    )
    p.add_argument("ticker", help="Ticker symbol (e.g. SNDK)")
    p.add_argument("--capital", type=float, default=10000.0,
                   help="Capital to deploy per round-trip in USD (default 10000). "
                        "Removed in W2 — engine becomes a pure recommendation tool.")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_DAYS,
                   help=f"Patience horizon in trading days (default {DEFAULT_HORIZON_DAYS})")
    p.add_argument("--conviction-dip", type=float, default=DEFAULT_CONVICTION_DIP,
                   help=f"Marginal P(touch dip) threshold (default {DEFAULT_CONVICTION_DIP})")
    p.add_argument("--conviction-rally-cond", type=float, default=DEFAULT_CONVICTION_RALLY_COND,
                   help=f"Conditional P(rally | dip) threshold (default {DEFAULT_CONVICTION_RALLY_COND})")
    p.add_argument("--mean-reversion", type=float, default=0.0,
                   help="Mean-reversion strength (default 0.0 = OFF; try 0.05/0.10/0.20 for sensitivity)")
    p.add_argument("--no-ai", action="store_true",
                   help="Skip all AI calls (math + backtest only).")
    p.add_argument("--peers", nargs="*", default=None,
                   help="Peer tickers for the peer-RS signal (e.g. --peers MU WDC). "
                        "No default: ticker registry supplies these in W2. As a "
                        "W0 transition shim, SNDK falls back to ['MU', 'WDC'].")
    p.add_argument("--show-rationale", action="store_true",
                   help="Verbose mode (currently default)")
    p.add_argument("--debug-spot-override", type=float, default=None,
                   help="Force spot to this value (debug only — used to test "
                        "AI cache invalidation on simulated spot moves without "
                        "waiting for live price changes). Removed/refactored "
                        "in W2 alongside the registry.")
    args = p.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())

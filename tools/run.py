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
    DEFAULT_HORIZON_DAYS,
)
from src.engine import run_pipeline


def main():
    p = argparse.ArgumentParser(
        description="diprally-engine (W0 single-ticker) — round-trip dip-and-rally framework",
    )
    p.add_argument("ticker", help="Ticker symbol (e.g. INTC)")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_DAYS,
                   help=f"Patience horizon in trading days (default {DEFAULT_HORIZON_DAYS})")
    p.add_argument("--conviction-dip", type=float, default=None,
                   help="Marginal P(touch dip) threshold. When omitted, uses "
                        "the σ-class default from config/diprally.yaml's "
                        "sigma_classes table (EXTREME 0.60, HIGH/MID 0.65).")
    p.add_argument("--conviction-rally-cond", type=float, default=None,
                   help="Conditional P(rally|dip) threshold. When omitted, uses "
                        "the σ-class default from config/diprally.yaml's "
                        "sigma_classes table (EXTREME/HIGH 0.75, MID 0.70).")
    p.add_argument("--mean-reversion", type=float, default=0.0,
                   help="Mean-reversion strength (default 0.0 = OFF; try 0.05/0.10/0.20 for sensitivity)")
    p.add_argument("--no-ai", action="store_true",
                   help="Skip all AI calls (math + backtest only). Equivalent to --tier T0.")
    p.add_argument("--tier", choices=("T0", "T1", "T2", "T3"), default="T3",
                   help="AI tier (W4 PR #27). T0=math only, T1=Haiku P1 only, "
                        "T2=Sonnet P1+P2, T3=Opus P1+Sonnet P2+Haiku stress "
                        "(default T3 — matches pre-W4 single-ticker behavior). "
                        "Multi-ticker batches will set this per ticker via the "
                        "broker (W4 PR #29).")
    p.add_argument("--peers", nargs="*", default=None,
                   help="Peer tickers for the peer-RS signal (e.g. --peers MU WDC). "
                        "When omitted, defaults to config/diprally.yaml's per-ticker "
                        "entry (stock_peers preferred, etf_peer fallback for EXTREME "
                        "names without comparable stock peers). Explicit --peers "
                        "overrides the registry entirely.")
    p.add_argument("--show-rationale", action="store_true",
                   help="Verbose mode (currently default)")
    p.add_argument("--emit-snapshot", action="store_true",
                   help="After the report, print one JSON line containing the "
                        "ticker's BrokerSnapshot (ticker / ambiguity / "
                        "qualifies_for_t2_plus / sigma_class). W5 orchestrator "
                        "collects these to drive the broker.")
    args = p.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())

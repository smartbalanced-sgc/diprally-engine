"""σ-class registry refresh advisor (PR #62).

Reads the recent CSV history per ticker, compares the auto-detected
σ-class against the registry hint in config/diprally.yaml, and
surfaces persistent mismatches that warrant a registry update.

Sacred design (sacred decision #1: data wins):
  - The registry hint is the operator's static "structural"
    classification of each ticker. Auto-detection is the
    per-run "this is the regime today" reading.
  - When auto-detection diverges from the registry hint for
    ≥ min_consecutive_runs consecutive runs (default 5),
    that's a structural shift — the registry is stale.
  - This tool DETECTS those shifts and prints a copy-paste
    YAML patch the operator can apply. It does NOT auto-edit
    YAML — operator decides whether the shift is real or a
    transient market regime.

Usage:
  python tools/registry_refresh.py
  python tools/registry_refresh.py --min-consecutive 3
  python tools/registry_refresh.py --tickers INTC AMAT MU

Exit code 0 always — this is advisory only.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.registry import classify, list_universe


_OUTPUT_ROOT = _REPO_ROOT / "output"


def _read_sigma_class_history(ticker: str, n_recent: int) -> list[str]:
    """Read the last n_recent rows of a ticker's history CSV and
    return the sigma_class column values (in chronological order).
    Returns empty list when CSV is missing or has no sigma_class
    column populated."""
    path = _OUTPUT_ROOT / f"round_trip_history_{ticker}.csv"
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    rows.sort(key=lambda r: r.get("date", ""))
    classes = []
    for r in rows[-n_recent:]:
        cls = (r.get("sigma_class") or "").strip()
        if cls:
            classes.append(cls)
    return classes


def analyze_ticker(ticker: str, min_consecutive: int) -> dict:
    """Analyze one ticker's σ-class history vs registry hint.
    Returns dict with:
      ticker, registry_hint, recent_classes, consistent,
      most_recent, suggested_action ('keep' / 'patch' / 'insufficient_data')
    """
    registry_hint = classify(ticker)  # None when ticker not in YAML
    recent = _read_sigma_class_history(ticker, n_recent=min_consecutive)

    if not recent:
        return {
            "ticker": ticker, "registry_hint": registry_hint,
            "recent_classes": [], "consistent": None,
            "most_recent": None, "suggested_action": "insufficient_data",
        }
    if len(recent) < min_consecutive:
        return {
            "ticker": ticker, "registry_hint": registry_hint,
            "recent_classes": recent, "consistent": None,
            "most_recent": recent[-1],
            "suggested_action": "insufficient_data",
        }

    # All classes in the window must agree to call it "consistent" —
    # one outlier means the shift isn't structural yet.
    unique = set(recent)
    consistent_class = recent[-1] if len(unique) == 1 else None

    if consistent_class is None:
        # Mixed — no structural shift detected. Hint stays.
        return {
            "ticker": ticker, "registry_hint": registry_hint,
            "recent_classes": recent, "consistent": False,
            "most_recent": recent[-1], "suggested_action": "keep",
        }

    if consistent_class == registry_hint:
        return {
            "ticker": ticker, "registry_hint": registry_hint,
            "recent_classes": recent, "consistent": True,
            "most_recent": consistent_class, "suggested_action": "keep",
        }

    return {
        "ticker": ticker, "registry_hint": registry_hint,
        "recent_classes": recent, "consistent": True,
        "most_recent": consistent_class, "suggested_action": "patch",
    }


def format_report(analyses: list[dict]) -> str:
    """Operator-readable summary + YAML patches for tickers that need them."""
    lines = []
    lines.append("=" * 78)
    lines.append("σ-CLASS REGISTRY REFRESH ADVISOR (PR #62)")
    lines.append("=" * 78)

    # Group by suggested_action for at-a-glance scanning.
    by_action: dict[str, list[dict]] = {
        "patch": [], "keep": [], "insufficient_data": [],
    }
    for a in analyses:
        by_action[a["suggested_action"]].append(a)

    # PATCH section — the actionable output.
    patches = by_action["patch"]
    if patches:
        lines.append("")
        lines.append(f"⚠ {len(patches)} ticker(s) need a registry update:")
        lines.append("")
        for a in patches:
            lines.append(
                f"  {a['ticker']:<8}  registry hint={a['registry_hint']}, "
                f"auto-detected={a['most_recent']} for "
                f"{len(a['recent_classes'])} consecutive runs"
            )
        lines.append("")
        lines.append("Copy-paste YAML patch for config/diprally.yaml:")
        lines.append("-" * 78)
        for a in patches:
            lines.append(f"  {a['ticker']}:")
            lines.append(f"    sigma_class: {a['most_recent']}  # was {a['registry_hint']} — PR #62 advisory")
        lines.append("-" * 78)
        lines.append("Review each suggestion before editing — a multi-month")
        lines.append("regime shift warrants an update; a transient frothy market")
        lines.append("does not. Sacred #1: data wins for the current run, but")
        lines.append("the static registry is the operator's structural call.")

    # KEEP section — no action needed.
    keeps = by_action["keep"]
    if keeps:
        lines.append("")
        lines.append(f"✓ {len(keeps)} ticker(s) registry hint stable:")
        for a in keeps:
            if a["consistent"]:
                lines.append(
                    f"  {a['ticker']:<8}  {a['registry_hint']} — auto-detect matches"
                )
            else:
                mix = Counter(a["recent_classes"]).most_common()
                mix_str = ", ".join(f"{c}={n}" for c, n in mix)
                lines.append(
                    f"  {a['ticker']:<8}  {a['registry_hint']} hint, mixed auto-detect "
                    f"({mix_str}) — no structural shift yet"
                )

    # INSUFFICIENT_DATA section.
    insufficient = by_action["insufficient_data"]
    if insufficient:
        lines.append("")
        lines.append(f"ⓘ {len(insufficient)} ticker(s) insufficient history:")
        for a in insufficient:
            if not a["recent_classes"]:
                lines.append(f"  {a['ticker']:<8}  no CSV history yet")
            else:
                lines.append(
                    f"  {a['ticker']:<8}  only {len(a['recent_classes'])} runs available — "
                    f"need more"
                )

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description="σ-class registry refresh advisor.",
    )
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Subset of tickers to analyze. Defaults to full universe.")
    p.add_argument("--min-consecutive", type=int, default=5,
                   help="Minimum consecutive runs at the same auto-detected "
                        "class before flagging a registry mismatch (default 5).")
    args = p.parse_args()

    tickers = [t.upper() for t in args.tickers] if args.tickers else list_universe()
    if not tickers:
        print("ERROR: no tickers to analyze")
        return 1

    analyses = [analyze_ticker(t, args.min_consecutive) for t in tickers]
    print(format_report(analyses))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Catalyst-occurrence audit triage (D-W10-1 data foundation).

Walks each ticker's round_trip_history CSV, parses the
pass2_catalysts_json column (W10 PR #54 capture), and surfaces
catalysts whose predicted date has elapsed but which the operator
hasn't yet marked OCCURRED / NOT_OCCURRED / PARTIAL / UNVERIFIABLE
in the audit ledger.

Why non-AI:
  - The runtime verifier (W6 PR #33) already does PLAUSIBILITY check
    against training data at engine time. That's preventative.
  - This is OCCURRENCE check — did the catalyst happen as Pass 1/Pass 2
    described? — which requires post-event ground truth.
  - PR #52 documented that the constrained-web-search verifier fails
    100% to produce parseable JSON. Repeating that pattern weekly across
    the universe is a guaranteed cost-without-value tax.
  - Operator manual review is the correct ground-truth source for
    hallucination-rate analysis (D-W10-1). This tool's job is to
    surface what needs review, not to guess at it.

Output:
  - stdout: structured table of catalysts whose date has elapsed and
    are not yet in the ledger ("PENDING REVIEW" rows)
  - output/catalyst_audit_ledger.csv: append-only audit ledger; row =
    one operator verification verdict. Tool reads this to know which
    catalysts have been reviewed; operator appends rows after their
    own primary-source check.

Ledger columns:
  audit_date           — when operator filed the verdict (YYYY-MM-DD)
  ticker               — ticker symbol
  catalyst_name        — name as Pass 2 emitted it
  catalyst_type        — type as Pass 2 emitted it
  predicted_date_window — date_or_window as Pass 2 emitted it
  predicted_direction  — direction_risk as Pass 2 emitted it
  predicted_magnitude  — magnitude as Pass 2 emitted it
  first_seen_date      — engine run date that first surfaced this catalyst
  verdict              — OCCURRED / PARTIAL / NOT_OCCURRED / UNVERIFIABLE
  reason               — operator's 1-line rationale
  source_url           — primary-source link the verdict rests on

Usage:
  python tools/catalyst_audit.py                    # full universe
  python tools/catalyst_audit.py --tickers INTC MU  # subset
  python tools/catalyst_audit.py --since-days 30    # only catalysts
                                                       first-seen in last N days

Exit code 0 always — advisory tool.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.registry import list_universe


_OUTPUT_ROOT = _REPO_ROOT / "output"
_LEDGER_FILENAME = "catalyst_audit_ledger.csv"

LEDGER_COLUMNS = [
    "audit_date", "ticker", "catalyst_name", "catalyst_type",
    "predicted_date_window", "predicted_direction", "predicted_magnitude",
    "first_seen_date", "verdict", "reason", "source_url",
]

VALID_VERDICTS = {"OCCURRED", "PARTIAL", "NOT_OCCURRED", "UNVERIFIABLE"}


# Date-extraction patterns. date_or_window is freeform LLM text — best-effort
# parse. If we can't extract a date, treat the catalyst as
# "indeterminate timing" and the operator can still review it on demand.
_DATE_PATTERNS = [
    # ISO date: 2026-05-08, possibly with /endrange
    (re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"), "%Y-%m-%d"),
    # "May 8, 2026" / "May 8 2026"
    (re.compile(r"\b([A-Za-z]+ \d{1,2},? \d{4})\b"), "%B %d, %Y"),
    # "May 2026"
    (re.compile(r"\b([A-Za-z]+ \d{4})\b"), "%B %Y"),
]
_QUARTER_RE = re.compile(r"\bQ([1-4])\s*([0-9]{4})\b")


def _extract_latest_date(text: str) -> "datetime | None":
    """Best-effort date extraction from a freeform date_or_window string.
    Returns the LATEST date we can parse — for range strings like
    '2026-04-01/2026-06-30' we want the end-of-window so we don't flag
    a catalyst as elapsed before its window actually closes."""
    if not text or not isinstance(text, str):
        return None
    candidates: list[datetime] = []
    for pat, fmt in _DATE_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(1).replace(",", "")
            try:
                # Strip the comma form for "%B %d %Y"
                _fmt = fmt.replace(", ", " ").replace(",", "")
                candidates.append(datetime.strptime(raw, _fmt))
            except ValueError:
                continue
    # Quarter expressions: Q2 2026 → end of Q2 = 2026-06-30
    for m in _QUARTER_RE.finditer(text):
        q = int(m.group(1))
        y = int(m.group(2))
        end_month = q * 3
        # last day of end_month
        try:
            if end_month in (3, 12):
                eom = 31
            elif end_month == 6:
                eom = 30
            else:  # 9
                eom = 30
            candidates.append(datetime(y, end_month, eom))
        except ValueError:
            continue
    return max(candidates) if candidates else None


def _load_ledger(path: Path) -> dict:
    """Read the audit ledger CSV. Returns {(ticker, catalyst_name): row_dict}
    keyed by the natural identity tuple. Missing file → empty dict."""
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("ticker", ""), row.get("catalyst_name", ""))
            out[key] = row
    return out


def _ensure_ledger_exists(path: Path) -> None:
    """Create an empty ledger with header if it doesn't exist. Operator
    appends rows manually after their primary-source review."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        w.writeheader()


def _read_catalyst_history(ticker: str, since_days: int) -> list[dict]:
    """Walk a ticker's round_trip_history CSV, parse pass2_catalysts_json
    on rows within `since_days`, return a list of unique catalyst dicts
    keyed by name. Each entry annotated with first_seen_date (earliest
    CSV row that surfaced it). Pass 1 fallback when Pass 2 absent."""
    path = _OUTPUT_ROOT / f"round_trip_history_{ticker}.csv"
    if not path.exists():
        return []
    cutoff = datetime.today() - timedelta(days=since_days)
    seen: dict[str, dict] = {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r.get("date", ""))
    for row in rows:
        try:
            row_date = datetime.strptime((row.get("date") or "")[:10],
                                          "%Y-%m-%d")
        except ValueError:
            continue
        if row_date < cutoff:
            continue
        # Prefer Pass 2's revised catalyst list (sacred #7: Pass 2 wins).
        # Fall back to Pass 1 when Pass 2 absent (T0/T1 runs).
        catalysts_raw = (row.get("pass2_catalysts_json")
                          or row.get("pass1_catalysts_json") or "")
        if not catalysts_raw:
            continue
        try:
            catalysts = json.loads(catalysts_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(catalysts, list):
            continue
        for c in catalysts:
            if not isinstance(c, dict):
                continue
            name = (c.get("name") or "").strip()
            if not name:
                continue
            if name in seen:
                continue  # earlier row already surfaced it; keep first-seen
            seen[name] = {
                "ticker": ticker,
                "catalyst_name": name,
                "catalyst_type": c.get("type", ""),
                "predicted_date_window": c.get("date_or_window", ""),
                "predicted_direction": c.get("direction_risk", ""),
                "predicted_magnitude": c.get("magnitude", ""),
                "first_seen_date": row_date.strftime("%Y-%m-%d"),
            }
    return list(seen.values())


def find_pending_reviews(tickers: list[str], since_days: int,
                          ledger: dict, today: "datetime | None" = None) -> list[dict]:
    """Walk all tickers, return catalysts whose predicted date has
    elapsed and which are not in the ledger. Sort by elapsed-since-due
    descending (most-overdue first)."""
    today = today or datetime.today()
    pending = []
    for t in tickers:
        for c in _read_catalyst_history(t, since_days):
            key = (c["ticker"], c["catalyst_name"])
            if key in ledger:
                continue
            due_date = _extract_latest_date(c["predicted_date_window"])
            c["due_date"] = due_date.strftime("%Y-%m-%d") if due_date else "(unparsed)"
            c["days_since_due"] = (today - due_date).days if due_date else None
            # Only surface catalysts whose date has elapsed (or unparseable —
            # operator decides). Future-dated catalysts: skip (not yet due).
            if due_date is not None and due_date > today:
                continue
            pending.append(c)
    # Sort: parseable elapsed first (most overdue), then unparsed at bottom.
    pending.sort(key=lambda x: (x["days_since_due"] is None,
                                 -(x["days_since_due"] or 0)))
    return pending


def format_pending_table(pending: list[dict]) -> str:
    """Render PENDING REVIEW table for stdout."""
    if not pending:
        return ("CATALYST AUDIT — no pending reviews. Either no catalyst "
                "dates have elapsed yet, or all elapsed catalysts are in "
                "the ledger.")
    lines = []
    lines.append("=" * 100)
    lines.append(f"CATALYST AUDIT — {len(pending)} pending operator review(s)")
    lines.append("=" * 100)
    lines.append(
        f"  {'TICKER':<8} {'DUE':<12} {'OVERDUE':<8} {'TYPE':<14} "
        f"{'DIR':<10} {'MAG':<6} CATALYST"
    )
    lines.append("  " + "-" * 96)
    for p in pending:
        overdue = (f"{p['days_since_due']}d"
                   if p["days_since_due"] is not None else "n/a")
        # Truncate long names so the table stays readable.
        name = p["catalyst_name"]
        if len(name) > 40:
            name = name[:37] + "..."
        lines.append(
            f"  {p['ticker']:<8} {p['due_date']:<12} {overdue:<8} "
            f"{(p['catalyst_type'] or '?')[:14]:<14} "
            f"{(p['predicted_direction'] or '?')[:10]:<10} "
            f"{(p['predicted_magnitude'] or '?')[:6]:<6} {name}"
        )
    lines.append("")
    lines.append("To file a verdict, append a row to output/catalyst_audit_ledger.csv:")
    lines.append("  audit_date,ticker,catalyst_name,catalyst_type,")
    lines.append("  predicted_date_window,predicted_direction,predicted_magnitude,")
    lines.append("  first_seen_date,verdict,reason,source_url")
    lines.append(f"Valid verdicts: {' / '.join(sorted(VALID_VERDICTS))}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Surface elapsed catalysts pending operator review."
    )
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Subset of tickers (default: full registry).")
    parser.add_argument("--since-days", type=int, default=90,
                        help="Only consider catalysts first surfaced in "
                             "the last N days (default 90).")
    parser.add_argument("--ledger",
                        default=str(_OUTPUT_ROOT / _LEDGER_FILENAME),
                        help="Path to the audit ledger CSV.")
    args = parser.parse_args()

    tickers = args.tickers or list_universe()
    ledger_path = Path(args.ledger)
    _ensure_ledger_exists(ledger_path)
    ledger = _load_ledger(ledger_path)
    pending = find_pending_reviews(tickers, args.since_days, ledger)
    print(format_pending_table(pending))
    print(f"\nLedger: {ledger_path}  ({len(ledger)} verdicts on file)")


if __name__ == "__main__":
    main()

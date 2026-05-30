"""Falsification harness for the 0-BUY anomaly (audit protocol step 1).

QUESTION
--------
Which gate in the verdict waterfall is the BINDING constraint that stops
tickers from ever reaching BUY? We have NOT measured this — we fixed defects
on theory. This harness measures it on the REAL roster against LIVE FMP data
by driving the existing, tested orchestrator and reading the CSVs it writes.

`verdict_state` is priority-ordered, so it IS the binding-gate label:

    REFUSED-TREND     (sacred #14: mom_30d < -25% AND no supporting catalyst)
    REFUSED-PARABOLA  (sacred #18: mom_30d >= class blow-off AND no bearish catalyst)
    REFUSED-METHOD    (sacred #16: MC vs PDE disagree)
    REFUSED-EV        (sacred #13: EV below the σ-class hurdle)
    WAIT              (no dip/rally pair cleared the conviction prefilter)
    BELOW-THRESHOLD   (a pair exists but conviction not strictly met)
    NEGATIVE-EV       (best pair EV < 0)
    BUY               (survived everything)

TWO REGIMES (run back to back, same trading day):

  Pass A — CATALYST-BLIND BASELINE.  `orchestrate.py --dry-run` runs Phase 1
           only: every ticker at T0 (no AI, FREE), empty catalysts.  Isolates
           the structural floor — how many names die at the catalyst-gated
           trend/parabola filters vs. survive to the EV/conviction gates on
           math alone.  Names that are NOT parabolic/falling won't trip gates
           1-2 even with empty catalysts, so their verdict here exposes
           whether the EV/conviction MATH independently blocks BUY.

  Pass B — REAL AI PASS.  full `orchestrate.py`, broker-allocated within the
           $2/day cap (NOT bypassed).  Real catalysts.  Tests whether
           AI-surfaced catalysts unblock the trend/parabola refusals — i.e.
           whether Defect A's catalyst restoration is enough to PRODUCE BUYs.

COST:  Pass A is free.  Pass B spends up to the $2/day cap (your normal daily
       AI cost).  --baseline-only skips Pass B (FMP-only, $0).

REQUIRES:  FMP_API_KEY in env + FMP host reachable (both passes).  Pass B also
           needs ANTHROPIC_API_KEY + AI host reachable; if AI can't run, Pass B
           degrades toward Pass A and ai_status reads INCOMPLETE/DEGRADED —
           itself a finding.

RUN:
    python tools/falsify_buy.py                  # baseline + real ($<=2)
    python tools/falsify_buy.py --baseline-only  # free, FMP only
    python tools/falsify_buy.py --tickers ARM MU AMAT
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_OUTPUT_DIR = _REPO_ROOT / "output"
_ORCHESTRATE = _REPO_ROOT / "tools" / "orchestrate.py"


def _f(row, key, default=None):
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _catalyst_count(row) -> int:
    raw = row.get("pass1_catalysts_json", "") or ""
    if not raw:
        return 0
    try:
        data = json.loads(raw)
        return len(data) if isinstance(data, list) else 0
    except (ValueError, TypeError):
        return 0


def _ticker_from_path(p: Path) -> str:
    # output/round_trip_history_{TICKER}.csv
    name = p.name
    stem = name[len("round_trip_history_"):-len(".csv")]
    return stem.upper()


def _latest_row(csv_path: Path) -> dict | None:
    try:
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return None
    return max(rows, key=lambda r: r.get("date", "")) if rows else None


def _snapshot() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(_OUTPUT_DIR.glob("round_trip_history_*.csv")):
        row = _latest_row(p)
        if row is not None:
            out[_ticker_from_path(p)] = row
    return out


def _fmt_pct(frac, default="   --  "):
    return default if frac is None else f"{frac*100:+6.2f}%"


def _fmt_ev(frac):
    """EV as bps WITH the mandatory % gloss (CLAUDE.md basis-point rule)."""
    return "      --      " if frac is None else f"{frac*1e4:+7.1f}bps({frac*100:+.2f}%)"


def _explain(row) -> str:
    vs = row.get("verdict_state", "?")
    ev = _f(row, "ev_pct_of_dip")
    pdip = _f(row, "p_dip")
    prc = _f(row, "p_rally_cond")
    cat = _catalyst_count(row)
    reasons = (row.get("refusal_reasons_all", "") or "").strip()
    base = {
        "REFUSED-TREND": f"falling-knife (#14), catalysts={cat}",
        "REFUSED-PARABOLA": f"blow-off (#18), catalysts={cat}",
        "REFUSED-METHOD": "MC vs PDE disagreement (#16)",
        "REFUSED-EV": f"EV {_fmt_ev(ev)} below σ-class hurdle (#13)",
        "WAIT": f"no pair cleared conviction prefilter (p_dip={_fmt_pct(pdip)})",
        "BELOW-THRESHOLD": f"conviction short (p_dip={_fmt_pct(pdip)}, p_rally|dip={_fmt_pct(prc)})",
        "NEGATIVE-EV": f"best EV {_fmt_ev(ev)} < 0",
        "BUY": f"PASSED — EV {_fmt_ev(ev)}",
    }.get(vs, "(no/unknown verdict — AI may not have delivered)")
    return base + (f"  | {reasons}" if reasons else "")


def _print_pass(title: str, snap: dict[str, dict]) -> Counter:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    if not snap:
        print("  (no output/round_trip_history_*.csv rows found — did the run write to output/?)")
        return Counter()
    hdr = (f"{'TICKER':<7}{'CLASS':<8}{'VERDICT':<17}{'p_dip':>8}{'p_r|d':>8}"
           f"  {'EV':>20}  {'cat':>4}{'tier':>5} {'ai_status':>11}")
    print(hdr)
    print("-" * len(hdr))
    gates = Counter()
    for tk in sorted(snap):
        r = snap[tk]
        vs = r.get("verdict_state", "?") or "(none)"
        gates[vs] += 1
        print(f"{tk:<7}{r.get('sigma_class',''):<8}{vs:<17}"
              f"{_fmt_pct(_f(r,'p_dip')):>8}{_fmt_pct(_f(r,'p_rally_cond')):>8}  "
              f"{_fmt_ev(_f(r,'ev_pct_of_dip')):>20}  {_catalyst_count(r):>4}"
              f"{(r.get('ai_tier','') or ''):>5} {(r.get('ai_status','') or ''):>11}")
    print("-" * len(hdr))
    print("  GATE HISTOGRAM (binding constraint per ticker):")
    for vs, n in gates.most_common():
        print(f"    {vs:<18} {n:>3}")
    print(f"    {'TOTAL':<18} {sum(gates.values()):>3}     BUYs: {gates.get('BUY',0)}")
    return gates


def _print_explainers(label: str, snap: dict[str, dict]):
    print(f"\n  WHY each non-BUY was blocked — {label}:")
    for tk in sorted(snap):
        r = snap[tk]
        if (r.get("verdict_state") or "") == "BUY":
            continue
        print(f"    {tk:<7} {(r.get('verdict_state') or '(none)'):<17} {_explain(r)}")


def _run_orchestrator(extra_args: list[str], label: str):
    cmd = [sys.executable, str(_ORCHESTRATE), *extra_args]
    print(f"\n>>> {label}\n    $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if res.returncode != 0:
        print(f"    (orchestrate.py exited {res.returncode} — partial results may still parse)")


def main():
    ap = argparse.ArgumentParser(description="0-BUY falsification harness")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--budget", type=float, default=None,
                    help="$/day cap for the real AI pass (default: orchestrator's $2)")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--bust-cache", action="store_true",
                    help="Forward --bust-cache to Pass B so cached AI from a "
                         "prior same-day run is bypassed. Use to validate AI "
                         "code changes (e.g. prompt restructuring) that would "
                         "otherwise be invisible because the cache replays.")
    args = ap.parse_args()

    common = []
    if args.tickers:
        common += ["--tickers", *args.tickers]
    common += ["--max-parallel", str(args.max_parallel)]

    print("#" * 92)
    print("# 0-BUY FALSIFICATION HARNESS")
    print("# Null to falsify: 'the engine CANNOT produce a BUY verdict.'")
    print("#" * 92)

    # Pass A — catalyst-blind baseline: Phase 1 only (all T0), FREE.
    _run_orchestrator(common + ["--dry-run", "--run-id", "falsify_baseline"],
                      "PASS A — catalyst-blind baseline (T0, no AI, FREE)")
    baseline = _snapshot()

    real = {}
    if not args.baseline_only:
        real_args = list(common)
        if args.budget is not None:
            real_args += ["--budget", str(args.budget)]
        real_args += ["--run-id", "falsify_real"]
        if args.bust_cache:
            real_args += ["--bust-cache"]
        _run_orchestrator(real_args, "PASS B — real AI pass (broker-capped at $2/day)")
        real = _snapshot()

    gates_a = _print_pass("PASS A — CATALYST-BLIND (T0, empty catalysts)", baseline)
    _print_explainers("PASS A (baseline)", baseline)

    gates_b = Counter()
    if not args.baseline_only:
        gates_b = _print_pass("PASS B — REAL AI (catalysts present, $-capped)", real)
        _print_explainers("PASS B (real AI)", real)

    print("\n" + "#" * 92)
    print("# FALSIFICATION VERDICT")
    print("#" * 92)
    buys_a = gates_a.get("BUY", 0)
    print(f"  Baseline (no AI) BUYs: {buys_a} / {sum(gates_a.values())}")
    if args.baseline_only:
        print("  (Re-run without --baseline-only to test whether catalysts unblock BUYs.)")
    else:
        buys_b = gates_b.get("BUY", 0)
        print(f"  Real (with AI)  BUYs: {buys_b} / {sum(gates_b.values())}")
        if buys_b > 0:
            print("  => NULL FALSIFIED: the engine CAN BUY once catalysts flow.")
            print("     0-BUY was catalyst supply (gates #14/#18). Defect A's restoration matters.")
        else:
            moved = sum(1 for tk in real
                        if tk in baseline
                        and (real[tk].get("verdict_state") != baseline[tk].get("verdict_state")))
            print("  => NULL SURVIVES: still 0 BUYs even WITH catalysts.")
            print(f"     {moved} tickers changed verdict between passes.")
            print("     If PASS B is dominated by REFUSED-EV / BELOW-THRESHOLD / NEGATIVE-EV,")
            print("     the binding constraint is the EV/conviction ARITHMETIC, not catalyst")
            print("     supply — and none of the recent fixes address that (Defect D made the")
            print("     EV hurdle STRICTER). That would be the real defect to attack next.")
    print("\n>>> Paste this ENTIRE output back.\n")


if __name__ == "__main__":
    sys.exit(main() or 0)

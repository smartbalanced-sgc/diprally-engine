"""Multi-ticker orchestrator (W5 PR #31) — library module.

Subprocess-based two-phase batch driver. See tools/orchestrate.py for
the CLI entry point. This module exposes the pure-Python pieces so
they can be unit-tested without spawning subprocesses (snapshot
parsing, summary formatting) plus the high-level run_batch() function
the CLI invokes.

Architecture:
  Phase 1: T0 snapshot collection per ticker.
  Broker:  src.broker.allocate() determines tiers under the cap.
  Phase 2: AI dispatch at each broker-assigned tier (T0 tickers skip).

Per-ticker subprocess logs land in output/orchestrator_<ts>/. The
summary table is returned by run_batch() AND written to SUMMARY.txt
inside the run dir. W5 PR #32 adds generate_aggregate_dashboard() —
an index.html cross-ticker ranking page with links to each ticker's
own dashboard.
"""
from __future__ import annotations

import concurrent.futures
import csv
import html
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.broker import BrokerAllocation, BrokerSnapshot

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SNAPSHOT_RE = re.compile(r"^BROKER_SNAPSHOT_JSON=(\{.*\})\s*$", re.MULTILINE)


@dataclass
class TickerRun:
    """One ticker's two-phase result. Mutable so the orchestrator
    can fill phase-2 fields after phase-1 + broker decision.

    PR #57 adds `delisted` flag — distinguishes "ticker is gone from
    market" (operator should remove from universe) from "engine
    failed for transient reason" (operator should investigate).
    """
    ticker: str
    phase1_returncode: Optional[int] = None
    snapshot: Optional[BrokerSnapshot] = None
    phase1_error: Optional[str] = None
    assigned_tier: str = "T0"
    phase2_returncode: Optional[int] = None
    phase2_error: Optional[str] = None
    elapsed_seconds: float = 0.0
    delisted: bool = False
    log_path: Optional[Path] = None


def _run_subprocess(cmd: list[str], log_path: Path,
                    timeout_seconds: int = 600) -> tuple[int, str, str]:
    """Run cmd, tee stdout+stderr to log_path, return (returncode, stdout, stderr)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        logf.write(f"=== {datetime.now().isoformat()} ===\n")
        logf.write(f"CMD: {' '.join(cmd)}\n")
        logf.write("=" * 78 + "\n")
        logf.flush()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout_seconds, cwd=_REPO_ROOT,
            )
        except subprocess.TimeoutExpired as e:
            logf.write(f"\nTIMEOUT after {timeout_seconds}s\n")
            return 124, e.stdout or "", e.stderr or ""
        logf.write(proc.stdout)
        if proc.stderr:
            logf.write("\n--- STDERR ---\n")
            logf.write(proc.stderr)
        return proc.returncode, proc.stdout, proc.stderr


def parse_snapshot(stdout: str) -> Optional[BrokerSnapshot]:
    """Extract the BROKER_SNAPSHOT_JSON= line and parse it. Returns None
    when the marker is absent, the JSON is malformed, or required
    fields are missing — caller treats this as a Phase 1 failure."""
    match = _SNAPSHOT_RE.search(stdout)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return BrokerSnapshot(
            ticker=str(data["ticker"]).upper(),
            ambiguity=float(data["ambiguity"]),
            qualifies_for_t2_plus=bool(data["qualifies_for_t2_plus"]),
            sigma_class=str(data["sigma_class"]),
            # PR #73: optional, defaults to False on older snapshot JSON
            limited_history=bool(data.get("limited_history", False)),
        )
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _phase1_single(ticker: str, run_dir: Path) -> TickerRun:
    """Phase 1: T0 collection for one ticker."""
    log_path = run_dir / f"{ticker}.phase1.log"
    cmd = [
        sys.executable, "tools/run.py", ticker,
        "--tier", "T0", "--emit-snapshot",
    ]
    t0 = time.time()
    rc, stdout, stderr = _run_subprocess(cmd, log_path)
    elapsed = time.time() - t0
    result = TickerRun(
        ticker=ticker, phase1_returncode=rc,
        log_path=log_path, elapsed_seconds=elapsed,
    )
    if rc != 0:
        err = (stderr or stdout or "").strip().splitlines()
        result.phase1_error = next(
            (line for line in reversed(err) if line.startswith("ERROR")),
            err[-1] if err else f"returncode {rc}",
        )
        # PR #57: distinguish DELISTED from generic failures so the
        # dashboard doesn't surface noise. yfinance fallback emits
        # "possibly delisted; no timezone found" on bankrupt /
        # de-listed names; FMP returns 404. Either marker → delisted.
        combined = (stderr or "") + "\n" + (stdout or "")
        if _looks_delisted(combined, ticker):
            result.delisted = True
        return result
    result.snapshot = parse_snapshot(stdout)
    if result.snapshot is None:
        result.phase1_error = "BROKER_SNAPSHOT_JSON missing from stdout"
    return result


def _looks_delisted(text: str, ticker: str) -> bool:
    """PR #57: pattern-match delisted-ticker markers in subprocess
    output. Conservative — only flips on STRONG indicators (the
    'delisted' substring is yfinance's own framing; FMP 404 on
    profile is the other strong signal). Doesn't flip on transient
    network errors, AI failures, or unrelated FetchError types."""
    if not text:
        return False
    lower = text.lower()
    if "delisted" in lower:
        return True
    # FMP 404 on the ticker's profile endpoint = ticker not in the
    # provider's universe. Distinguish from other 404s (e.g. missing
    # earnings calendar) by requiring the symbol in the URL.
    if "404" in lower and (
        f"profile?symbol={ticker.lower()}" in lower
        or f"profile/{ticker.lower()}" in lower
    ):
        return True
    return False


def _phase2_single(ticker: str, tier: str, run_dir: Path
                    ) -> tuple[int, Optional[str]]:
    """AI dispatch at the broker-assigned tier. Skips T0 (Phase 1 already
    produced the report). Returns (returncode, error_or_None)."""
    if tier == "T0":
        return 0, None
    log_path = run_dir / f"{ticker}.phase2.log"
    cmd = [sys.executable, "tools/run.py", ticker, "--tier", tier]
    rc, _, stderr = _run_subprocess(cmd, log_path)
    if rc != 0:
        err_lines = (stderr or "").strip().splitlines()
        err = next(
            (line for line in reversed(err_lines) if line.startswith("ERROR")),
            err_lines[-1] if err_lines else f"returncode {rc}",
        )
        return rc, err
    return rc, None


def run_phase1(tickers: list[str], run_dir: Path,
               max_parallel: int = 1,
               progress=print) -> list[TickerRun]:
    """Phase 1 driver. max_parallel=1 is sequential (predictable HTTP cache
    behavior); >1 uses a thread pool. progress(str) is called with
    per-ticker status lines (default: print)."""
    results: list[TickerRun] = [None] * len(tickers)  # type: ignore
    progress(f"\nPhase 1 — T0 snapshot collection across {len(tickers)} tickers")
    progress("-" * 78)
    if max_parallel <= 1:
        for i, t in enumerate(tickers):
            progress(f"  [{i+1}/{len(tickers)}] {t} ...")
            r = _phase1_single(t, run_dir)
            results[i] = r
            if r.snapshot:
                progress(
                    f"     ambiguity={r.snapshot.ambiguity:.2f} "
                    f"σ={r.snapshot.sigma_class} "
                    f"T2+qual={'✓' if r.snapshot.qualifies_for_t2_plus else ' '} "
                    f"({r.elapsed_seconds:.0f}s)"
                )
            else:
                tag = "DELISTED" if r.delisted else f"FAILED — {r.phase1_error}"
                progress(f"     {tag}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
            future_to_idx = {ex.submit(_phase1_single, t, run_dir): i
                             for i, t in enumerate(tickers)}
            for fut in concurrent.futures.as_completed(future_to_idx):
                i = future_to_idx[fut]
                r = fut.result()
                results[i] = r
                if r.snapshot:
                    tag = f"ambiguity={r.snapshot.ambiguity:.2f}"
                elif r.delisted:
                    tag = "DELISTED"
                else:
                    tag = f"FAILED — {r.phase1_error}"
                progress(f"  {r.ticker:<8} {tag}  ({r.elapsed_seconds:.0f}s)")
    return results


def run_phase2(allocation: BrokerAllocation, results: list[TickerRun],
               run_dir: Path, max_parallel: int = 1,
               progress=print) -> None:
    """AI dispatch driver. Mutates results in-place with phase2 outcomes."""
    ai_tickers = [r for r in results
                  if r.snapshot is not None
                  and allocation.assignments.get(r.ticker, "T0") != "T0"]
    if not ai_tickers:
        progress("\nPhase 2 — no tickers assigned to AI tiers; skipping.")
        return
    progress(f"\nPhase 2 — AI dispatch for {len(ai_tickers)} tickers at broker-assigned tiers")
    progress("-" * 78)
    if max_parallel <= 1:
        for r in ai_tickers:
            tier = allocation.assignments[r.ticker]
            r.assigned_tier = tier
            progress(f"  {r.ticker:<8} → {tier} ...")
            t0 = time.time()
            rc, err = _phase2_single(r.ticker, tier, run_dir)
            elapsed = time.time() - t0
            r.phase2_returncode = rc
            r.phase2_error = err
            r.elapsed_seconds += elapsed
            progress(f"     {'OK' if rc == 0 else 'FAILED — ' + (err or '')}  ({elapsed:.0f}s)")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futures = {}
            for r in ai_tickers:
                tier = allocation.assignments[r.ticker]
                r.assigned_tier = tier
                futures[ex.submit(_phase2_single, r.ticker, tier, run_dir)] = r
            for fut in concurrent.futures.as_completed(futures):
                r = futures[fut]
                rc, err = fut.result()
                r.phase2_returncode = rc
                r.phase2_error = err
                progress(f"  {r.ticker:<8} → {r.assigned_tier}  "
                         + ("OK" if rc == 0 else f"FAILED — {err}"))


def format_summary(results: list[TickerRun],
                    allocation: Optional[BrokerAllocation]) -> str:
    """Build the run summary table — pure function, deterministic."""
    lines = []
    lines.append("=" * 78)
    lines.append(
        f"ORCHESTRATOR SUMMARY — {datetime.now():%Y-%m-%d %H:%M}"
    )
    lines.append("=" * 78)
    n_ok_p1 = sum(1 for r in results if r.snapshot is not None)
    n_fail_p1 = len(results) - n_ok_p1
    lines.append(f"Phase 1 (T0 snapshots): {n_ok_p1} ok, {n_fail_p1} failed")
    if allocation is not None:
        lines.append(
            f"Broker:  spend ${allocation.spent_usd:.2f} of ${allocation.cap_usd:.2f} cap "
            f"({allocation.spent_usd / allocation.cap_usd * 100:.0f}%)"
        )
    n_ok_p2 = sum(1 for r in results
                  if r.phase2_returncode == 0 and r.assigned_tier != "T0")
    n_fail_p2 = sum(1 for r in results
                    if r.phase2_returncode not in (None, 0))
    lines.append(f"Phase 2 (AI runs):      {n_ok_p2} ok, {n_fail_p2} failed")
    lines.append("")
    lines.append(
        f"  {'Ticker':<8} {'σ-class':<8} {'Tier':<5} {'Ambig':>7} {'T2+qual':>8} {'Status':<20}"
    )
    lines.append("  " + "-" * 70)
    for r in sorted(results, key=lambda x: (
        -(x.snapshot.ambiguity if x.snapshot else -1.0), x.ticker
    )):
        ambig = f"{r.snapshot.ambiguity:.2f}" if r.snapshot else "  -  "
        sigma_cls = r.snapshot.sigma_class if r.snapshot else "?"
        qual = ("✓" if r.snapshot and r.snapshot.qualifies_for_t2_plus
                else (" " if r.snapshot else "?"))
        if r.snapshot is None:
            if r.delisted:
                status = "DELISTED — remove from universe"
            else:
                status = f"P1 FAIL: {r.phase1_error or 'unknown'}"
        elif r.phase2_returncode not in (None, 0):
            status = f"P2 FAIL: {r.phase2_error or 'unknown'}"
        else:
            status = "OK"
        lines.append(
            f"  {r.ticker:<8} {sigma_cls:<8} {r.assigned_tier:<5} "
            f"{ambig:>7} {qual:>8} {status:<20}"
        )
    return "\n".join(lines)


# =============================================================================
# Aggregate dashboard (W5 PR #32)
# =============================================================================

@dataclass
class TickerDecision:
    """One row in the aggregate dashboard. Combines a TickerRun with
    the latest CSV row from output/round_trip_history_<TICKER>.csv —
    so the dashboard reflects what the engine actually persisted, not
    a parallel computation."""
    ticker: str
    sigma_class: str
    tier: str
    ambiguity: Optional[float]
    qualifies_for_t2_plus: Optional[bool]
    spot: Optional[float]
    dip_target: Optional[float]
    rally_target: Optional[float]
    p_round_trip: Optional[float]
    ev_bps_of_dip: Optional[float]
    verdict: str               # BUY / WAIT / REFUSED-EV / REFUSED-TREND / REFUSED-METHOD / FAIL
    dashboard_href: Optional[str] = None
    status_note: str = ""      # human-readable extra context
    # PR #86 — dual-EV strategy fields. verdict_subtype: "DIRECT" /
    # "WAIT-FOR-DIP" indicating which entry strategy maximized EV.
    # ev_direct_bps / ev_wait_bps surface BOTH so the operator can see
    # the alternative. expected_rally_date / expected_dip_date are
    # CALENDAR DATES (per user spec — no trading-day numbers).
    verdict_subtype: str = "DIRECT"
    ev_direct_bps: Optional[float] = None
    ev_wait_bps: Optional[float] = None
    p_dip_filled: Optional[float] = None
    p_rally_hit: Optional[float] = None
    expected_rally_date: Optional[str] = None  # e.g. "Jun 18, 2026"
    expected_dip_date: Optional[str] = None


_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"


def _latest_history_row(ticker: str) -> Optional[dict]:
    """Read the last CSV row from output/round_trip_history_<TICKER>.csv.
    Returns None when the file is missing or empty — the dashboard
    renders a degraded row in that case."""
    path = _OUTPUT_ROOT / f"round_trip_history_{ticker}.csv"
    if not path.exists():
        return None
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    # Latest by date — sort defensively in case rows aren't in order.
    rows.sort(key=lambda r: r.get("date", ""))
    return rows[-1]


def _history_as_price_df(ticker: str):
    """Return a pandas DataFrame with 'Date' + 'Close' columns covering
    the last ~90 daily bars for `ticker`.

    2026-05-24 audit fix: previously read from
    output/round_trip_history_<TICKER>.csv (the engine's own outcome
    log) which only has 1 row per engine-run. With <5 engine runs the
    portfolio gate received None for every ticker and silently never
    deduplicated — defeating sacred #6's substitute-idea protection
    until day 5+ of running. Worse, even at day 30 only 30 snapshots
    would be available, vs the gate's design target of 60d daily-bar
    correlation.

    Fixed: now calls fetch_history() to pull real FMP/yfinance daily
    bars (years of history available immediately). Correlation gate
    becomes functional on day 1 — exactly as the W8 design intended.

    Returns None when FMP+yfinance both fail (gate accepts defensively
    when correlation cannot be computed)."""
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return None
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        # No API access — fall back to engine-CSV best-effort to keep
        # the gate alive in degraded mode (e.g. unit tests, --no-fetch).
        return _history_from_csv_fallback(ticker)
    try:
        from src.data_fetch import fetch_history
        # 2026-05-25 bug fix: fetch_history's lookback_days param is
        # CALENDAR days, but the portfolio gate's correlation_window_days
        # is TRADING days. 90 calendar days ≈ 63 trading days, which is
        # below the gate's minimum-bars requirement, causing the gate to
        # defensively accept every BUY and miss real correlations.
        #
        # Diagnostic that surfaced the bug: AMAT/LRCX have empirical
        # ρ=0.911 over 90 trading days but the gate said "0 correlation
        # pairs noted" because fetch_history(90) returned only 63 bars.
        #
        # Fix: request ~140 calendar days to ensure ≥90 trading days
        # are available (90 × 7/5 = 126, +14 buffer for holidays).
        df = fetch_history(ticker, api_key, lookback_days=140)
        if df is None or len(df) < 30:
            return None
        return df
    except Exception:
        # Provider failure — try CSV fallback rather than blocking the gate
        return _history_from_csv_fallback(ticker)


def _history_from_csv_fallback(ticker: str):
    """Read engine outcome CSV. Pre-audit-2026-05-24 implementation
    retained as a degraded fallback when FMP+yfinance are unavailable
    (CI / tests / network-down). Returns DataFrame with 'Date' + 'Close'
    or None when the file is absent or has < 5 rows."""
    path = _OUTPUT_ROOT / f"round_trip_history_{ticker}.csv"
    if not path.exists():
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            d = r.get("date", "")
            s = r.get("spot", "")
            if not d or not s:
                continue
            try:
                rows.append({"Date": pd.to_datetime(d[:10]),
                             "Close": float(s)})
            except (ValueError, TypeError):
                continue
    if len(rows) < 5:
        return None
    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    return df


def _parse_float(s, default=None):
    if s is None or s == "":
        return default
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _decision_from_run(run: TickerRun) -> TickerDecision:
    """Combine a TickerRun + the ticker's latest CSV row into a single
    aggregate-dashboard record."""
    snap = run.snapshot
    row = _latest_history_row(run.ticker) if snap is not None else None
    decision = TickerDecision(
        ticker=run.ticker,
        sigma_class=snap.sigma_class if snap else "?",
        tier=run.assigned_tier,
        ambiguity=snap.ambiguity if snap else None,
        qualifies_for_t2_plus=snap.qualifies_for_t2_plus if snap else None,
        spot=_parse_float(row.get("spot")) if row else None,
        dip_target=_parse_float(row.get("recommended_dip")) if row else None,
        rally_target=_parse_float(row.get("recommended_rally")) if row else None,
        p_round_trip=_parse_float(row.get("p_round_trip")) if row else None,
        ev_bps_of_dip=None,
        verdict="FAIL",
        dashboard_href=f"../{run.ticker.lower()}_dipnrally_dashboard.html",
        status_note="",
    )
    if row:
        ev_pct = _parse_float(row.get("ev_pct_of_dip"))
        if ev_pct is not None:
            decision.ev_bps_of_dip = ev_pct * 10000.0
        # PR #86 — dual-EV fields. Empty in legacy CSV rows (pre-#86).
        decision.verdict_subtype = (row.get("verdict_subtype") or "DIRECT").strip()
        decision.ev_direct_bps = _parse_float(row.get("ev_direct_bps"))
        decision.ev_wait_bps = _parse_float(row.get("ev_wait_bps"))
        decision.p_dip_filled = _parse_float(row.get("p_dip_filled"))
        decision.p_rally_hit = _parse_float(row.get("p_rally_hit"))
        # Compute calendar dates for expected timing.
        days_to_dip = _parse_float(row.get("expected_days_to_dip"))
        # expected_days_to_rally not currently in CSV — derive from dip
        # + dip_to_rally if needed; for now leave None when not available.
        try:
            from src.market_calendar import add_trading_days
            from datetime import datetime as _dt
            today_d = _dt.now().date()
            if days_to_dip is not None and days_to_dip > 0:
                d = add_trading_days(today_d, int(round(days_to_dip)))
                decision.expected_dip_date = d.strftime("%b %d, %Y")
            # For the rally, use ev_pct_of_dip horizon midpoint heuristic
            # (median of expected_days_to_dip + days_dip_to_rally if both
            # present, else just horizon_days estimate). Simpler: use
            # the engine's horizon as the upper bound.
            horizon = int(row.get("horizon_days") or 20)
            # Median rally date — use horizon midpoint as a rough proxy
            # when fine-grained data isn't in CSV.
            r_days = max(1, horizon // 2)
            d_rally = add_trading_days(today_d, r_days)
            decision.expected_rally_date = d_rally.strftime("%b %d, %Y")
        except Exception:
            pass
    # Verdict: engine writes verdict_state to CSV directly (audit fix
    # 2026-05-24). Read it from CSV rather than reconstructing from
    # dip/EV alone — the old reconstruction silently misclassified
    # sacred-#14 / #18 / #16 refusals as BUY/WAIT.
    if snap is None:
        # PR #57: distinguish DELISTED from generic Phase 1 failure.
        if run.delisted:
            decision.verdict = "DELISTED"
            decision.status_note = (
                "Ticker delisted / removed from data provider — "
                "consider removing from config/diprally.yaml universe"
            )
        else:
            decision.verdict = "FAIL"
            decision.status_note = run.phase1_error or "Phase 1 failed"
    else:
        decision.verdict, decision.status_note = _verdict_from_row(row, decision)
    return decision


# Status-note templates per engine verdict_state. Kept in orchestrator
# (not engine) so the per-ticker report and aggregate dashboard can
# evolve their wording independently.
_VERDICT_NOTES = {
    "BUY": lambda d: (
        f"Dip ${d.dip_target:.2f} → Rally ${d.rally_target:.2f}, "
        f"P(round-trip) = {(d.p_round_trip or 0) * 100:.0f}%"
    ),
    "WAIT": lambda d: "No qualifying pair at current spot",
    "BELOW-THRESHOLD": lambda d: (
        f"Best-EV fallback ${d.dip_target or 0:.2f}/${d.rally_target or 0:.2f} — "
        f"did not meet strict conviction. DO NOT TRADE."
    ),
    "NEGATIVE-EV": lambda d: (
        f"${d.dip_target or 0:.2f}/${d.rally_target or 0:.2f} meets conviction "
        f"but EV {d.ev_bps_of_dip:+.1f}bps negative. SKIP."
        if d.ev_bps_of_dip is not None else "EV negative — SKIP"
    ),
    "REFUSED-EV": lambda d: (
        f"EV/dip {d.ev_bps_of_dip:.1f}bps below 50bps hurdle (sacred #13)"
        if d.ev_bps_of_dip is not None
        else "EV below 50bps hurdle (sacred #13)"
    ),
    "REFUSED-TREND": lambda d: (
        "Refused on sacred #14 trend filter — falling-knife regime with "
        "no in-horizon bullish/two-sided catalyst"
    ),
    "REFUSED-PARABOLA": lambda d: (
        "Refused on sacred #18 parabola filter — blow-off momentum with "
        "no in-horizon bearish de-rating catalyst"
    ),
    "REFUSED-METHOD": lambda d: (
        "Refused on sacred #16 — MC / PDE / closed-form disagree beyond "
        "tolerance. Investigate σ / drift inputs."
    ),
}


def _verdict_from_row(row: dict, decision) -> tuple[str, str]:
    """Read engine-emitted verdict_state from the latest CSV row.
    Falls back to legacy reconstruction if the column is absent (CSV
    written by a pre-audit engine build)."""
    raw = ((row or {}).get("verdict_state") or "").strip() if row else ""
    if raw in _VERDICT_NOTES:
        return raw, _VERDICT_NOTES[raw](decision)
    # Legacy CSV (no verdict_state column) — reconstruct from dip/EV.
    # This is the pre-audit logic, preserved for old history rows so
    # the dashboard doesn't break on a mixed-history universe.
    if decision.dip_target is None or decision.dip_target == 0.0:
        return "WAIT", "No qualifying pair at current spot"
    if decision.ev_bps_of_dip is not None and decision.ev_bps_of_dip < 50.0:
        return "REFUSED-EV", (
            f"EV/dip {decision.ev_bps_of_dip:.1f}bps below 50bps hurdle (sacred #13)"
        )
    return "BUY", (
        f"Dip ${decision.dip_target:.2f} → Rally ${decision.rally_target:.2f}, "
        f"P(round-trip) = {(decision.p_round_trip or 0) * 100:.0f}%"
    )


_VERDICT_COLORS = {
    "BUY":                "#1a7f37",     # green
    "WAIT":               "#6e7781",     # neutral gray
    "BELOW-THRESHOLD":    "#6e7781",     # neutral gray — best-EV fallback, do not trade
    "NEGATIVE-EV":        "#bc4c00",     # orange — meets conviction but loses on average
    "REFUSED-EV":         "#bc4c00",     # orange
    "REFUSED-TREND":      "#bc4c00",
    "REFUSED-PARABOLA":   "#bc4c00",
    "REFUSED-CORRELATED": "#8250df",     # purple (PR #55 — substitute idea)
    "REFUSED-METHOD":     "#cf222e",     # red
    "FAIL":               "#cf222e",
    "DELISTED":           "#57606a",     # darker gray — operator action ≠ trader action
}


_TRADING212_URL_BASE = "https://www.trading212.com/trading-instruments/invest/"


def _trading212_url(ticker: str) -> str:
    """Trading212 instrument URL for a US ticker. Most US tickers work
    as <TICKER>.US. Class-share tickers with dashes (MOG-A) are used
    as-is — Trading212's URL routing accepts both dash and underscore
    forms on these names."""
    return f"{_TRADING212_URL_BASE}{ticker.upper()}.US"


def _spot_source_counts() -> dict:
    """PR #85: scan today's CSV rows across all tickers and tally how
    many used live_quote vs daily_bar_fallback. Returns
    {'live_quote': N, 'daily_bar_fallback': M, 'unknown': K}. Empty
    dict on read failure."""
    today_iso = datetime.now().strftime("%Y-%m-%d")
    counts = {"live_quote": 0, "daily_bar_fallback": 0, "unknown": 0}
    try:
        for path in _OUTPUT_ROOT.glob("round_trip_history_*.csv"):
            try:
                with open(path) as f:
                    rows = list(csv.DictReader(f))
                if not rows:
                    continue
                last = rows[-1]
                # Only count rows from TODAY (sacred #11 dedup means at
                # most one row per ticker per day).
                if str(last.get("date", ""))[:10] != today_iso:
                    continue
                src = (last.get("spot_source") or "").strip()
                if src in counts:
                    counts[src] += 1
                else:
                    counts["unknown"] += 1
            except Exception:
                continue
    except Exception:
        pass
    return counts


def _spot_source_line() -> str:
    """Plain-English line indicating whether spot prices are live
    intraday quotes or carried-forward closes. PR #76: holiday-aware —
    distinguishes weekend, NYSE holiday (Memorial Day, Good Friday, etc.),
    half-day session, and regular session. PR #85: data-driven — reports
    the ACTUAL spot source for today's run, not a heuristic."""
    now = datetime.now()
    today = now.date()
    try:
        from src.market_calendar import (
            is_trading_day, last_trading_day, holiday_name, early_close_time,
        )
    except Exception:
        # Defensive: fall back to weekday-only logic if the calendar
        # module is unavailable (shouldn't happen post-PR #76).
        weekday = now.weekday()
        if weekday >= 5:
            return (
                "Spot prices: Prior close (markets closed weekend). Weekend "
                "quotes carry forward Friday's last regular-session close."
            )
        return (
            "Spot prices: Live FMP quote queried at run time "
            "(intraday may be delayed up to 15 min; after-hours = today's close)."
        )

    if not is_trading_day(today):
        last_open = last_trading_day(today)
        h = holiday_name(today)
        if today.weekday() >= 5:
            return (
                f"⚠ Spot prices: Markets closed (weekend). Quotes shown are "
                f"the last regular-session close from {last_open:%a %Y-%m-%d}. "
                f"Engine analysis is correct for that data; act on next "
                f"trading day's open with awareness of any overnight gap."
            )
        return (
            f"⚠ Spot prices: NYSE CLOSED today ({h or 'holiday'}). Quotes "
            f"shown are the last regular-session close from "
            f"{last_open:%a %Y-%m-%d}. Engine analysis is correct for that "
            f"data; act on next trading day's open with awareness of any "
            f"overnight gap."
        )

    # Today IS a trading day. Report what the engine actually fetched.
    counts = _spot_source_counts()
    live = counts.get("live_quote", 0)
    fallback = counts.get("daily_bar_fallback", 0)
    total_today = live + fallback + counts.get("unknown", 0)
    ec = early_close_time(today)
    session_note = (
        f" — NYSE half-day session (early close {ec.strftime('%H:%M')} ET)"
        if ec is not None else ""
    )
    if total_today == 0:
        # No CSV rows from today yet (orchestrator is rendering before
        # the per-ticker pipelines have written) — report the policy.
        return (
            f"Spot prices: live FMP /stable/quote per ticker, fall back "
            f"to last daily-bar close on failure{session_note}."
        )
    if fallback == 0:
        return (
            f"Spot prices: live FMP /stable/quote for {live}/{total_today} "
            f"tickers (current intraday or last session close){session_note}."
        )
    return (
        f"⚠ Spot prices: live FMP /stable/quote for {live}/{total_today} "
        f"tickers; {fallback} fell back to last daily-bar close (may be "
        f"stale by up to ~2h post-market-close — check those rows' "
        f"spot_source column){session_note}."
    )


def _ambiguity_tooltip_text(amb: float, tier: str) -> str:
    """Plain-English tooltip for ambiguity (PR #88 rewrite).

    Ambiguity is the math-layer's SELF-CONFIDENCE score. It is NOT a
    touch probability — it does NOT mean "X% of paths hit a target."
    It's a composite of:
      - σ-anchor disagreement: how much do GARCH / realized vol /
        options-IV disagree on this ticker's volatility?
      - Signal alignment: are the ~10 drift signals pointing the same
        direction or scattered?
      - Cross-check divergence: do MC, PDE, and closed-form math
        agree on touch probabilities?
    LOW ambiguity = math is internally consistent and confident.
    HIGH ambiguity = math sees conflicting signals; AI critique adds
    real value (which is why the broker uses ambiguity to decide tier).
    """
    if amb is None:
        return "Ambiguity unavailable for this ticker."
    if amb <= 0.10:
        bucket = "VERY LOW"
        narrative = "Math layer is extremely confident in its own outputs."
    elif amb <= 0.20:
        bucket = "LOW"
        narrative = "Math layer is highly confident."
    elif amb <= 0.40:
        bucket = "LOW-MEDIUM"
        narrative = "Moderate confidence — some signal disagreement."
    elif amb <= 0.50:
        bucket = "MEDIUM"
        narrative = "Moderate uncertainty — AI critique adds value here."
    elif amb <= 0.65:
        bucket = "MEDIUM-HIGH"
        narrative = "High uncertainty — math signals conflict; AI critique has strong leverage."
    else:
        bucket = "HIGH"
        narrative = "Very high uncertainty — math layer doesn't agree with itself."
    return (
        f"<strong>{amb:.2f} Ambiguity ({bucket}):</strong> Math-layer "
        f"SELF-confidence score — NOT a touch probability. {narrative} "
        f"Broker assigned tier {tier}."
    )


def _prt_tooltip_text(prt: float | None) -> str:
    """Tooltip for P(round-trip) — PR #88 plain-English rewrite.

    P(round-trip) is the probability that BOTH dip touches FIRST and
    rally touches AFTER over the horizon. It is NOT P(dip)×P(rally)
    because those events are conditional — a path that dipped has
    already had downside drift, so its conditional rally probability
    is LOWER than the unconditional. That's why you can have P(dip)=86%
    and P(rally)=67% but P(round-trip)=38% (not 57% from independent
    multiplication).
    """
    if prt is None:
        return "P(round-trip) unavailable for this ticker."
    pct = prt * 100
    if pct >= 60:
        nuance = "Strong joint probability of both legs filling."
    elif pct >= 55:
        nuance = "Solid joint probability."
    elif pct >= 50:
        nuance = "Just above coin-flip."
    else:
        nuance = "Below coin-flip — weaker setup."
    try:
        from src.config import DEFAULT_HORIZON_DAYS as _H
        h = int(_H)
    except Exception:
        h = 20
    return (
        f"<strong>{pct:.0f}% P(round-trip):</strong> {pct:.0f}% of 100,000 "
        f"simulated {h}-trading-day paths touch dip FIRST then rally AFTER. "
        f"NOT equal to P(dip)×P(rally): a path that dipped tends to have "
        f"downward drift, so conditional rally probability is lower. {nuance}"
    )


def _ev_tooltip_text(ev_bps: float | None) -> str:
    """Tooltip for EV bps value, expressed in both bps and %."""
    if ev_bps is None:
        return "EV unavailable for this ticker."
    pct = ev_bps / 100.0
    if ev_bps >= 50:
        nuance = "Profitable on average — clears the trade-quality hurdle."
    elif ev_bps >= 0:
        nuance = "Marginally positive; thin margin above zero."
    elif ev_bps >= -50:
        nuance = "Slightly negative — loses small amount on average."
    else:
        nuance = "Significantly negative — loses money on average after friction."
    return (
        f"<strong>{ev_bps:+.0f} bps EV ({pct:+.2f}%):</strong> If you "
        f"took this trade 100 times under identical conditions, your "
        f"average return per trade — after weighting wins, losses, "
        f"no-fill paths, and round-trip friction — would be "
        f"{pct:+.2f}% of your entry price. {nuance}"
    )


# Verdict sort priority — BUYs first (by EV desc), then REFUSED-CORRELATED
# (substitute idea, was a BUY), then refusals by EV desc, then WAIT,
# then operator-action verdicts (FAIL/DELISTED) at the bottom.
_VERDICT_SORT_PRIORITY = {
    "BUY":                100,
    "REFUSED-CORRELATED": 90,
    "BELOW-THRESHOLD":    80,
    "NEGATIVE-EV":        70,
    "REFUSED-EV":         60,
    "REFUSED-PARABOLA":   50,
    "REFUSED-TREND":      40,
    "REFUSED-METHOD":     30,
    "WAIT":               20,
    "FAIL":               10,
    "DELISTED":           0,
}


def _sort_decisions_for_dashboard(decisions: list) -> list:
    """PR #88 — sort by conviction × gain (operator-actionable ordering).

    Priority bands (high → low):
      1. BUYs            — sorted by (P_rally × conditional_gain) desc.
         This is the operator's true "best opportunity" ranking: high
         probability of meaningful gain rises first.
      2. REFUSED-EV      — sorted by EV bps DESC (closest to hurdle first;
                           these are the names worth watching).
      3. REFUSED-PARABOLA — sorted by EV bps desc within this group
                           (the ones the parabola filter blocked).
      4. REFUSED-METHOD / REFUSED-TREND — math couldn't agree.
      5. WAIT            — no setup found at any qualifying threshold.
      6. FAIL / DELISTED — data layer problems.
    """
    BAND_BUY = 100
    BAND_REFUSED_EV = 70
    BAND_REFUSED_PARABOLA = 60
    BAND_REFUSED_METHOD = 50
    BAND_REFUSED_TREND = 40
    BAND_WAIT = 20
    BAND_FAIL = 0
    BAND_REFUSED_CORRELATED = 80   # was a BUY-equivalent before the gate dropped it
    BAND_MAP = {
        "BUY": BAND_BUY,
        "REFUSED-CORRELATED": BAND_REFUSED_CORRELATED,
        "REFUSED-EV": BAND_REFUSED_EV,
        "REFUSED-PARABOLA": BAND_REFUSED_PARABOLA,
        "REFUSED-METHOD": BAND_REFUSED_METHOD,
        "REFUSED-TREND": BAND_REFUSED_TREND,
        "WAIT": BAND_WAIT,
        "FAIL": BAND_FAIL,
        "DELISTED": BAND_FAIL,
    }

    def conviction_gain_score(d):
        """Operator-actionable score for BUY ranking: P(rally hit) ×
        conditional gain pct. Higher = more attractive. Falls back to
        EV bps when conviction fields aren't populated (legacy CSV
        rows / test fixtures)."""
        if (d.p_rally_hit is not None and d.spot is not None
                and d.rally_target is not None and d.spot > 0):
            gain_pct = (d.rally_target - d.spot) / d.spot
            return float(d.p_rally_hit) * float(gain_pct) * 1e4  # → ~bps
        # Fallback: EV bps. Same monotonic intent (higher = better).
        return d.ev_bps_of_dip if d.ev_bps_of_dip is not None else -1e9

    def sort_key(d):
        band = BAND_MAP.get(d.verdict, BAND_FAIL)
        if d.verdict == "BUY":
            score = conviction_gain_score(d)
        else:
            # Within refusal bands, EV closer to zero (less negative) ranks first
            score = d.ev_bps_of_dip if d.ev_bps_of_dip is not None else -1e9
        return (-band, -score, d.ticker)

    return sorted(decisions, key=sort_key)


def _render_dual_ev_detail_row(d) -> str:
    """PR #87 — institutional detail row under each ticker's main row.
    Shows BOTH entry strategies' EV breakdown, win/lose probabilities,
    and expected calendar dates so the operator can SEE why each
    verdict was reached and which leg failed (for refusals)."""
    # Skip detail row for failed phase 1 / delisted tickers (no math data).
    if d.ev_direct_bps is None and d.ev_wait_bps is None:
        return ""

    def fmt_bps(v):
        if v is None:
            return "—"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.0f} bps ({sign}{v/100:.2f}%)"

    def fmt_pct(v):
        if v is None:
            return "—"
        return f"{v*100:.0f}%"

    # Color-code EVs: green if positive, red if negative, gray if None.
    def ev_color(v):
        if v is None:
            return "#777"
        return "#26a269" if v >= 0 else "#c01c28"

    direct_ev_str = fmt_bps(d.ev_direct_bps)
    wait_ev_str = fmt_bps(d.ev_wait_bps)
    direct_color = ev_color(d.ev_direct_bps)
    wait_color = ev_color(d.ev_wait_bps)

    # PR #88 — removed ★ winner marker (always-WAIT-wins in current
    # data was misleading). The verdict_subtype is still in the CSV
    # for analytics; UI just shows both strategies cleanly.
    direct_winner = ""
    wait_winner = ""

    # Conditional gain for DIRECT (rally - spot) and WAIT (rally - dip)
    direct_gain_str = "—"
    wait_gain_str = "—"
    if d.spot and d.rally_target and d.spot > 0:
        direct_gain_pct = (d.rally_target - d.spot) / d.spot * 100
        direct_gain_str = f"+{direct_gain_pct:.2f}% if rally hits"
    if d.dip_target and d.rally_target and d.dip_target > 0:
        wait_gain_pct = (d.rally_target - d.dip_target) / d.dip_target * 100
        wait_gain_str = f"+{wait_gain_pct:.2f}% if filled & rally hits"

    p_rally_str = fmt_pct(d.p_rally_hit) if d.p_rally_hit is not None else "—"
    p_dip_str = fmt_pct(d.p_dip_filled) if d.p_dip_filled is not None else "—"

    # Calendar timing
    rally_date = d.expected_rally_date or "—"
    dip_date = d.expected_dip_date or "—"

    # Pre-format dollar amounts to avoid nested f-string ternary issues
    spot_fmt = f"${d.spot:.2f}" if d.spot else "$—"
    rally_fmt = f"${d.rally_target:.2f}" if d.rally_target else "$—"
    dip_fmt = f"${d.dip_target:.2f}" if d.dip_target else "$—"

    return f"""      <tr class="dual-ev-detail" data-verdict="{html.escape(d.verdict)}">
        <td colspan="6" class="dual-ev-cell">
          <div class="dual-ev-grid">
            <div class="dual-ev-col">
              <div class="dual-ev-strategy">DIRECT entry @ {spot_fmt} <span class="dual-ev-winner">{direct_winner}</span></div>
              <div class="dual-ev-row"><span class="dual-ev-label">Target:</span> {rally_fmt} &nbsp; <span class="dual-ev-detail-text">{direct_gain_str}</span></div>
              <div class="dual-ev-row"><span class="dual-ev-label">P(rally hits):</span> {p_rally_str}</div>
              <div class="dual-ev-row"><span class="dual-ev-label">EV (unconditional):</span> <span style="color:{direct_color}">{direct_ev_str}</span></div>
              <div class="dual-ev-row"><span class="dual-ev-label">Expected rally:</span> {rally_date}</div>
            </div>
            <div class="dual-ev-col">
              <div class="dual-ev-strategy">WAIT-FOR-DIP @ {dip_fmt} <span class="dual-ev-winner">{wait_winner}</span></div>
              <div class="dual-ev-row"><span class="dual-ev-label">P(dip fills):</span> {p_dip_str}</div>
              <div class="dual-ev-row"><span class="dual-ev-label">Target:</span> {rally_fmt} &nbsp; <span class="dual-ev-detail-text">{wait_gain_str}</span></div>
              <div class="dual-ev-row"><span class="dual-ev-label">EV (unconditional, incl. no-fill paths):</span> <span style="color:{wait_color}">{wait_ev_str}</span></div>
              <div class="dual-ev-row"><span class="dual-ev-label">Expected dip:</span> {dip_date}</div>
            </div>
          </div>
        </td>
      </tr>"""


def _render_dashboard_html(decisions: list, allocation,
                            href_prefix: str = "") -> str:
    """Render the aggregate-dashboard HTML.

    2026-05-24 refresh: dark theme + subtle pattern, collapsible legend
    that overlays content (doesn't push), Trading212 ticker links,
    per-ticker engine dashboard linked via the Dip cell, edge-aware
    tooltips on Ambiguity/P(RT)/EV cells, scroll-to-top button, mobile-
    responsive card layout under 768px. Default sort surfaces BUYs first
    by highest EV.
    """
    spent = allocation.spent_usd if allocation else 0.0
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Annual projection at 252 trading days
    annual_estimate = spent * 252

    # Sort decisions: BUYs first (highest EV desc), then refusals, then
    # WAIT / FAIL / DELISTED at the bottom.
    sorted_decisions = _sort_decisions_for_dashboard(decisions)

    # PR #88 — navigation chip strip above the table. Clicking a ticker
    # scrolls to its row in the table. Each chip is color-coded by
    # verdict so the operator can visually scan the universe state.
    nav_chips = []
    for d in sorted_decisions:
        chip_color = _VERDICT_COLORS.get(d.verdict, "#6e7681")
        nav_chips.append(
            f'<a href="#row-{d.ticker}" class="ticker-chip" '
            f'style="border-color:{chip_color}" '
            f'title="{html.escape(d.verdict)}">{html.escape(d.ticker)}</a>'
        )
    ticker_nav_html = " ".join(nav_chips)

    rows_html = []
    for d in sorted_decisions:
        color = _VERDICT_COLORS.get(d.verdict, "#6e7681")
        ambig_str = f"{d.ambiguity:.2f}" if d.ambiguity is not None else "—"
        spot_str = f"${d.spot:,.2f}" if d.spot is not None else "—"
        rally_str = f"${d.rally_target:,.2f}" if d.rally_target else "—"
        prt_str = (f"{d.p_round_trip*100:.0f}%"
                   if d.p_round_trip is not None else "—")
        if d.ev_bps_of_dip is not None:
            ev_str = f"{d.ev_bps_of_dip:+.1f} ({d.ev_bps_of_dip/100:+.2f}%)"
        else:
            ev_str = "—"

        # PR #88: detail-row dip price links to per-ticker dashboard
        # (was the table's Dip cell; column removed in this PR). The
        # link is preserved on dip_cell here for the detail row to
        # consume in _render_dual_ev_detail_row.
        dashboard_href = f"{href_prefix}{d.ticker.lower()}_dipnrally_dashboard.html"
        if d.dip_target:
            dip_cell = (
                f'<a href="{html.escape(dashboard_href)}">'
                f'${d.dip_target:,.2f}</a>'
            )
        else:
            dip_cell = "—"

        # Ticker cell → Trading212 (new tab) PLUS small per-ticker
        # dashboard chip 📊 (internal) so the operator can still reach
        # the per-ticker analysis with one click.
        t212 = _trading212_url(d.ticker)
        ticker_cell = (
            f'<a href="{html.escape(t212)}" target="_blank" rel="noopener">'
            f'{html.escape(d.ticker)}</a>'
            f' <a href="{html.escape(dashboard_href)}" class="ticker-details-link" '
            f'title="Per-ticker dip-rally analysis">📊</a>'
        )

        # Tooltips
        if d.ambiguity is not None:
            amb_tip = html.escape(
                _ambiguity_tooltip_text(d.ambiguity, d.tier), quote=False
            ).replace("&lt;strong&gt;", "<strong>").replace("&lt;/strong&gt;", "</strong>")
            ambig_cell = (
                f'<span class="tt">{ambig_str}'
                f'<span class="tt-content">{amb_tip}</span></span>'
            )
        else:
            ambig_cell = ambig_str

        if d.p_round_trip is not None:
            prt_tip = html.escape(
                _prt_tooltip_text(d.p_round_trip), quote=False
            ).replace("&lt;strong&gt;", "<strong>").replace("&lt;/strong&gt;", "</strong>")
            prt_cell = (
                f'<span class="tt">{prt_str}'
                f'<span class="tt-content">{prt_tip}</span></span>'
            )
        else:
            prt_cell = prt_str

        if d.ev_bps_of_dip is not None:
            ev_tip = html.escape(
                _ev_tooltip_text(d.ev_bps_of_dip), quote=False
            ).replace("&lt;strong&gt;", "<strong>").replace("&lt;/strong&gt;", "</strong>")
            ev_cell = (
                f'<span class="tt">{ev_str}'
                f'<span class="tt-content">{ev_tip}</span></span>'
            )
        else:
            ev_cell = ev_str

        # PR #88: trimmed main row. Spot/Dip/Rally/P(RT)/EV bps moved
        # to the detail row below (was duplicating data per your screenshot
        # critique). Main row carries only the headline categorical info.
        ticker_anchor = f'row-{d.ticker}'
        rows_html.append(f"""      <tr id="{ticker_anchor}" data-verdict="{html.escape(d.verdict)}">
        <td class="mobile-ticker">{ticker_cell}</td>
        <td data-label="σ-class">{html.escape(d.sigma_class)}</td>
        <td data-label="Tier">{html.escape(d.tier)}</td>
        <td class="num" data-label="Ambiguity">{ambig_cell}</td>
        <td class="mobile-verdict" data-label="Verdict"><span class="verdict" style="background:{color}">{html.escape(d.verdict)}</span></td>
        <td class="note mobile-note">{html.escape(d.status_note)}</td>
      </tr>
{_render_dual_ev_detail_row(d)}""")

    n_buy = sum(1 for d in decisions if d.verdict == "BUY")
    n_wait = sum(1 for d in decisions if d.verdict == "WAIT")
    n_refused = sum(1 for d in decisions if d.verdict.startswith("REFUSED"))
    n_fail = sum(1 for d in decisions if d.verdict == "FAIL")

    spot_source = _spot_source_line()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dip and Rally Engine — universe ranking ({now})</title>
<style>
:root {{
    --bg-dark: #0d1117;
    --bg-card: #161b22;
    --bg-card-elevated: #1c2128;
    --border: #30363d;
    --border-subtle: #21262d;
    --text-primary: #f0f6fc;
    --text-secondary: #c9d1d9;
    --text-tertiary: #adb6c0;
    --text-muted: #909dab;
    --link: #58a6ff;
    --link-hover: #79b8ff;
    --accent: #388bfd;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    margin: 0; padding: 32px 24px;
    color: var(--text-primary);
    background: var(--bg-dark);
    background-image:
        radial-gradient(circle at 1px 1px, rgba(255,255,255,0.04) 1px, transparent 0),
        radial-gradient(ellipse at 50% 0%, rgba(56,139,253,0.08) 0%, transparent 60%);
    background-size: 24px 24px, 100% 800px;
    background-attachment: fixed;
    min-height: 100vh; line-height: 1.5;
}}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ font-size: 26px; font-weight: 600; margin: 0 0 6px;
      color: var(--text-primary); letter-spacing: -0.4px; }}
.meta {{ color: var(--text-secondary); font-size: 13px; margin-bottom: 4px; }}
.meta .meta-sep {{ color: var(--text-muted); margin: 0 6px; }}
.meta-strong {{ color: var(--text-primary); font-weight: 500; }}
.spot-source {{ color: var(--text-tertiary); font-size: 12px;
                 margin-bottom: 20px; font-style: italic; }}
.summary-row {{ display: flex; gap: 12px; margin: 20px 0;
                 flex-wrap: wrap; align-items: flex-start; position: relative; }}
.summary-tile {{ background: var(--bg-card); border: 1px solid var(--border-subtle);
                  padding: 12px 18px; border-radius: 8px; font-size: 12px;
                  flex: 0 0 auto; color: var(--text-tertiary);
                  text-transform: uppercase; letter-spacing: 0.4px;
                  box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset; }}
.summary-tile strong {{ display: block; font-size: 22px; font-weight: 600;
                         color: var(--text-primary); text-transform: none;
                         letter-spacing: -0.2px; }}
.legend-wrapper {{ flex: 1 1 280px; min-width: 240px; position: relative;
                    background: var(--bg-card);
                    border: 1px solid var(--border-subtle);
                    border-radius: 8px; }}
.legend-toggle {{ width: 100%; background: none; border: none; cursor: pointer;
                   padding: 14px 18px; text-align: left; color: var(--text-primary);
                   font-size: 13px; font-weight: 500; font-family: inherit;
                   display: flex; align-items: center; gap: 10px; user-select: none; }}
.legend-toggle:hover {{ background: rgba(255,255,255,0.02); }}
.legend-toggle-arrow {{ font-size: 10px; color: var(--text-muted);
                         transition: transform 0.3s cubic-bezier(0.4,0,0.2,1);
                         display: inline-block; }}
.legend-wrapper.open .legend-toggle-arrow {{ transform: rotate(90deg); }}
.legend-body {{ position: absolute; top: 100%; left: 0; right: 0;
                 background: var(--bg-card-elevated); border: 1px solid var(--border);
                 border-radius: 8px; margin-top: 4px; z-index: 50;
                 box-shadow: 0 12px 28px rgba(0,0,0,0.55); overflow: hidden;
                 max-height: 0; opacity: 0; transform: translateY(-4px);
                 transition: max-height 0.4s cubic-bezier(0.4,0,0.2,1),
                             opacity 0.25s ease-out,
                             transform 0.3s cubic-bezier(0.4,0,0.2,1),
                             padding 0.3s ease;
                 padding: 0 20px; pointer-events: none; }}
.legend-wrapper.open .legend-body {{ max-height: 80vh; opacity: 1;
                                      transform: translateY(0);
                                      padding: 18px 20px; pointer-events: auto;
                                      overflow-y: auto; }}
.legend-section {{ padding: 10px 0; }}
.legend-section + .legend-section {{ border-top: 1px solid var(--border-subtle);
                                      margin-top: 8px; }}
.legend-section-title {{ display: block; font-weight: 600;
                          color: var(--text-primary); font-size: 11px;
                          text-transform: uppercase; letter-spacing: 0.8px;
                          margin-bottom: 14px; }}
.legend-verdict-row {{ display: flex; gap: 14px; margin-bottom: 14px;
                        align-items: flex-start; }}
.legend-verdict-row:last-child {{ margin-bottom: 0; }}
.legend-verdict-row .vchip {{ flex: 0 0 auto; display: inline-block;
                               padding: 3px 9px; border-radius: 4px;
                               color: #fff; font-size: 10px; font-weight: 600;
                               white-space: nowrap; margin-top: 2px;
                               min-width: 110px; text-align: center;
                               letter-spacing: 0.3px; }}
.legend-verdict-row .vdef {{ flex: 1; color: var(--text-secondary);
                              line-height: 1.6; font-size: 12.5px; }}
.legend-col-row {{ margin-bottom: 10px; line-height: 1.6;
                    color: var(--text-secondary); font-size: 12.5px; }}
.legend-col-row:last-child {{ margin-bottom: 0; }}
.legend-col-row strong {{ font-weight: 600; color: var(--text-primary); }}
.table-container {{ background: var(--bg-card); border: 1px solid var(--border-subtle);
                     border-radius: 8px; overflow: hidden;
                     box-shadow: 0 2px 6px rgba(0,0,0,0.2); }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
thead th {{ text-align: left; background: var(--bg-card-elevated);
             padding: 12px 14px; border-bottom: 1px solid var(--border);
             cursor: pointer; user-select: none; font-weight: 600;
             color: var(--text-primary); font-size: 11px;
             text-transform: uppercase; letter-spacing: 0.6px; }}
thead th:hover {{ background: rgb(40, 47, 58); }}
tbody td {{ padding: 11px 14px; border-bottom: 1px solid var(--border-subtle);
             vertical-align: middle; color: var(--text-secondary); }}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover {{ background: rgba(56,139,253,0.04); }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums;
          color: var(--text-primary); }}
td.note {{ color: var(--text-tertiary); font-size: 11.5px;
           max-width: 280px; line-height: 1.4; }}
.verdict {{ display: inline-block; color: rgb(255,255,255);
             padding: 3px 9px; border-radius: 12px; font-size: 11px;
             font-weight: 600; letter-spacing: 0.3px; white-space: nowrap; }}
a {{ color: var(--link); text-decoration: none; font-weight: 600; }}
a:hover {{ color: var(--link-hover); text-decoration: underline; }}
.tt {{ position: relative; cursor: help;
       text-decoration: underline dotted; text-underline-offset: 3px;
       text-decoration-color: var(--text-muted); }}
.tt .tt-content {{ visibility: hidden; opacity: 0;
                    background: rgb(20,22,27); color: var(--text-primary);
                    text-align: left; border: 1px solid var(--border);
                    border-radius: 6px; padding: 11px 13px;
                    position: fixed; z-index: 1000; max-width: 280px;
                    width: max-content; font-size: 11.5px; line-height: 1.55;
                    font-weight: 400; transition: opacity 0.15s;
                    pointer-events: none;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.6); }}
.tt:hover .tt-content {{ visibility: visible; opacity: 1; }}
.tt strong {{ color: var(--text-primary); }}
/* PR #88 — ticker navigation strip above the table */
.ticker-nav {{
    padding: 12px 16px;
    background: rgba(56,139,253,0.06);
    border-radius: 8px;
    margin: 16px 0;
    line-height: 2.2;
}}
.ticker-nav-label {{
    color: var(--text-muted);
    margin-right: 8px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.ticker-chip {{
    display: inline-block;
    padding: 3px 9px;
    margin: 0 2px;
    border-radius: 4px;
    border: 1px solid var(--border);
    border-left-width: 3px;
    color: var(--text-primary);
    text-decoration: none;
    font-size: 12px;
    font-weight: 500;
    background: rgba(255,255,255,0.02);
}}
.ticker-chip:hover {{
    background: rgba(56,139,253,0.12);
    color: var(--text-primary);
    transform: translateY(-1px);
    transition: all 0.12s;
}}
table#universe tr[id^="row-"] {{ scroll-margin-top: 80px; }}
.note-col {{ min-width: 220px; }}
/* PR #87 — dual-EV detail row under each ticker showing both entry strategies */
tr.dual-ev-detail td.dual-ev-cell {{
    background: rgba(56,139,253,0.04);
    border-top: 1px dashed var(--border);
    padding: 10px 16px 14px 16px;
}}
.dual-ev-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    font-size: 12px;
}}
.dual-ev-col {{
    border-left: 3px solid rgba(56,139,253,0.4);
    padding: 4px 12px 4px 12px;
}}
.dual-ev-strategy {{
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 6px;
    font-size: 13px;
}}
.dual-ev-row {{
    color: var(--text-secondary);
    margin: 2px 0;
}}
.dual-ev-label {{
    color: var(--text-muted);
    display: inline-block;
    min-width: 200px;
}}
.dual-ev-detail-text {{
    color: var(--text-primary);
    font-style: italic;
}}
.dual-ev-winner {{
    color: #d29922;
    font-weight: 600;
    margin-left: 6px;
}}
@media (max-width: 768px) {{
    .dual-ev-grid {{ grid-template-columns: 1fr; gap: 16px; }}
}}
.scroll-top {{ position: fixed; bottom: 24px; right: 24px;
                width: 46px; height: 46px; border-radius: 50%;
                background: rgba(56,139,253,0.92); color: #fff;
                border: 1px solid rgba(255,255,255,0.12); cursor: pointer;
                font-size: 20px; line-height: 1; opacity: 0;
                transform: translateY(12px);
                transition: opacity 0.3s ease,
                            transform 0.3s cubic-bezier(0.4,0,0.2,1),
                            background 0.2s ease;
                pointer-events: none;
                box-shadow: 0 6px 16px rgba(0,0,0,0.5);
                z-index: 200; display: flex; align-items: center;
                justify-content: center; padding-bottom: 2px;
                font-family: inherit; }}
.scroll-top.visible {{ opacity: 1; transform: translateY(0); pointer-events: auto; }}
.scroll-top:hover {{ background: var(--accent); }}
.scroll-top:active {{ transform: translateY(0) scale(0.95); }}
footer {{ margin-top: 32px; color: var(--text-tertiary); font-size: 11.5px;
           text-align: center; padding: 16px;
           border-top: 1px solid var(--border-subtle); }}
@media (max-width: 768px) {{
    body {{ padding: 16px 12px; }}
    h1 {{ font-size: 20px; }}
    .summary-row {{ gap: 8px; }}
    .summary-tile {{ padding: 8px 12px; font-size: 11px; }}
    .summary-tile strong {{ font-size: 18px; }}
    .legend-wrapper {{ flex-basis: 100%; }}
    .legend-verdict-row {{ display: block; margin-bottom: 14px;
                            overflow: hidden; }}
    .legend-verdict-row .vchip {{ float: left; margin: 0 10px 4px 0;
                                   min-width: 0; padding: 3px 8px; }}
    .legend-verdict-row .vdef {{ display: block; font-size: 12px;
                                  line-height: 1.55; }}
    .legend-section-title {{ margin-bottom: 12px; }}
    .legend-col-row {{ font-size: 12px; line-height: 1.55; }}
    table, thead, tbody, tr, td {{ display: block; }}
    thead {{ display: none; }}
    .table-container {{ background: transparent; border: none; box-shadow: none; }}
    tbody tr {{ background: var(--bg-card);
                 border: 1px solid var(--border-subtle);
                 border-radius: 10px; margin-bottom: 14px; padding: 16px;
                 box-shadow: 0 2px 6px rgba(0,0,0,0.25); }}
    tbody tr:hover {{ background: var(--bg-card); }}
    tbody td {{ padding: 6px 0; border-bottom: none;
                 text-align: left !important; font-size: 13px;
                 display: flex; justify-content: space-between;
                 gap: 12px; align-items: center;
                 color: var(--text-secondary); }}
    tbody td.num {{ color: var(--text-primary); }}
    tbody td:before {{ content: attr(data-label); font-weight: 600;
                        color: var(--text-tertiary); flex: 0 0 auto;
                        font-size: 10px; text-transform: uppercase;
                        letter-spacing: 0.5px; }}
    tbody td.mobile-ticker, tbody td.mobile-verdict {{ display: block;
                                                         padding: 0;
                                                         margin-bottom: 8px; }}
    tbody td.mobile-ticker:before, tbody td.mobile-verdict:before {{ content: none; }}
    tbody td.mobile-ticker {{ font-size: 20px; font-weight: 600; }}
    tbody td.mobile-ticker a {{ font-size: 20px; color: var(--link); }}
    tbody td.mobile-verdict {{ margin-bottom: 12px; padding-bottom: 12px;
                                border-bottom: 1px solid var(--border-subtle); }}
    tbody td.mobile-note {{ margin-top: 12px; padding-top: 12px;
                             border-top: 1px solid var(--border-subtle);
                             color: var(--text-tertiary); font-size: 11.5px;
                             display: block; max-width: none; }}
    tbody td.mobile-note:before {{ content: none; }}
    .scroll-top {{ bottom: 16px; right: 16px; width: 42px; height: 42px; }}
}}
</style>
</head>
<body>
<div class="container">
<h1>Dip and Rally Engine</h1>
<div class="meta">
    Generated <span class="meta-strong">{now}</span>
    <span class="meta-sep">·</span>
    <span class="meta-strong">{len(decisions)} tickers</span>
    <span class="meta-sep">·</span>
    Run cost <span class="meta-strong">${spent:.2f}</span> ·
    ~<span class="meta-strong">${annual_estimate:.0f}</span> p.a. if run every trading day
</div>
<div class="spot-source">{html.escape(spot_source)}</div>
<div class="summary-row">
    <div class="summary-tile"><strong>{n_buy}</strong>BUY</div>
    <div class="summary-tile"><strong>{n_wait}</strong>WAIT</div>
    <div class="summary-tile"><strong>{n_refused}</strong>REFUSED</div>
    <div class="summary-tile"><strong>{n_fail}</strong>FAIL</div>
    <div class="legend-wrapper" id="legendWrapper">
        <button class="legend-toggle" id="legendToggle" aria-expanded="false">
            <span class="legend-toggle-arrow">▶</span>
            Legend — verdicts &amp; column meanings
        </button>
        <div class="legend-body" id="legendBody">
            <div class="legend-section">
                <span class="legend-section-title">Verdicts</span>
                <div class="legend-verdict-row">
                    <span class="vchip" style="background:#2ea043">BUY</span>
                    <span class="vdef">The math and AI agree on a positive-expected-return swing setup. Place a limit-buy at the Dip price; the system projects the Rally price will be touched within 60 trading days for a profit. Size externally per your own risk tolerance.</span>
                </div>
                <div class="legend-verdict-row">
                    <span class="vchip" style="background:#c2580a">REFUSED-EV</span>
                    <span class="vdef">Even if both dip and rally fill, the expected return after commissions and slippage is too low (or negative). Don't trade — the math says you lose money on average. Wait for a better setup.</span>
                </div>
                <div class="legend-verdict-row">
                    <span class="vchip" style="background:#c2580a">REFUSED-PARABOLA</span>
                    <span class="vdef">The stock is up too much in the last 30 days with no concrete bearish reason to expect mean-reversion. Buying a blow-off move without a thesis is statistically loss-making. Wait for cool-down or for a bearish catalyst (earnings reset, regulatory action, secondary offering).</span>
                </div>
                <div class="legend-verdict-row">
                    <span class="vchip" style="background:#da3633">REFUSED-METHOD</span>
                    <span class="vdef">The three independent math models (Monte Carlo, PDE, closed-form) disagree on this trade. Publishing a recommendation when the engine can't agree with itself would mean publishing a number it can't verify. Wait for inputs to stabilize.</span>
                </div>
                <div class="legend-verdict-row">
                    <span class="vchip" style="background:#8957e5">⚠ CORRELATED note</span>
                    <span class="vdef">A BUY that tracks another already-accepted BUY closely (correlation ≥ 0.75 over last 90 days). The engine surfaces the correlation as a flag in the status note but does NOT silence the signal — for swing trading, correlated dip-and-rally events are independent opportunities, not "one bet doubled." Operator decides whether to take both, one, or scale.</span>
                </div>
                <div class="legend-verdict-row">
                    <span class="vchip" style="background:#6e7681">WAIT</span>
                    <span class="vdef">At the current spot, no defensible dip-and-rally pair could be found that meets the conviction thresholds. The math couldn't surface a tradeable setup today. Re-check after the next market close.</span>
                </div>
                <div class="legend-verdict-row">
                    <span class="vchip" style="background:#da3633">FAIL</span>
                    <span class="vdef">Data fetch failed for this ticker — probably a temporary provider issue. Investigate the per-ticker log; usually resolved on the next cycle.</span>
                </div>
            </div>
            <div class="legend-section">
                <span class="legend-section-title">Columns</span>
                <div class="legend-col-row"><strong>σ-class</strong> — volatility bucket (MID under 60%, HIGH 60-100%, EXTREME above 100% annualized std dev of returns).</div>
                <div class="legend-col-row"><strong>Tier</strong> — AI compute level: T0 = math-only, T1 = + Pass 1 (Haiku quick scan), T2 = + Pass 2 (Sonnet adversarial critique), T3 = + stress test.</div>
                <div class="legend-col-row"><strong>Ambiguity</strong> — math layer uncertainty score (0-1). Lower = engine is confident. Higher = AI critique has more leverage.</div>
                <div class="legend-col-row"><strong>P(RT)</strong> — joint probability of round-trip success: both Dip AND Rally prices hit within 60 trading days. Computed from 100,000 Monte Carlo simulations.</div>
                <div class="legend-col-row"><strong>EV bps (%)</strong> — expected return per share after friction, in basis points of dip price. 1 bp = 0.01%; +50 bps = +0.50%. Positive = profitable on average; negative = loses money on average.</div>
            </div>
        </div>
    </div>
</div>
<div class="table-container">
<div id="ticker-nav" class="ticker-nav">
  <span class="ticker-nav-label">Jump to:</span>
  {ticker_nav_html}
</div>

<table id="universe">
    <thead>
        <tr>
            <th>Ticker</th>
            <th>σ-class</th>
            <th>Tier</th>
            <th>Ambiguity</th>
            <th>Verdict</th>
            <th class="note-col">Reason / status</th>
        </tr>
    </thead>
    <tbody>
{chr(10).join(rows_html)}
    </tbody>
</table>
</div>
<footer>
    Click ticker → opens Trading212 instrument page in new tab ·
    Click Dip price → opens per-ticker engine dashboard ·
    Hover dotted-underlined values for explanations ·
    Click column headers to sort
</footer>
</div>
<button class="scroll-top" id="scrollTopBtn" aria-label="Scroll to top">↑</button>
<script>
// Legend smooth toggle
(function() {{
    const wrapper = document.getElementById('legendWrapper');
    const toggle = document.getElementById('legendToggle');
    if (!wrapper || !toggle) return;
    toggle.addEventListener('click', function() {{
        const isOpen = wrapper.classList.toggle('open');
        toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    }});
    document.addEventListener('click', function(e) {{
        if (!wrapper.contains(e.target) && wrapper.classList.contains('open')) {{
            wrapper.classList.remove('open');
            toggle.setAttribute('aria-expanded', 'false');
        }}
    }});
}})();
// Edge-aware tooltip positioning
(function() {{
    document.querySelectorAll('.tt').forEach(function(tt) {{
        const tip = tt.querySelector('.tt-content');
        if (!tip) return;
        function position() {{
            const anchorRect = tt.getBoundingClientRect();
            tip.style.visibility = 'hidden';
            tip.style.opacity = '1';
            tip.style.left = '0';
            tip.style.top = '0';
            const tipRect = tip.getBoundingClientRect();
            const vw = window.innerWidth;
            const margin = 8;
            let top = anchorRect.top - tipRect.height - 8;
            if (top < margin) top = anchorRect.bottom + 8;
            let left = anchorRect.left + (anchorRect.width / 2) - (tipRect.width / 2);
            if (left + tipRect.width > vw - margin) left = vw - tipRect.width - margin;
            if (left < margin) left = margin;
            tip.style.left = left + 'px';
            tip.style.top = top + 'px';
            tip.style.visibility = '';
        }}
        tt.addEventListener('mouseenter', position);
        tt.addEventListener('focus', position);
        window.addEventListener('scroll', function() {{
            if (tt.matches(':hover')) position();
        }}, {{passive: true}});
        window.addEventListener('resize', function() {{
            if (tt.matches(':hover')) position();
        }});
    }});
}})();
// Scroll-to-top button
(function() {{
    const btn = document.getElementById('scrollTopBtn');
    if (!btn) return;
    let hideTimeout;
    function onScroll() {{
        if (window.scrollY > 300) {{
            btn.classList.add('visible');
            clearTimeout(hideTimeout);
            hideTimeout = setTimeout(function() {{
                btn.classList.remove('visible');
            }}, 1500);
        }} else {{
            btn.classList.remove('visible');
            clearTimeout(hideTimeout);
        }}
    }}
    window.addEventListener('scroll', onScroll, {{passive: true}});
    btn.addEventListener('click', function() {{
        window.scrollTo({{top: 0, behavior: 'smooth'}});
    }});
    btn.addEventListener('mouseenter', function() {{ clearTimeout(hideTimeout); }});
    btn.addEventListener('mouseleave', function() {{
        if (window.scrollY > 300) {{
            hideTimeout = setTimeout(function() {{
                btn.classList.remove('visible');
            }}, 1500);
        }}
    }});
}})();
// Click-to-sort headers (preserve existing behavior)
document.querySelectorAll('#universe thead th').forEach((th, idx) => {{
    let asc = true;
    th.addEventListener('click', () => {{
        const tbody = th.closest('table').querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const num_re = /^[+-]?\\$?[\\d,.]+%?$/;
        rows.sort((a, b) => {{
            const av = a.children[idx].textContent.trim();
            const bv = b.children[idx].textContent.trim();
            const a_num = num_re.test(av) ? parseFloat(av.replace(/[$,%]/g,'')) : NaN;
            const b_num = num_re.test(bv) ? parseFloat(bv.replace(/[$,%]/g,'')) : NaN;
            let cmp;
            if (!isNaN(a_num) && !isNaN(b_num)) cmp = a_num - b_num;
            else cmp = av.localeCompare(bv);
            return asc ? cmp : -cmp;
        }});
        asc = !asc;
        rows.forEach(r => tbody.appendChild(r));
    }});
}});
</script>
</body>
</html>
"""


def generate_aggregate_dashboard(results: list[TickerRun],
                                  allocation,
                                  run_dir: Path) -> Path:
    """Generate the index.html cross-ticker ranking page in two places:

      output/<run_dir>/index.html  — audit artifact (per-ticker links
                                      use ../ to reach output/)
      output/index.html             — stable bookmark target (per-ticker
                                      links are siblings)

    PR #55 wires the W8 portfolio correlation gate (PR #49 module): any
    BUY-verdict ticker whose 60d return correlation against an
    already-accepted higher-EV ticker exceeds the threshold gets
    re-tagged as REFUSED-CORRELATED so the trader sees one
    representative per cluster instead of substitute ideas.

    Returns the path of the run-dir copy (operator-bookmarkable URL is
    the stable one).
    """
    decisions = [_decision_from_run(r) for r in results]
    # Rank: ambiguity desc within rank groups; FAILs at bottom.
    decisions.sort(key=lambda d: (
        d.verdict == "FAIL",                  # FAILs last
        -(d.ambiguity or -1.0),               # higher ambiguity first
        d.ticker,
    ))

    # PR #55: portfolio correlation gate applied to BUY-verdict tickers
    # only. WAIT / REFUSED / FAIL pass through untouched (no
    # recommendation to dedupe). Dropped BUYs get re-tagged with the
    # gate's reason in status_note + verdict flipped to REFUSED-
    # CORRELATED so the dashboard shows it clearly.
    #
    # 2026-05-24 audit fix #2: always log the gate's outcome so the
    # operator can verify it ran. Previously the gate was silent on
    # the success path, indistinguishable from "didn't run".
    try:
        from src.portfolio import (
            PortfolioRecommendation,
            gate_by_correlation,
        )
        buy_decisions = [d for d in decisions if d.verdict == "BUY"]
        if len(buy_decisions) < 2:
            print(f"   Portfolio gate: skipped — only {len(buy_decisions)} "
                  f"BUY(s) (need ≥2 to evaluate correlation)")
        else:
            recs = []
            skipped_for_history = []
            for d in buy_decisions:
                hist_df = _history_as_price_df(d.ticker)
                if hist_df is None:
                    skipped_for_history.append(d.ticker)
                    continue
                recs.append(PortfolioRecommendation(
                    ticker=d.ticker,
                    ev_bps=d.ev_bps_of_dip if d.ev_bps_of_dip is not None else 0.0,
                    history_df=hist_df,
                ))
            # PR #77: surface the gate-excluded tickers on the dashboard
            # too. These BUYs were dropped at `_history_as_price_df`
            # (no FMP data / <30 bars / fetch failure) — gate never saw
            # them. Previously the skip was only printed; the trader saw
            # a BUY with no LIMITED-HISTORY hint.
            for d in decisions:
                if d.ticker in skipped_for_history:
                    existing = d.status_note or ""
                    sep = " · " if existing else ""
                    d.status_note = (
                        f"{existing}{sep}⚠ LIMITED-HISTORY: bypassed "
                        f"correlation gate (no usable price history)"
                    )
            if len(recs) < 2:
                print(f"   Portfolio gate: skipped — only {len(recs)} "
                      f"BUY(s) have usable price history "
                      f"(skipped: {', '.join(skipped_for_history) or 'none'})")
            else:
                gate = gate_by_correlation(recs)
                # PR #74: gate is now INFORMATIONAL, not exclusionary.
                # Sacred #6 (operator sizes externally) — engine surfaces
                # signals, operator decides whether to take correlated
                # bets or pick one. For a swing trader catching independent
                # dip-and-rally EVENTS (not building a long-term portfolio),
                # correlated BUYs are not "one bet doubled" — they're
                # multiple independent timing opportunities. Annotate the
                # correlation as a flag on the BUY; don't silence it.
                for d in decisions:
                    if d.ticker in gate.dropped:
                        reason = gate.dropped[d.ticker]
                        # Append to existing BUY status_note instead of
                        # overwriting (preserves the Dip → Rally summary)
                        existing = d.status_note or ""
                        sep = " · " if existing else ""
                        d.status_note = f"{existing}{sep}⚠ CORRELATED: {reason}"
                # PR #77: also surface gate.bypassed — tickers passed to
                # the gate but with insufficient history to actually
                # evaluate correlation (passed _history_as_price_df's
                # 30-bar floor but < window_days+1).
                for d in decisions:
                    if d.ticker in gate.bypassed:
                        reason = gate.bypassed[d.ticker]
                        existing = d.status_note or ""
                        sep = " · " if existing else ""
                        d.status_note = (
                            f"{existing}{sep}⚠ LIMITED-HISTORY: {reason}"
                        )
                noted = len(gate.dropped)
                print(f"   Portfolio gate (INFORMATIONAL): evaluated {len(recs)} "
                      f"BUY(s), noted {noted} correlation pair(s), "
                      f"{len(gate.bypassed)} bypassed for limited history "
                      f"(threshold ρ ≥ {_gate_threshold_for_log():.2f}). "
                      f"All BUYs remain visible — operator decides.")
                for t, reason in gate.dropped.items():
                    print(f"     • {t}: {reason}")
                for t, reason in gate.bypassed.items():
                    print(f"     ⚠ bypassed {t}: {reason}")
    except Exception as _e:
        # Gate failure must NOT block dashboard generation. Log + skip.
        print(f"   WARNING: portfolio gate skipped: {_e}")

    # Audit copy inside run_dir — one level deep, links use "../".
    run_html = _render_dashboard_html(decisions, allocation, href_prefix="../")
    run_target = run_dir / "index.html"
    run_target.write_text(run_html)

    # Stable bookmarkable copy in output/ — siblings, no prefix.
    stable_html = _render_dashboard_html(decisions, allocation, href_prefix="")
    (_OUTPUT_ROOT / "index.html").write_text(stable_html)
    return run_target


def _gate_threshold_for_log() -> float:
    """Pull the configured threshold for the gate's log line. Defensive —
    if config isn't loadable for some reason (tests, missing YAML),
    return the documented default."""
    try:
        from src.config import PORTFOLIO_GATE
        return float(PORTFOLIO_GATE.correlation_threshold)
    except Exception:
        return 0.85

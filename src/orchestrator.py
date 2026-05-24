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
                progress(f"     FAILED — {r.phase1_error}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
            future_to_idx = {ex.submit(_phase1_single, t, run_dir): i
                             for i, t in enumerate(tickers)}
            for fut in concurrent.futures.as_completed(future_to_idx):
                i = future_to_idx[fut]
                r = fut.result()
                results[i] = r
                tag = (f"ambiguity={r.snapshot.ambiguity:.2f}"
                       if r.snapshot else f"FAILED — {r.phase1_error}")
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
    """Read all rows from output/round_trip_history_<TICKER>.csv and
    return a pandas DataFrame with 'Date' + 'Close' columns. 'Close'
    is the engine-run-time spot price for that prediction date —
    a daily-close proxy that's good enough for 60d-correlation
    computation in the portfolio gate (PR #55 wiring of PR #49).
    Returns None when the file is absent or has < 5 rows (gate
    accepts defensively below the minimum-data threshold)."""
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
    # Verdict logic — mirror reporter.py's headline decision tree.
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
    elif decision.dip_target is None or decision.dip_target == 0.0:
        decision.verdict = "WAIT"
        decision.status_note = "No qualifying pair at current spot"
    elif decision.ev_bps_of_dip is not None and decision.ev_bps_of_dip < 50.0:
        decision.verdict = "REFUSED-EV"
        decision.status_note = (
            f"EV/dip {decision.ev_bps_of_dip:.1f}bps below 50bps hurdle (sacred #13)"
        )
    else:
        decision.verdict = "BUY"
        decision.status_note = (
            f"Dip ${decision.dip_target:.2f} → Rally ${decision.rally_target:.2f}, "
            f"P(round-trip) = {(decision.p_round_trip or 0) * 100:.0f}%"
        )
    return decision


_VERDICT_COLORS = {
    "BUY":                "#1a7f37",     # green
    "WAIT":               "#6e7781",     # neutral gray
    "REFUSED-EV":         "#bc4c00",     # orange
    "REFUSED-TREND":      "#bc4c00",
    "REFUSED-PARABOLA":   "#bc4c00",
    "REFUSED-CORRELATED": "#8250df",     # purple (PR #55 — substitute idea)
    "REFUSED-METHOD":     "#cf222e",     # red
    "FAIL":               "#cf222e",
    "DELISTED":           "#57606a",     # darker gray — operator action ≠ trader action
}


def _render_dashboard_html(decisions: list[TickerDecision], allocation,
                            href_prefix: str = "") -> str:
    """Render the aggregate-dashboard HTML. href_prefix is prepended
    to each ticker's dashboard link — empty when writing to output/
    (siblings); "../" when writing inside a run_dir (one level deeper)."""
    spent = allocation.spent_usd if allocation else 0.0
    cap = allocation.cap_usd if allocation else 0.0
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    rows_html = []
    for d in decisions:
        color = _VERDICT_COLORS.get(d.verdict, "#6e7781")
        ambig_str = f"{d.ambiguity:.2f}" if d.ambiguity is not None else "—"
        ev_str = f"{d.ev_bps_of_dip:+.1f}" if d.ev_bps_of_dip is not None else "—"
        spot_str = f"${d.spot:,.2f}" if d.spot is not None else "—"
        dip_str = f"${d.dip_target:,.2f}" if d.dip_target else "—"
        rally_str = f"${d.rally_target:,.2f}" if d.rally_target else "—"
        prt_str = f"{d.p_round_trip*100:.0f}%" if d.p_round_trip is not None else "—"
        href = f"{href_prefix}{d.ticker.lower()}_dipnrally_dashboard.html"
        ticker_cell = (
            f'<a href="{html.escape(href)}">{html.escape(d.ticker)}</a>'
        )
        rows_html.append(f"""      <tr data-verdict="{d.verdict}">
        <td>{ticker_cell}</td>
        <td>{html.escape(d.sigma_class)}</td>
        <td>{html.escape(d.tier)}</td>
        <td class="num">{ambig_str}</td>
        <td><span class="verdict" style="background:{color}">{d.verdict}</span></td>
        <td class="num">{spot_str}</td>
        <td class="num">{dip_str}</td>
        <td class="num">{rally_str}</td>
        <td class="num">{prt_str}</td>
        <td class="num">{ev_str}</td>
        <td class="note">{html.escape(d.status_note)}</td>
      </tr>""")

    n_buy = sum(1 for d in decisions if d.verdict == "BUY")
    n_wait = sum(1 for d in decisions if d.verdict == "WAIT")
    n_refused = sum(1 for d in decisions if d.verdict.startswith("REFUSED"))
    n_fail = sum(1 for d in decisions if d.verdict == "FAIL")
    n_delisted = sum(1 for d in decisions if d.verdict == "DELISTED")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>diprally-engine — universe ranking ({now})</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
          margin: 24px; color: #1f2328; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .meta {{ color: #6e7781; font-size: 13px; margin-bottom: 16px; }}
  .summary {{ display: flex; gap: 24px; margin: 12px 0 20px; flex-wrap: wrap; }}
  .summary div {{ background: #f6f8fa; padding: 8px 14px; border-radius: 6px;
                  font-size: 13px; }}
  .summary strong {{ font-size: 18px; display: block; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  thead th {{ text-align: left; background: #f6f8fa; padding: 8px 10px;
              border-bottom: 1px solid #d0d7de; cursor: pointer;
              user-select: none; }}
  thead th:hover {{ background: #eaeef2; }}
  tbody td {{ padding: 6px 10px; border-bottom: 1px solid #eaeef2;
              vertical-align: middle; }}
  tbody tr:hover {{ background: #f6f8fa; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.note {{ color: #6e7781; font-size: 12px; max-width: 350px; }}
  .verdict {{ display: inline-block; color: #fff; padding: 2px 8px;
              border-radius: 10px; font-size: 11px; font-weight: 600; }}
  a {{ color: #0969da; text-decoration: none; font-weight: 600; }}
  a:hover {{ text-decoration: underline; }}
  footer {{ margin-top: 24px; color: #6e7781; font-size: 12px; }}
</style>
</head>
<body>
  <h1>diprally-engine — universe ranking</h1>
  <div class="meta">
    Generated {now} ·
    {len(decisions)} tickers ·
    broker spend ${spent:.2f} of ${cap:.2f} cap ({(spent/cap*100) if cap else 0:.0f}%)
  </div>
  <div class="summary">
    <div><strong>{n_buy}</strong>BUY</div>
    <div><strong>{n_wait}</strong>WAIT</div>
    <div><strong>{n_refused}</strong>REFUSED</div>
    <div><strong>{n_fail}</strong>FAIL</div>
    <div><strong>{n_delisted}</strong>DELISTED</div>
  </div>
  <table id="universe">
    <thead>
      <tr>
        <th>Ticker</th>
        <th>σ-class</th>
        <th>Tier</th>
        <th>Ambiguity</th>
        <th>Verdict</th>
        <th>Spot</th>
        <th>Dip</th>
        <th>Rally</th>
        <th>P(RT)</th>
        <th>EV bps</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(rows_html)}
    </tbody>
  </table>
  <footer>
    Click column headers to sort. Tickers link to per-ticker dashboards.
    Subprocess logs per ticker in this run's directory.
  </footer>
<script>
  // Click-to-sort header behavior. Numeric vs. text inferred from
  // first non-empty cell in the column.
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
    try:
        from src.portfolio import (
            PortfolioRecommendation,
            gate_by_correlation,
        )
        buy_decisions = [d for d in decisions if d.verdict == "BUY"]
        if len(buy_decisions) >= 2:
            recs = []
            for d in buy_decisions:
                hist_df = _history_as_price_df(d.ticker)
                if hist_df is None:
                    continue
                recs.append(PortfolioRecommendation(
                    ticker=d.ticker,
                    ev_bps=d.ev_bps_of_dip if d.ev_bps_of_dip is not None else 0.0,
                    history_df=hist_df,
                ))
            if len(recs) >= 2:
                gate = gate_by_correlation(recs)
                for d in decisions:
                    if d.ticker in gate.dropped:
                        d.verdict = "REFUSED-CORRELATED"
                        d.status_note = gate.dropped[d.ticker]
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

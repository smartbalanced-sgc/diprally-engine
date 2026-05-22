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
inside the run dir.
"""
from __future__ import annotations

import concurrent.futures
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
    can fill phase-2 fields after phase-1 + broker decision."""
    ticker: str
    phase1_returncode: Optional[int] = None
    snapshot: Optional[BrokerSnapshot] = None
    phase1_error: Optional[str] = None
    assigned_tier: str = "T0"
    phase2_returncode: Optional[int] = None
    phase2_error: Optional[str] = None
    elapsed_seconds: float = 0.0
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
        return result
    result.snapshot = parse_snapshot(stdout)
    if result.snapshot is None:
        result.phase1_error = "BROKER_SNAPSHOT_JSON missing from stdout"
    return result


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

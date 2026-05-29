"""Defect F — same-day re-run billing integrity.

The engine never double-charges: a cache hit makes no new API call ($0.00
incremental). The actual defect was a LEDGER ERASURE — a cache-hit re-run
wrote ai_cost_total=0.00, and same-day dedup (sacred #11, append_history_row)
replaced the original real-cost row with it, erasing the day's AI spend from
the canonical per-(ticker,date) ledger.

Fix: replay_costs_from_cache recovers the original run's costs from the cache
payload (which already stores them) so the day's recorded cost is preserved
across re-runs. Incremental cost stays $0.00; the ledger stays honest.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.engine import replay_costs_from_cache, append_history_row, CSV_COLUMNS


# ---------------------------------------------------------------------------
# replay_costs_from_cache — recover the day's incurred cost from a payload
# ---------------------------------------------------------------------------

def test_recovers_all_four_cost_components():
    payload = {
        "pass1_cost": 0.20, "pass2_cost": 0.08,
        "verification_cost": 0.02, "stress_cost": 0.01,
    }
    p1, p2, ver, st = replay_costs_from_cache(payload)
    assert (p1, p2, ver, st) == (0.20, 0.08, 0.02, 0.01)
    assert p1 + p2 + ver + st == pytest.approx(0.31)


def test_legacy_payload_missing_fields_defaults_zero():
    """A pre-Defect-F cache payload may lack some cost fields → 0.0, no crash."""
    p1, p2, ver, st = replay_costs_from_cache({"pass1_cost": 0.20})
    assert (p1, p2, ver, st) == (0.20, 0.0, 0.0, 0.0)


def test_none_values_coerced_to_zero():
    payload = {"pass1_cost": None, "pass2_cost": 0.05,
               "verification_cost": None, "stress_cost": None}
    assert replay_costs_from_cache(payload) == (0.0, 0.05, 0.0, 0.0)


def test_non_dict_payload_safe():
    assert replay_costs_from_cache(None) == (0.0, 0.0, 0.0, 0.0)
    assert replay_costs_from_cache("garbage") == (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Ledger preservation: dedup must not erase the day's spend on re-run
# ---------------------------------------------------------------------------

def _row(date, cost):
    r = {c: "" for c in CSV_COLUMNS}
    r["date"] = date
    r["ai_cost_total"] = f"{cost:.2f}"
    return r


def _read_cost(path, date):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["date"] == date]
    assert len(rows) == 1, f"expected one canonical row per date, got {len(rows)}"
    return float(rows[0]["ai_cost_total"])


def test_replay_row_preserves_day_cost_not_zero(tmp_path):
    """Original fresh run records $0.30. A same-day cache-hit re-run, using
    the recovered cost, writes $0.30 again — dedup keeps one row at $0.30.
    (The pre-fix bug wrote $0.00, erasing the spend.)"""
    hist = tmp_path / "AMAT_history.csv"
    payload = {"pass1_cost": 0.20, "pass2_cost": 0.08,
               "verification_cost": 0.02, "stress_cost": 0.0}
    day_cost = sum(replay_costs_from_cache(payload))

    append_history_row(hist, _row("2026-05-29", 0.30))      # fresh run
    append_history_row(hist, _row("2026-05-29", day_cost))  # cache-hit re-run

    assert _read_cost(hist, "2026-05-29") == pytest.approx(0.30)


def test_pre_fix_behavior_would_have_erased_cost(tmp_path):
    """Documents the bug: a $0.00 replay row overwrites the real cost."""
    hist = tmp_path / "AMAT_history.csv"
    append_history_row(hist, _row("2026-05-29", 0.30))
    append_history_row(hist, _row("2026-05-29", 0.0))  # buggy replay
    assert _read_cost(hist, "2026-05-29") == pytest.approx(0.0)  # erased

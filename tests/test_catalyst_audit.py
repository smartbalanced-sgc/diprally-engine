"""Tests for tools/catalyst_audit.py — D-W10-1 catalyst-occurrence
triage tool.

The tool surfaces catalysts whose predicted date has elapsed but
which the operator hasn't yet verified in the ledger. It is the
data-foundation step for D-W10-1's per-ticker hallucination-rate
analysis (which builds later, once N≥30 days of operator verdicts
have accumulated).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# tools/ is a CLI dir, not a package — import by path.
_spec = importlib.util.spec_from_file_location(
    "catalyst_audit", _REPO_ROOT / "tools" / "catalyst_audit.py"
)
catalyst_audit = importlib.util.module_from_spec(_spec)
sys.modules["catalyst_audit"] = catalyst_audit
_spec.loader.exec_module(catalyst_audit)


def _write_csv(tmp_path, ticker, rows):
    """Write a minimal round_trip_history CSV. rows is a list of
    (date_str, catalysts_list_or_None) — catalysts serialized as JSON
    into pass2_catalysts_json."""
    from src.engine import CSV_COLUMNS
    path = tmp_path / f"round_trip_history_{ticker}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for date_str, catalysts in rows:
            row = {c: "" for c in CSV_COLUMNS}
            row["date"] = date_str
            if catalysts is not None:
                row["pass2_catalysts_json"] = json.dumps(catalysts)
            w.writerow(row)


# =============================================================================
# _extract_latest_date — freeform date parsing
# =============================================================================

def test_extract_iso_date():
    d = catalyst_audit._extract_latest_date("2026-05-08")
    assert d == datetime(2026, 5, 8)


def test_extract_iso_date_range_returns_latest():
    """A range like 2026-04-01/2026-06-30 should return the END date
    so we don't flag a catalyst overdue before its window closes."""
    d = catalyst_audit._extract_latest_date("2026-04-01/2026-06-30")
    assert d == datetime(2026, 6, 30)


def test_extract_natural_date():
    d = catalyst_audit._extract_latest_date("May 8, 2026")
    assert d == datetime(2026, 5, 8)


def test_extract_quarter_returns_quarter_end():
    """Q2 2026 → end of Q2 = 2026-06-30."""
    d = catalyst_audit._extract_latest_date("Q2 2026")
    assert d == datetime(2026, 6, 30)


def test_extract_unparseable_returns_none():
    assert catalyst_audit._extract_latest_date("ongoing") is None
    assert catalyst_audit._extract_latest_date("") is None
    assert catalyst_audit._extract_latest_date(None) is None


# =============================================================================
# _read_catalyst_history — CSV-aware dedupe
# =============================================================================

def test_no_csv_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(catalyst_audit, "_OUTPUT_ROOT", tmp_path)
    assert catalyst_audit._read_catalyst_history("NOEXIST", 30) == []


def test_dedupes_same_catalyst_across_rows(tmp_path, monkeypatch):
    """If the same catalyst appears in multiple CSV rows (engine ran
    multiple days), the tool returns it once with the EARLIEST
    first_seen_date."""
    monkeypatch.setattr(catalyst_audit, "_OUTPUT_ROOT", tmp_path)
    cat = [{"name": "Q2 earnings beat", "type": "earnings",
            "date_or_window": "2026-05-08", "direction_risk": "bullish",
            "magnitude": "high"}]
    _write_csv(tmp_path, "INTC", [
        ("2026-05-01", cat),
        ("2026-05-02", cat),
        ("2026-05-03", cat),
    ])
    catalysts = catalyst_audit._read_catalyst_history("INTC", since_days=365)
    assert len(catalysts) == 1
    assert catalysts[0]["first_seen_date"] == "2026-05-01"


def test_since_days_filter_respected(tmp_path, monkeypatch):
    """Catalysts surfaced in CSV rows OLDER than since_days are skipped."""
    monkeypatch.setattr(catalyst_audit, "_OUTPUT_ROOT", tmp_path)
    today = datetime.today()
    old_date = today.replace(year=today.year - 1).strftime("%Y-%m-%d")
    new_date = today.strftime("%Y-%m-%d")
    _write_csv(tmp_path, "INTC", [
        (old_date, [{"name": "Old catalyst", "date_or_window": "2025-01-01"}]),
        (new_date, [{"name": "New catalyst", "date_or_window": "2026-05-01"}]),
    ])
    catalysts = catalyst_audit._read_catalyst_history("INTC", since_days=30)
    names = [c["catalyst_name"] for c in catalysts]
    assert "Old catalyst" not in names
    assert "New catalyst" in names


def test_falls_back_to_pass1_when_pass2_absent(tmp_path, monkeypatch):
    """T0/T1 runs have no pass2_catalysts_json but may have pass1_*.
    Tool falls back to Pass 1 list when Pass 2 absent."""
    from src.engine import CSV_COLUMNS
    monkeypatch.setattr(catalyst_audit, "_OUTPUT_ROOT", tmp_path)
    path = tmp_path / "round_trip_history_INTC.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        row = {c: "" for c in CSV_COLUMNS}
        row["date"] = datetime.today().strftime("%Y-%m-%d")
        row["pass1_catalysts_json"] = json.dumps([
            {"name": "Pass-1-only catalyst", "date_or_window": "2026-05-08"}
        ])
        # NO pass2_catalysts_json
        w.writerow(row)
    catalysts = catalyst_audit._read_catalyst_history("INTC", since_days=365)
    assert len(catalysts) == 1
    assert catalysts[0]["catalyst_name"] == "Pass-1-only catalyst"


def test_malformed_json_skipped_gracefully(tmp_path, monkeypatch):
    from src.engine import CSV_COLUMNS
    monkeypatch.setattr(catalyst_audit, "_OUTPUT_ROOT", tmp_path)
    path = tmp_path / "round_trip_history_INTC.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        row = {c: "" for c in CSV_COLUMNS}
        row["date"] = datetime.today().strftime("%Y-%m-%d")
        row["pass2_catalysts_json"] = "{not valid json"
        w.writerow(row)
    catalysts = catalyst_audit._read_catalyst_history("INTC", since_days=365)
    assert catalysts == []  # malformed → skipped, no crash


# =============================================================================
# find_pending_reviews
# =============================================================================

def test_future_date_catalyst_not_surfaced(tmp_path, monkeypatch):
    """A catalyst whose predicted date is still in the future is NOT
    pending review — only elapsed catalysts surface."""
    monkeypatch.setattr(catalyst_audit, "_OUTPUT_ROOT", tmp_path)
    today = datetime.today()
    future = today.replace(year=today.year + 1).strftime("%Y-%m-%d")
    _write_csv(tmp_path, "INTC", [
        (today.strftime("%Y-%m-%d"),
         [{"name": "Future catalyst", "date_or_window": future}]),
    ])
    pending = catalyst_audit.find_pending_reviews(
        ["INTC"], since_days=30, ledger={}, today=today
    )
    assert pending == []


def test_elapsed_catalyst_surfaced(tmp_path, monkeypatch):
    """A catalyst whose date is past should be in pending."""
    monkeypatch.setattr(catalyst_audit, "_OUTPUT_ROOT", tmp_path)
    today = datetime(2026, 6, 1)
    _write_csv(tmp_path, "INTC", [
        ("2026-05-01",
         [{"name": "Q1 earnings", "date_or_window": "2026-05-08",
           "direction_risk": "bullish", "magnitude": "high",
           "type": "earnings"}]),
    ])
    pending = catalyst_audit.find_pending_reviews(
        ["INTC"], since_days=365, ledger={}, today=today
    )
    assert len(pending) == 1
    assert pending[0]["catalyst_name"] == "Q1 earnings"
    assert pending[0]["days_since_due"] == 24


def test_ledger_filters_out_reviewed(tmp_path, monkeypatch):
    """A catalyst already in the ledger does NOT appear in pending."""
    monkeypatch.setattr(catalyst_audit, "_OUTPUT_ROOT", tmp_path)
    today = datetime(2026, 6, 1)
    _write_csv(tmp_path, "INTC", [
        ("2026-05-01",
         [{"name": "Q1 earnings", "date_or_window": "2026-05-08"}]),
    ])
    ledger = {("INTC", "Q1 earnings"): {"verdict": "OCCURRED"}}
    pending = catalyst_audit.find_pending_reviews(
        ["INTC"], since_days=365, ledger=ledger, today=today
    )
    assert pending == []


def test_pending_sorted_by_overdue_desc(tmp_path, monkeypatch):
    """Most-overdue catalyst surfaces first."""
    monkeypatch.setattr(catalyst_audit, "_OUTPUT_ROOT", tmp_path)
    today = datetime(2026, 6, 1)
    _write_csv(tmp_path, "A", [
        ("2026-04-01",
         [{"name": "Cat A (60d overdue)", "date_or_window": "2026-04-01"}]),
    ])
    _write_csv(tmp_path, "B", [
        ("2026-05-25",
         [{"name": "Cat B (7d overdue)", "date_or_window": "2026-05-25"}]),
    ])
    pending = catalyst_audit.find_pending_reviews(
        ["A", "B"], since_days=365, ledger={}, today=today
    )
    assert pending[0]["catalyst_name"].startswith("Cat A")  # most overdue first
    assert pending[1]["catalyst_name"].startswith("Cat B")


def test_unparseable_date_surfaces_at_bottom(tmp_path, monkeypatch):
    """A catalyst with unparseable date_or_window ('ongoing') still
    appears in pending but at the bottom of the sort."""
    monkeypatch.setattr(catalyst_audit, "_OUTPUT_ROOT", tmp_path)
    today = datetime(2026, 6, 1)
    _write_csv(tmp_path, "A", [
        ("2026-04-01",
         [{"name": "Dated", "date_or_window": "2026-04-01"}]),
    ])
    _write_csv(tmp_path, "B", [
        ("2026-04-01",
         [{"name": "Undated", "date_or_window": "ongoing"}]),
    ])
    pending = catalyst_audit.find_pending_reviews(
        ["A", "B"], since_days=365, ledger={}, today=today
    )
    assert pending[-1]["catalyst_name"] == "Undated"
    assert pending[-1]["due_date"] == "(unparsed)"


# =============================================================================
# Ledger I/O
# =============================================================================

def test_ledger_creation_when_missing(tmp_path):
    ledger_path = tmp_path / "ledger.csv"
    catalyst_audit._ensure_ledger_exists(ledger_path)
    assert ledger_path.exists()
    # Empty header-only
    with open(ledger_path) as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def test_load_ledger_keyed_by_ticker_and_name(tmp_path):
    ledger_path = tmp_path / "ledger.csv"
    with open(ledger_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=catalyst_audit.LEDGER_COLUMNS)
        w.writeheader()
        w.writerow({
            "audit_date": "2026-06-01", "ticker": "INTC",
            "catalyst_name": "Q1 earnings", "catalyst_type": "earnings",
            "predicted_date_window": "2026-05-08",
            "predicted_direction": "bullish", "predicted_magnitude": "high",
            "first_seen_date": "2026-05-01", "verdict": "OCCURRED",
            "reason": "Reported May 8 in line with consensus", "source_url": "",
        })
    ledger = catalyst_audit._load_ledger(ledger_path)
    assert ("INTC", "Q1 earnings") in ledger
    assert ledger[("INTC", "Q1 earnings")]["verdict"] == "OCCURRED"


# =============================================================================
# format_pending_table
# =============================================================================

def test_format_empty_message():
    output = catalyst_audit.format_pending_table([])
    assert "no pending reviews" in output


def test_format_includes_ticker_and_overdue():
    pending = [{
        "ticker": "INTC", "catalyst_name": "Q1 earnings",
        "catalyst_type": "earnings", "predicted_date_window": "2026-05-08",
        "predicted_direction": "bullish", "predicted_magnitude": "high",
        "first_seen_date": "2026-05-01", "due_date": "2026-05-08",
        "days_since_due": 24,
    }]
    output = catalyst_audit.format_pending_table(pending)
    assert "INTC" in output
    assert "Q1 earnings" in output
    assert "24d" in output
    assert "OCCURRED" in output  # ledger verdict instructions

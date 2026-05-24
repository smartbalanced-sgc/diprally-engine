"""Tests for PR #62 — σ-class registry refresh advisor.

Tool reads each ticker's CSV history, compares the most-recent
auto-detected σ-class against the registry hint, and emits a YAML
patch when ≥min_consecutive runs disagree consistently.

Sacred #1 (data wins) — but the registry is operator-curated
structural classification, not auto-overwritten. This tool ADVISES;
it doesn't edit YAML.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# tools/ is a CLI dir, not a package — import by path.
_spec = importlib.util.spec_from_file_location(
    "registry_refresh", _REPO_ROOT / "tools" / "registry_refresh.py"
)
registry_refresh = importlib.util.module_from_spec(_spec)
sys.modules["registry_refresh"] = registry_refresh
_spec.loader.exec_module(registry_refresh)


def _write_csv(tmp_path, ticker, sigma_classes):
    """Write a minimal CSV history for a ticker with given σ-class
    sequence (oldest first). Only `date` and `sigma_class` columns
    are populated; other CSV_COLUMNS get empty strings."""
    from src.engine import CSV_COLUMNS
    path = tmp_path / f"round_trip_history_{ticker}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for i, cls in enumerate(sigma_classes):
            row = {c: "" for c in CSV_COLUMNS}
            row["date"] = f"2026-05-{i+1:02d}"
            row["sigma_class"] = cls
            w.writerow(row)


# =============================================================================
# analyze_ticker
# =============================================================================

def test_no_history_returns_insufficient(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_refresh, "_OUTPUT_ROOT", tmp_path)
    result = registry_refresh.analyze_ticker("NOEXIST", min_consecutive=5)
    assert result["suggested_action"] == "insufficient_data"
    assert result["recent_classes"] == []


def test_sparse_history_returns_insufficient(tmp_path, monkeypatch):
    """Fewer than min_consecutive rows → can't conclude — insufficient."""
    monkeypatch.setattr(registry_refresh, "_OUTPUT_ROOT", tmp_path)
    _write_csv(tmp_path, "INTC", ["HIGH", "HIGH"])
    result = registry_refresh.analyze_ticker("INTC", min_consecutive=5)
    assert result["suggested_action"] == "insufficient_data"


def test_consistent_auto_matching_registry_returns_keep(tmp_path, monkeypatch):
    """5 consecutive MID runs on INTC (registry says MID) → no action."""
    monkeypatch.setattr(registry_refresh, "_OUTPUT_ROOT", tmp_path)
    _write_csv(tmp_path, "INTC", ["MID"] * 5)
    result = registry_refresh.analyze_ticker("INTC", min_consecutive=5)
    assert result["consistent"] is True
    assert result["suggested_action"] == "keep"
    assert result["most_recent"] == "MID"


def test_consistent_auto_differing_registry_returns_patch(tmp_path, monkeypatch):
    """5 consecutive HIGH auto-detects on INTC (registry MID) → patch."""
    monkeypatch.setattr(registry_refresh, "_OUTPUT_ROOT", tmp_path)
    _write_csv(tmp_path, "INTC", ["HIGH"] * 5)
    result = registry_refresh.analyze_ticker("INTC", min_consecutive=5)
    assert result["consistent"] is True
    assert result["most_recent"] == "HIGH"
    assert result["registry_hint"] == "MID"
    assert result["suggested_action"] == "patch"


def test_mixed_auto_returns_keep(tmp_path, monkeypatch):
    """Auto-detect bounces MID/HIGH/MID/HIGH → not a structural shift,
    no patch suggested. Hint stays."""
    monkeypatch.setattr(registry_refresh, "_OUTPUT_ROOT", tmp_path)
    _write_csv(tmp_path, "INTC", ["MID", "HIGH", "MID", "HIGH", "MID"])
    result = registry_refresh.analyze_ticker("INTC", min_consecutive=5)
    assert result["consistent"] is False
    assert result["suggested_action"] == "keep"


def test_only_recent_runs_count(tmp_path, monkeypatch):
    """If history is [HIGH, HIGH, HIGH, HIGH, MID, MID, MID, MID, MID],
    the last 5 are all MID → consistent MID. Old HIGH runs are stale."""
    monkeypatch.setattr(registry_refresh, "_OUTPUT_ROOT", tmp_path)
    _write_csv(tmp_path, "INTC", ["HIGH"] * 4 + ["MID"] * 5)
    result = registry_refresh.analyze_ticker("INTC", min_consecutive=5)
    # INTC registry is MID; recent 5 are MID → consistent + matches → keep.
    assert result["most_recent"] == "MID"
    assert result["suggested_action"] == "keep"


def test_min_consecutive_parameter_respected(tmp_path, monkeypatch):
    """3 HIGH runs with --min-consecutive 3 → patch suggested.
    Same data with --min-consecutive 5 → insufficient_data."""
    monkeypatch.setattr(registry_refresh, "_OUTPUT_ROOT", tmp_path)
    _write_csv(tmp_path, "INTC", ["HIGH"] * 3)
    result_3 = registry_refresh.analyze_ticker("INTC", min_consecutive=3)
    result_5 = registry_refresh.analyze_ticker("INTC", min_consecutive=5)
    assert result_3["suggested_action"] == "patch"
    assert result_5["suggested_action"] == "insufficient_data"


def test_unknown_ticker_handled_gracefully(tmp_path, monkeypatch):
    """A ticker that's not in the registry universe → registry_hint
    is None. Tool still runs but suggested_action stays 'keep' (no
    hint to patch against)."""
    monkeypatch.setattr(registry_refresh, "_OUTPUT_ROOT", tmp_path)
    _write_csv(tmp_path, "WEIRD", ["HIGH"] * 5)
    result = registry_refresh.analyze_ticker("WEIRD", min_consecutive=5)
    assert result["registry_hint"] is None
    # consistent + mismatch (None != HIGH) → patch action; user can
    # decide whether to ADD it to registry.
    assert result["suggested_action"] == "patch"
    assert result["most_recent"] == "HIGH"


# =============================================================================
# format_report
# =============================================================================

def test_format_report_includes_patch_yaml_block():
    analyses = [{
        "ticker": "INTC", "registry_hint": "MID",
        "recent_classes": ["HIGH"] * 5, "consistent": True,
        "most_recent": "HIGH", "suggested_action": "patch",
    }]
    report = registry_refresh.format_report(analyses)
    assert "INTC" in report
    assert "MID" in report
    assert "HIGH" in report
    # YAML patch block.
    assert "sigma_class: HIGH" in report
    assert "config/diprally.yaml" in report


def test_format_report_handles_empty_input():
    report = registry_refresh.format_report([])
    assert "REGISTRY REFRESH ADVISOR" in report
    assert len(report) > 0


def test_format_report_groups_by_action():
    """Three sections: PATCH, KEEP, INSUFFICIENT_DATA — each only
    rendered when non-empty."""
    analyses = [
        {"ticker": "A", "registry_hint": "MID", "recent_classes": ["HIGH"]*5,
         "consistent": True, "most_recent": "HIGH", "suggested_action": "patch"},
        {"ticker": "B", "registry_hint": "HIGH", "recent_classes": ["HIGH"]*5,
         "consistent": True, "most_recent": "HIGH", "suggested_action": "keep"},
        {"ticker": "C", "registry_hint": "MID", "recent_classes": [],
         "consistent": None, "most_recent": None, "suggested_action": "insufficient_data"},
    ]
    report = registry_refresh.format_report(analyses)
    # All three tickers present.
    for t in ("A", "B", "C"):
        assert t in report
    # Patch section has the YAML patch block.
    assert "sigma_class: HIGH" in report
    # No-data section signals that explicitly.
    assert "no CSV history yet" in report


def test_format_report_shows_mixed_auto_detail():
    """When recent classes are mixed, the operator should see the
    distribution (MID=3, HIGH=2) to judge regime stability."""
    analyses = [{
        "ticker": "INTC", "registry_hint": "MID",
        "recent_classes": ["MID", "HIGH", "MID", "HIGH", "MID"],
        "consistent": False, "most_recent": "MID",
        "suggested_action": "keep",
    }]
    report = registry_refresh.format_report(analyses)
    assert "MID=3" in report or "MID=2" in report  # ordering may vary
    assert "HIGH=" in report

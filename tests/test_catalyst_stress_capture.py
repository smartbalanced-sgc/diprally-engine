"""Tests for W10 PR #61 — catalyst-stress capture to CSV.

The AI stress test produces drift-shock predictions per top-3 catalyst
("if Q2 earnings disappoint by 20%, drift shifts -12pp"). Currently
shown in the report but not persisted. PR #61 captures these to CSV
so future W10 analysis can correlate predicted shocks against
realized drawdowns once 30+ resolved predictions accumulate.

Pure data plumbing — zero engine-output behavior change. Same pattern
as PR #47 (outcome capture) and PR #54 (catalyst capture).
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.engine import CSV_COLUMNS, _compact_stress_json


def test_empty_input_returns_empty_string():
    assert _compact_stress_json([]) == ""
    assert _compact_stress_json(None) == ""


def test_keeps_required_fields():
    """The captured payload retains catalyst_name → name and
    drift_shock_pp_on_disappointment → shock_pp."""
    stress = [
        {"catalyst_name": "Q2 earnings",
         "drift_shock_pp_on_disappointment": -12.5,
         "reasoning": "Beat-miss asymmetry typical for cyclicals"},
        {"catalyst_name": "Apple foundry deal",
         "drift_shock_pp_on_disappointment": -18.0,
         "reasoning": "Strategic implications go beyond revenue"},
    ]
    serialized = _compact_stress_json(stress)
    parsed = json.loads(serialized)
    assert len(parsed) == 2
    assert parsed[0]["name"] == "Q2 earnings"
    assert parsed[0]["shock_pp"] == -12.5
    assert parsed[1]["shock_pp"] == -18.0


def test_drops_reasoning_to_keep_csv_bounded():
    """reasoning can be 200+ chars per catalyst; multiplied across
    3 stress entries that explodes CSV row width. Persisted in
    stdout log instead — CSV gets name + numeric shock only."""
    stress = [
        {"catalyst_name": "X", "drift_shock_pp_on_disappointment": -10.0,
         "reasoning": "A very long explanation " * 50},
    ]
    serialized = _compact_stress_json(stress)
    parsed = json.loads(serialized)
    assert "reasoning" not in parsed[0]
    assert len(serialized) < 200


def test_skips_non_dict_entries():
    stress = [
        {"catalyst_name": "real", "drift_shock_pp_on_disappointment": -5.0},
        "not a dict",
        None,
        {"catalyst_name": "also real", "drift_shock_pp_on_disappointment": -8.0},
    ]
    serialized = _compact_stress_json(stress)
    parsed = json.loads(serialized)
    assert len(parsed) == 2
    assert parsed[0]["name"] == "real"
    assert parsed[1]["name"] == "also real"


def test_skips_unparseable_shock_values():
    """Defensive: a missing or garbage shock value shouldn't crash
    the serializer. Entry is skipped with no exception."""
    stress = [
        {"catalyst_name": "good", "drift_shock_pp_on_disappointment": -10.0},
        {"catalyst_name": "missing_shock"},
        {"catalyst_name": "bad_shock", "drift_shock_pp_on_disappointment": "not a number"},
    ]
    serialized = _compact_stress_json(stress)
    parsed = json.loads(serialized)
    # "missing_shock" defaults to 0.0 (graceful); "bad_shock" skipped.
    # Implementation choice: float-conversion failure → skip the
    # whole entry; None → falls back to 0.0 via the `or` guard.
    names = [p["name"] for p in parsed]
    assert "good" in names
    assert "bad_shock" not in names


def test_compact_format_no_spaces():
    """Same byte-saving discipline as PR #54: compact JSON, no spaces."""
    stress = [{"catalyst_name": f"Cat {i}",
                "drift_shock_pp_on_disappointment": -float(i * 5)}
               for i in range(3)]
    serialized = _compact_stress_json(stress)
    assert ", " not in serialized
    assert len(serialized) < 300


def test_csv_columns_contain_new_capture_field():
    assert "catalyst_stress_json" in CSV_COLUMNS


def test_payload_round_trips_through_csv_dictwriter():
    """Special characters in catalyst names (commas, quotes, slashes)
    survive csv.DictWriter's escaping."""
    stress = [
        {"catalyst_name": 'Q2 earnings, with comma and "quotes"',
         "drift_shock_pp_on_disappointment": -12.5},
        {"catalyst_name": "Apple foundry deal / M&A close",
         "drift_shock_pp_on_disappointment": -18.0},
    ]
    serialized = _compact_stress_json(stress)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["catalyst_stress_json"])
    writer.writeheader()
    writer.writerow({"catalyst_stress_json": serialized})
    buf.seek(0)
    reader = csv.DictReader(buf)
    rows = list(reader)
    assert len(rows) == 1
    re_parsed = json.loads(rows[0]["catalyst_stress_json"])
    assert re_parsed[0]["name"] == 'Q2 earnings, with comma and "quotes"'
    assert re_parsed[1]["name"] == "Apple foundry deal / M&A close"


def test_zero_shock_preserved():
    """A drift_shock of exactly 0.0 should still be captured (signals
    'AI ran stress test but found no shock' — different from
    'stress test didn't run')."""
    stress = [
        {"catalyst_name": "neutral", "drift_shock_pp_on_disappointment": 0.0},
    ]
    serialized = _compact_stress_json(stress)
    parsed = json.loads(serialized)
    assert len(parsed) == 1
    assert parsed[0]["shock_pp"] == 0.0

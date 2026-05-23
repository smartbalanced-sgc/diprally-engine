"""Tests for W10 PR #54 — D-W10-1 catalyst-accuracy capture.

The engine writes Pass 1 / Pass 2 catalysts + verification verdicts
to the CSV at run time. Future PR #55 (analysis layer) reads these
back to compute per-ticker hallucination rates by checking whether
each predicted catalyst actually occurred as described.

This PR is observational — pure data capture, zero engine-output
behavior change. Tests verify the serializers produce valid JSON
that round-trips cleanly through csv.DictWriter.
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

from src.engine import (
    CSV_COLUMNS,
    _compact_catalysts_json,
    _compact_verdicts_json,
)


# =============================================================================
# _compact_catalysts_json
# =============================================================================

def test_compact_catalysts_empty_input():
    assert _compact_catalysts_json([]) == ""
    assert _compact_catalysts_json(None) == ""


def test_compact_catalysts_keeps_required_fields():
    catalysts = [
        {"name": "Q2 earnings", "type": "earnings",
         "date_or_window": "2026-07-23", "direction_risk": "two-sided",
         "magnitude": "high", "sources": ["sec.gov", "ir.intel.com"]},
    ]
    serialized = _compact_catalysts_json(catalysts)
    parsed = json.loads(serialized)
    assert len(parsed) == 1
    row = parsed[0]
    assert row["name"] == "Q2 earnings"
    assert row["type"] == "earnings"
    assert row["date_or_window"] == "2026-07-23"
    assert row["direction_risk"] == "two-sided"
    assert row["magnitude"] == "high"
    # Sources should be DROPPED (not needed for accuracy analysis).
    assert "sources" not in row


def test_compact_catalysts_drops_verification_fields():
    """Pass 2 catalysts that survived PR #33 verification have
    verification_verdict / verification_reasoning / verification_url
    appended. These go in the SEPARATE verdicts column — they don't
    belong in the catalysts JSON."""
    catalysts = [
        {"name": "earnings", "type": "earnings",
         "date_or_window": "2026-07-23", "direction_risk": "two-sided",
         "magnitude": "high",
         "verification_verdict": "UNVERIFIED",
         "verification_reasoning": "no SEC filing match",
         "verification_url": None,
         "magnitude_pre_verification": "high"},
    ]
    serialized = _compact_catalysts_json(catalysts)
    parsed = json.loads(serialized)
    assert "verification_verdict" not in parsed[0]
    assert "magnitude_pre_verification" not in parsed[0]
    # Magnitude field reflects the POST-verification value (which is what
    # the signal_from_catalyst_proximity reads).
    assert parsed[0]["magnitude"] == "high"


def test_compact_catalysts_skips_non_dict_entries():
    """Defensive: garbage entries shouldn't crash the serializer."""
    catalysts = [
        {"name": "real", "type": "earnings", "date_or_window": "2026-07-23",
         "direction_risk": "two-sided", "magnitude": "high"},
        "not a dict",
        None,
        42,
        {"name": "another", "type": "M&A", "date_or_window": "2026-Q3",
         "direction_risk": "bearish", "magnitude": "med"},
    ]
    serialized = _compact_catalysts_json(catalysts)
    parsed = json.loads(serialized)
    assert len(parsed) == 2
    assert parsed[0]["name"] == "real"
    assert parsed[1]["name"] == "another"


def test_compact_catalysts_compact_format():
    """Serializer uses compact JSON (no spaces) to keep CSV row width
    bounded. A 5-catalyst payload should fit in well under 1KB."""
    catalysts = [
        {"name": f"Catalyst {i}", "type": "earnings",
         "date_or_window": f"2026-07-{i+1:02d}",
         "direction_risk": "two-sided", "magnitude": "med"}
        for i in range(5)
    ]
    serialized = _compact_catalysts_json(catalysts)
    assert ", " not in serialized  # compact format, no spaces
    assert len(serialized) < 1024  # under 1KB for 5 catalysts


# =============================================================================
# _compact_verdicts_json
# =============================================================================

def test_compact_verdicts_empty():
    assert _compact_verdicts_json([]) == ""
    assert _compact_verdicts_json(None) == ""


def test_compact_verdicts_keeps_required_fields():
    verifications = [
        {"catalyst_name": "earnings", "verdict": "VERIFIED",
         "reasoning": "Q2 8-K filed", "supporting_url": "https://sec.gov/..."},
        {"catalyst_name": "phantom deal", "verdict": "REFUTED",
         "reasoning": "no SEC filing match", "supporting_url": None},
    ]
    serialized = _compact_verdicts_json(verifications)
    parsed = json.loads(serialized)
    assert len(parsed) == 2
    assert parsed[0]["catalyst_name"] == "earnings"
    assert parsed[0]["verdict"] == "VERIFIED"
    assert parsed[0]["reasoning"] == "Q2 8-K filed"
    assert parsed[1]["verdict"] == "REFUTED"


def test_compact_verdicts_skips_non_dict():
    verifications = [
        {"catalyst_name": "real", "verdict": "VERIFIED", "reasoning": "ok"},
        "not a dict",
        None,
    ]
    serialized = _compact_verdicts_json(verifications)
    parsed = json.loads(serialized)
    assert len(parsed) == 1
    assert parsed[0]["catalyst_name"] == "real"


# =============================================================================
# CSV round-trip
# =============================================================================

def test_csv_columns_contain_new_capture_fields():
    """The 3 D-W10-1 capture fields must be in CSV_COLUMNS so
    csv.DictWriter actually writes them."""
    for col in ("pass1_catalysts_json", "pass2_catalysts_json",
                 "verification_verdicts_json"):
        assert col in CSV_COLUMNS, f"D-W10-1 column {col!r} missing from CSV_COLUMNS"


def test_serialized_payload_round_trips_through_csv_dictwriter():
    """The compact JSON strings must survive csv.DictWriter's
    quoting/escaping. Commas and quotes in catalyst names are the
    main risk — verify a representative payload makes it through."""
    catalysts = [
        {"name": 'Q2 earnings, with comma and "quotes"',
         "type": "earnings", "date_or_window": "2026-07-23",
         "direction_risk": "two-sided", "magnitude": "high"},
        {"name": "Apple foundry deal/M&A",
         "type": "M&A", "date_or_window": "2026-Q3/Q4",
         "direction_risk": "bullish", "magnitude": "med"},
    ]
    serialized = _compact_catalysts_json(catalysts)

    # Round-trip via DictWriter → DictReader.
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["pass1_catalysts_json"])
    writer.writeheader()
    writer.writerow({"pass1_catalysts_json": serialized})
    buf.seek(0)
    reader = csv.DictReader(buf)
    rows = list(reader)
    assert len(rows) == 1
    re_parsed = json.loads(rows[0]["pass1_catalysts_json"])
    assert re_parsed[0]["name"] == 'Q2 earnings, with comma and "quotes"'
    assert re_parsed[1]["date_or_window"] == "2026-Q3/Q4"

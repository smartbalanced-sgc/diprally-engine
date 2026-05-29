"""Tests for Defect C — analyst price-target REVISION signal.

signal_from_pt_revision reads the per-analyst price-target-news stream
(FMP /stable/price-target-news) and produces the recency-weighted mean
CHANGE in analyst-implied return: (new_PT - prior_PT)/spot_at_post. This
is the engine's native drift unit (same as signal_from_analyst_targets'
target/spot-1), so there is no scale coefficient. Distinct from
signal_from_revision_momentum (upgrade/downgrade counts) and
signal_from_analyst_targets (implied-return LEVEL). Endpoint schema
verified live 2026-05-29 (MU, HTTP 200).

The prior target is parsed from the title's "from $X"; entries without a
parseable prior (e.g. StreetInsider "PT Raised to $1,150") are not
measurable revisions and are skipped. Direction is intrinsic to
(new - prior), so there is no keyword parsing.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import SIGNAL_PT_REVISION as CFG
from src.signals import signal_from_pt_revision

_TODAY = datetime.date(2026, 5, 29)


def _entry(days_ago, price_target, title, price_when_posted=1000.0):
    d = _TODAY - datetime.timedelta(days=days_ago)
    return {
        "publishedDate": d.strftime("%Y-%m-%dT12:00:00.000Z"),
        "priceTarget": price_target,
        "priceWhenPosted": price_when_posted,
        "title": title,
    }


# ---------------------------------------------------------------------------
# Empty / degenerate / non-revisions
# ---------------------------------------------------------------------------

def test_empty_news_is_none_signal():
    s = signal_from_pt_revision([], today=_TODAY)
    assert s["drift"] is None
    assert s["source_quality"] == "NONE_FOUND"


def test_initiations_and_no_prior_skipped():
    """No 'from $X' in the title → not a measurable revision → skipped.
    Covers initiations, reiterations, and StreetInsider-style 'to $X' only."""
    news = [
        _entry(2, 1200, "MU initiated with Buy at Citi"),
        _entry(3, 1300, "Coverage reiterated Overweight at UBS"),
        _entry(4, 1150, "MU PT Raised to $1,150 at Mizuho"),  # no 'from'
    ]
    assert signal_from_pt_revision(news, today=_TODAY)["drift"] is None


def test_out_of_window_excluded():
    news = [_entry(CFG.lookback_days + 30, 1500,
                   "PT raised to $1,500 from $1,000 at Firm")]
    assert signal_from_pt_revision(news, today=_TODAY)["drift"] is None


# ---------------------------------------------------------------------------
# Native-unit magnitude: drift = (new - prior)/spot, decay-weighted
# ---------------------------------------------------------------------------

def test_single_raise_change_in_implied_return():
    """1000 -> 1100 at spot 1000 = +10% change in implied return. age 0 →
    weight 1 → drift == +0.10 exactly (no scale factor)."""
    news = [_entry(0, 1100, "PT raised to $1,100 from $1,000 at Firm",
                   price_when_posted=1000.0)]
    s = signal_from_pt_revision(news, today=_TODAY)
    assert s["drift"] == pytest.approx(0.10, rel=1e-6)


def test_single_cut_negative_drift_sign_from_arithmetic():
    """900 from 1000 at spot 1000 = -10%. Direction comes purely from
    (new - prior); no keyword parsing involved."""
    news = [_entry(0, 900, "Target nudged to $900 from $1,000 at Firm",
                   price_when_posted=1000.0)]
    s = signal_from_pt_revision(news, today=_TODAY)
    assert s["drift"] == pytest.approx(-0.10, rel=1e-6)


def test_spot_denominator_is_price_when_posted():
    """(1100 - 1000)/spot, kept under drift_cap_abs to isolate the denominator.
    spot 1000 → +10%; spot 800 → +12.5%. Confirms the denominator is
    priceWhenPosted, not either target."""
    hi_spot = signal_from_pt_revision(
        [_entry(0, 1100, "PT raised to $1,100 from $1,000 at F", 1000.0)],
        today=_TODAY)["drift"]
    lo_spot = signal_from_pt_revision(
        [_entry(0, 1100, "PT raised to $1,100 from $1,000 at F", 800.0)],
        today=_TODAY)["drift"]
    assert hi_spot == pytest.approx(0.10, rel=1e-6)
    assert lo_spot == pytest.approx(0.125, rel=1e-6)


def test_comma_thousands_parsed():
    """'from $1,000' with thousands separator parses to 1000."""
    news = [_entry(0, 1500, "Micron PT raised to $1,500 from $1,000 at DA Davidson",
                   price_when_posted=1000.0)]
    s = signal_from_pt_revision(news, today=_TODAY)
    # (1500-1000)/1000 = +0.50 → clamped at per_entry_cap (0.50) → drift 0.50,
    # then bounded by drift_cap_abs.
    assert s["drift"] == pytest.approx(min(0.50, CFG.drift_cap_abs), rel=1e-6)


def test_per_entry_cap_clamps_outlier_via_notes():
    """A 1000 -> 5000 (+400%) revision is clamped to per_entry_cap before it
    can dominate. The aggregate drift also hits drift_cap_abs, so per_entry_cap
    is isolated via the Δimplied-return shown in the notes."""
    s = signal_from_pt_revision(
        [_entry(0, 5000, "PT raised to $5,000 from $1,000 at Firm", 1000.0)],
        today=_TODAY)
    assert f"{CFG.per_entry_cap*100:+.1f}%" in s["notes"]


def test_drift_cap_abs_enforced():
    news = [_entry(i, 5000, "PT raised to $5,000 from $1,000 at Firm", 1000.0)
            for i in range(8)]
    s = signal_from_pt_revision(news, today=_TODAY)
    assert abs(s["drift"]) <= CFG.drift_cap_abs + 1e-9
    assert s["drift"] == pytest.approx(CFG.drift_cap_abs, rel=1e-6)


def test_recency_weight_favors_newer():
    """A fresh -20% cut and a stale +20% raise of equal magnitude → the
    decay-weighted mean leans negative (fresher entry dominates)."""
    news = [
        _entry(0, 800, "PT lowered to $800 from $1,000 at FreshCo", 1000.0),
        _entry(80, 1200, "PT raised to $1,200 from $1,000 at StaleCo", 1000.0),
    ]
    assert signal_from_pt_revision(news, today=_TODAY)["drift"] < 0


def test_equal_offsetting_revisions_same_age_net_zero():
    """+20% and -20% at the same age cancel → ~0 drift."""
    news = [
        _entry(5, 1200, "PT raised to $1,200 from $1,000 at A", 1000.0),
        _entry(5, 800, "PT lowered to $800 from $1,000 at B", 1000.0),
    ]
    s = signal_from_pt_revision(news, today=_TODAY)
    assert s["drift"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Confidence + contract
# ---------------------------------------------------------------------------

def test_confidence_scales_with_count():
    one = [_entry(0, 1100, "PT raised to $1,100 from $1,000 at A", 1000.0)]
    assert signal_from_pt_revision(one, today=_TODAY)["confidence"] == "LOW"

    med = [_entry(i, 1100, f"PT raised to $1,100 from $1,000 at F{i}", 1000.0)
           for i in range(CFG.conf_medium_count)]
    assert signal_from_pt_revision(med, today=_TODAY)["confidence"] in ("MEDIUM", "HIGH")

    hi = [_entry(i, 1100, f"PT raised to $1,100 from $1,000 at F{i}", 1000.0)
          for i in range(CFG.conf_high_count)]
    assert signal_from_pt_revision(hi, today=_TODAY)["confidence"] == "HIGH"


def test_signal_dict_contract():
    news = [_entry(0, 1100, "PT raised to $1,100 from $1,000 at Firm", 1000.0)]
    s = signal_from_pt_revision(news, today=_TODAY)
    assert set(s) == {"drift", "confidence", "source_quality",
                      "sources_count", "notes"}
    assert s["source_quality"] == "PRIMARY"
    assert s["sources_count"] == 1


def test_malformed_entries_skipped_not_crashed():
    news = [
        "not a dict",
        {"publishedDate": "bad-date", "priceTarget": 100,
         "priceWhenPosted": 90, "title": "PT raised to $100 from $90 at X"},
        {"publishedDate": _TODAY.strftime("%Y-%m-%dT12:00:00Z"),
         "priceTarget": None, "priceWhenPosted": 100,
         "title": "PT raised to $1 from $1 at Y"},  # new_pt None → skip
        {"publishedDate": _TODAY.strftime("%Y-%m-%dT12:00:00Z"),
         "priceTarget": 100, "priceWhenPosted": 0,
         "title": "PT raised to $100 from $90 at Z"},  # spot 0 → skip
        _entry(0, 1100, "PT raised to $1,100 from $1,000 at OK", 1000.0),
    ]
    s = signal_from_pt_revision(news, today=_TODAY)
    assert s["sources_count"] == 1  # only the well-formed entry counted

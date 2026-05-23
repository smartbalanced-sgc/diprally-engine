"""Tests for PR #41 — post-parabola refusal gate.

Mirror of sacred #14 (falling-knife trend filter) for blow-off tops.
The gate fires when:
  - RSI ≥ rsi_threshold (overheated, default 70)
  - AND YTD return ≥ ytd_threshold (already had the run, default +150%)
  - AND no AI-surfaced bearish/two-sided de-rating catalyst in horizon

The _has_bearish_derating_catalyst helper is pure-function. The full
refusal-path test requires running the engine and isn't in scope here.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import PARABOLA_FILTER_MOM_30D_THRESHOLD
from src.engine import _has_bearish_derating_catalyst


TODAY = date(2026, 5, 22)


def _ai(*catalysts):
    """Build a fake effective_ai with the given catalysts."""
    return SimpleNamespace(catalysts=list(catalysts))


def _cat(name, direction, days_ahead):
    """Catalyst dict matching what Pass 1/Pass 2 emit."""
    d = TODAY + timedelta(days=days_ahead)
    return {
        "name": name,
        "type": "earnings",
        "date_or_window": d.strftime("%Y-%m-%d"),
        "magnitude": "high",
        "direction_risk": direction,
    }


def test_threshold_loaded_from_yaml():
    """PR #41 / PR #44 threshold must load from YAML, not hardcoded.
    Must be a positive momentum value (parabolic = explosive UP move)."""
    assert PARABOLA_FILTER_MOM_30D_THRESHOLD > 0.0
    assert PARABOLA_FILTER_MOM_30D_THRESHOLD < 5.0  # sanity bound


def test_bearish_catalyst_in_horizon_blocks_refusal():
    """A bearish de-rating catalyst means the math layer has a reason
    to anchor on mean-reversion → parabola filter does NOT fire."""
    # Patch datetime.now() to return TODAY for deterministic test.
    import src.engine as eng
    orig = eng.__dict__.get('_test_today_override')
    # The helper imports datetime locally; we mock at signals layer.
    ai = _ai(_cat("Q2 earnings miss", "bearish", 30))
    # Run the helper directly — uses datetime.now().date() which we
    # can't easily override here; instead, use a date 30 days from
    # real "now" to guarantee in-horizon.
    from datetime import datetime as _dt
    today = _dt.now().date()
    d = today + timedelta(days=30)
    ai2 = _ai({"name": "earnings", "type": "earnings",
                "date_or_window": d.strftime("%Y-%m-%d"),
                "magnitude": "high", "direction_risk": "bearish"})
    assert _has_bearish_derating_catalyst(ai2, horizon_days=60) is True


def test_twosided_catalyst_does_NOT_count_as_derating():
    """PR #45 design change: two-sided catalysts (generic earnings,
    sector readthrough, macro events) are the math layer's default
    assumption and do NOT specifically point toward de-rating.
    Asymmetry with sacred #14 is intentional — the parabola filter
    requires a BEARISH thesis, not a bidirectional one. Otherwise
    every parabola slips through on a generic 'earnings in horizon'
    catalyst, defeating the gate's purpose."""
    from datetime import datetime as _dt
    today = _dt.now().date()
    d = today + timedelta(days=20)
    ai = _ai({"name": "Q2 earnings", "type": "earnings",
               "date_or_window": d.strftime("%Y-%m-%d"),
               "magnitude": "high", "direction_risk": "two-sided"})
    assert _has_bearish_derating_catalyst(ai, horizon_days=60) is False


def test_bearish_skew_variant_counts_as_derating():
    """PR #46: Pass 1/Pass 2 commonly emit 'bearish-skew' for mean-
    reversion catalysts (e.g. 'profit-taking after +204% YTD').
    The filter accepts any direction_risk string starting with
    'bearish' so these lexical variants don't silently bypass the
    gate. Same semantic intent as PR #45 (specifically bearish, not
    bidirectional), just lexically tolerant."""
    from datetime import datetime as _dt
    today = _dt.now().date()
    d = today + timedelta(days=20)
    for variant in ("bearish", "bearish-skew", "Bearish",
                     "BEARISH-skew", "bearish/down"):
        ai = _ai({"name": "mean reversion", "type": "macro",
                   "date_or_window": d.strftime("%Y-%m-%d"),
                   "magnitude": "med", "direction_risk": variant})
        assert _has_bearish_derating_catalyst(ai, horizon_days=60) is True, \
            f"variant {variant!r} should count as bearish"


def test_bullish_only_catalyst_does_not_block_refusal():
    """A purely bullish catalyst does NOT provide a de-rating thesis —
    it would actually accelerate the parabolic move. Parabola filter
    should still fire."""
    from datetime import datetime as _dt
    today = _dt.now().date()
    d = today + timedelta(days=20)
    ai = _ai({"name": "FDA approval", "type": "regulatory",
               "date_or_window": d.strftime("%Y-%m-%d"),
               "magnitude": "high", "direction_risk": "bullish"})
    assert _has_bearish_derating_catalyst(ai, horizon_days=60) is False


def test_no_ai_means_no_de_rating_catalyst():
    """In --no-ai mode effective_ai is None → strict reading: no
    catalysts known → can't block parabola refusal."""
    assert _has_bearish_derating_catalyst(None, horizon_days=60) is False


def test_empty_catalysts_means_no_de_rating_catalyst():
    ai = _ai()
    assert _has_bearish_derating_catalyst(ai, horizon_days=60) is False


def test_catalyst_outside_horizon_doesnt_count():
    """A bearish catalyst beyond the horizon doesn't help the thesis —
    by the time it fires we're already long the parabola."""
    from datetime import datetime as _dt
    today = _dt.now().date()
    # 120 days ahead with horizon 60 → out of window
    d = today + timedelta(days=120)
    ai = _ai({"name": "next year earnings", "type": "earnings",
               "date_or_window": d.strftime("%Y-%m-%d"),
               "magnitude": "high", "direction_risk": "bearish"})
    assert _has_bearish_derating_catalyst(ai, horizon_days=60) is False


def test_catalyst_unparseable_date_skipped():
    """Defensive: catalysts with garbage dates are skipped."""
    ai = _ai({"name": "TBD", "type": "earnings",
               "date_or_window": "soon",
               "magnitude": "high", "direction_risk": "bearish"})
    assert _has_bearish_derating_catalyst(ai, horizon_days=60) is False

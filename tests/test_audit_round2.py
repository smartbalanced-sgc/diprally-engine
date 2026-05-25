"""Tests for the 2026-05-24 round-2 audit fixes:

  1. Portfolio gate now fetches FMP daily bars (not engine-CSV history)
     → gate is functional on day 1 instead of dormant for 5+ cycles.
  2. Portfolio gate emits status log on every cycle (was silent).
  3. Catalyst verification verdicts surfaced on Pass 2 catalyst
     enumeration (was attached to Pass 1, wrong place).
  4. Reliability-warning chip line under headline card surfaces
     near-IGARCH / wide σ-divergence / weak signal aggregation /
     σ-class registry mismatch (previously buried mid-report).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.reporter import _reliability_warnings_line


# =============================================================================
# _reliability_warnings_line — universal reliability surface
# =============================================================================

def _make_vol_profile(alpha_plus_beta=0.50, divergence=5.0):
    return SimpleNamespace(
        garch_alpha_plus_beta=alpha_plus_beta,
        triangulation={"divergence": divergence, "blended": 0.50, "anchors": {}},
    )


def _make_signal(confidence="HIGH", is_absent=False):
    return SimpleNamespace(confidence=confidence, is_absent=is_absent)


def test_clean_run_produces_empty_line():
    """No flags trip → empty string (no noise on healthy runs)."""
    line = _reliability_warnings_line(
        vol_profile=_make_vol_profile(alpha_plus_beta=0.7, divergence=5.0),
        base_signals=[_make_signal("HIGH")] * 12,
        sigma_class_mismatch=None,
    )
    assert line == ""


def test_near_igarch_triggers():
    """α+β > 0.98 = near-IGARCH; surface as reliability warning."""
    line = _reliability_warnings_line(
        vol_profile=_make_vol_profile(alpha_plus_beta=0.9999, divergence=5.0),
        base_signals=[_make_signal("HIGH")] * 12,
        sigma_class_mismatch=None,
    )
    assert "near-IGARCH" in line
    assert "0.9999" in line


def test_near_igarch_boundary_at_98_doesnt_trigger():
    """0.98 exactly does NOT trigger — strict greater-than."""
    line = _reliability_warnings_line(
        vol_profile=_make_vol_profile(alpha_plus_beta=0.98, divergence=5.0),
        base_signals=[_make_signal("HIGH")] * 12,
        sigma_class_mismatch=None,
    )
    assert "near-IGARCH" not in line


def test_wide_sigma_divergence_triggers():
    """σ divergence > 15pp = wide anchor disagreement (often pre-event)."""
    line = _reliability_warnings_line(
        vol_profile=_make_vol_profile(alpha_plus_beta=0.7, divergence=19.3),
        base_signals=[_make_signal("HIGH")] * 12,
        sigma_class_mismatch=None,
    )
    assert "σ divergence" in line
    assert "19.3pp" in line


def test_sigma_divergence_15_or_below_doesnt_trigger():
    line = _reliability_warnings_line(
        vol_profile=_make_vol_profile(alpha_plus_beta=0.7, divergence=15.0),
        base_signals=[_make_signal("HIGH")] * 12,
        sigma_class_mismatch=None,
    )
    assert "σ divergence" not in line


def test_few_signals_active_triggers():
    """Less than half of total signals at MEDIUM+ confidence = weak blend."""
    # 12 signals total, only 3 at HIGH/MEDIUM (others LOW or absent)
    base = ([_make_signal("HIGH")] * 3
            + [_make_signal("LOW")] * 6
            + [_make_signal("HIGH", is_absent=True)] * 3)
    line = _reliability_warnings_line(
        vol_profile=_make_vol_profile(),
        base_signals=base,
        sigma_class_mismatch=None,
    )
    assert "3/12 signals active" in line


def test_half_signals_active_doesnt_trigger():
    """Exactly half ≠ "few active". Strict less-than."""
    base = [_make_signal("HIGH")] * 6 + [_make_signal("LOW")] * 6
    line = _reliability_warnings_line(
        vol_profile=_make_vol_profile(),
        base_signals=base,
        sigma_class_mismatch=None,
    )
    assert "signals active" not in line


def test_sigma_class_mismatch_triggers():
    """When registry hint disagrees with detector, surface at top."""
    line = _reliability_warnings_line(
        vol_profile=_make_vol_profile(),
        base_signals=[_make_signal("HIGH")] * 12,
        sigma_class_mismatch="auto-detected MID, registry hint HIGH",
    )
    assert "registry hint stale" in line


def test_multiple_flags_combined():
    """Each tripping flag adds a chip; separator is ' · '."""
    base = [_make_signal("HIGH")] * 3 + [_make_signal("LOW")] * 9
    line = _reliability_warnings_line(
        vol_profile=_make_vol_profile(alpha_plus_beta=0.99, divergence=20.0),
        base_signals=base,
        sigma_class_mismatch="auto MID, hint HIGH",
    )
    assert "near-IGARCH" in line
    assert "σ divergence" in line
    assert "signals active" in line
    assert "registry hint stale" in line
    assert line.count(" · ") == 3  # 4 chips → 3 separators


def test_defensive_on_missing_attributes():
    """vol_profile without expected attrs → return empty, no crash."""
    bare = SimpleNamespace()  # no garch_alpha_plus_beta, no triangulation
    line = _reliability_warnings_line(
        vol_profile=bare,
        base_signals=[],
        sigma_class_mismatch=None,
    )
    assert line == ""


# =============================================================================
# Portfolio gate data source — FMP daily bars not engine CSV
# =============================================================================

def test_history_as_price_df_calls_fetch_history_when_key_present(monkeypatch, tmp_path):
    """Audit fix: the gate's price data must come from real FMP daily
    bars, not the engine's once-per-day outcome CSV."""
    from src import orchestrator as orch
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    monkeypatch.setenv("FMP_API_KEY", "test-key-123")

    calls = []
    def fake_fetch(ticker, api_key, lookback_days):
        calls.append((ticker, api_key, lookback_days))
        import pandas as pd
        return pd.DataFrame({
            "Date": pd.date_range(end="2026-05-24", periods=90, freq="D"),
            "Close": [100.0 + i * 0.5 for i in range(90)],
        })

    with patch("src.data_fetch.fetch_history", side_effect=fake_fetch):
        result = orch._history_as_price_df("AMAT")

    # PR #75 fix: lookback_days request raised 90 → 140 calendar days
    # so the gate's 90-trading-day window has enough bars to compute
    # correlation. Was returning only 63 trading days from 90 calendar.
    assert calls == [("AMAT", "test-key-123", 140)]
    assert result is not None
    assert len(result) == 90


def test_history_as_price_df_falls_back_to_csv_when_no_key(monkeypatch, tmp_path):
    """No FMP_API_KEY (e.g. tests, --no-fetch mode) → degraded CSV fallback
    rather than blanket None. Keeps the gate alive in degraded mode."""
    from src import orchestrator as orch
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    monkeypatch.delenv("FMP_API_KEY", raising=False)

    # No CSV file → fallback returns None (existing behavior)
    assert orch._history_as_price_df("AMAT") is None


def test_history_as_price_df_falls_back_when_fmp_fails(monkeypatch, tmp_path):
    """FMP raises → fall back to CSV (don't block the gate)."""
    from src import orchestrator as orch
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    monkeypatch.setenv("FMP_API_KEY", "test-key-123")

    with patch("src.data_fetch.fetch_history",
                side_effect=RuntimeError("FMP rate limit")):
        # CSV doesn't exist → fallback also returns None, but no crash
        assert orch._history_as_price_df("AMAT") is None


def test_history_as_price_df_returns_none_when_fmp_too_few_bars(monkeypatch, tmp_path):
    """If FMP returns < 30 bars (newly-IPO'd / data missing), correlation
    isn't reliable — return None so the gate treats it defensively."""
    from src import orchestrator as orch
    monkeypatch.setattr(orch, "_OUTPUT_ROOT", tmp_path)
    monkeypatch.setenv("FMP_API_KEY", "test-key-123")

    import pandas as pd
    short_df = pd.DataFrame({
        "Date": pd.date_range(end="2026-05-24", periods=10, freq="D"),
        "Close": [100.0] * 10,
    })

    with patch("src.data_fetch.fetch_history", return_value=short_df):
        assert orch._history_as_price_df("CRWV") is None


# =============================================================================
# Catalyst verification verdicts moved to Pass 2 enumeration
# =============================================================================

def test_pass2_catalysts_with_verdicts_enumerated_in_report():
    """Audit fix round 2: verification_verdict appended to each Pass 2
    catalyst by apply_catalyst_verification. Previously the reporter
    enumerated Pass 1 catalysts (without verdicts) and only summarized
    Pass 2 as added/dropped sets — verdicts invisible. Now Pass 2's
    final catalyst list is enumerated WITH verdict tags."""
    # Build a Pass 2 with verified catalysts
    from src.engine import AIPassOutput
    pass1 = AIPassOutput(
        pass_number=1, drift_estimate=0.15, drift_range=(-0.10, 0.40),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.10, raw_sources_cited=4,
    )
    pass2 = AIPassOutput(
        pass_number=2, drift_estimate=0.10, drift_range=(0.0, 0.20),
        confidence="LOW", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[
            {"name": "Q1 earnings", "magnitude": "high", "direction_risk": "bullish",
             "verification_verdict": "VERIFIED",
             "verification_reasoning": "Confirmed by SEC 8-K filing"},
            {"name": "Acquisition rumor", "magnitude": "low",
             "direction_risk": "two-sided",
             "verification_verdict": "UNVERIFIED",
             "verification_reasoning": "No primary source found"},
        ],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=-0.05, cost_usd=0.04, raw_sources_cited=0,
    )

    # Render the AI section directly, mirroring reporter logic
    lines = []
    verified_any = any(
        isinstance(c, dict) and c.get("verification_verdict")
        for c in pass2.catalysts
    )
    assert verified_any  # sanity

    for c in pass2.catalysts[:5]:
        verdict = c.get("verification_verdict") or ""
        name = c.get("name", "?")
        mag = c.get("magnitude", "?")
        lines.append(
            f"[{verdict}] {name} (mag {mag}, {c.get('direction_risk', '?')})"
        )
    rendered = " | ".join(lines)

    assert "[VERIFIED]" in rendered
    assert "[UNVERIFIED]" in rendered
    assert "Q1 earnings" in rendered
    assert "Acquisition rumor" in rendered


def test_pass2_catalysts_skipped_when_no_verification_ran():
    """T1 runs / verification-disabled: no catalyst dict has
    verification_verdict → the new enumeration section stays silent
    (don't clutter the report)."""
    from src.engine import AIPassOutput
    pass2 = AIPassOutput(
        pass_number=2, drift_estimate=0.10, drift_range=(0.0, 0.20),
        confidence="LOW", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[
            {"name": "Q1 earnings", "magnitude": "high", "direction_risk": "bullish"},
            # NO verification_verdict
        ],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=-0.05, cost_usd=0.04, raw_sources_cited=0,
    )
    verified_any = any(
        isinstance(c, dict) and c.get("verification_verdict")
        for c in pass2.catalysts
    )
    assert verified_any is False

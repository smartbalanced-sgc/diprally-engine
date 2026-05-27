"""Tests for PR #89 — parabola threshold recalibration + verdict renames +
plain-English AI surfacing + max-parallel-4 + 429-retry safety.

Context: previous 6 cycles produced 0 BUYs. Two root causes:
  1. Parabola filter thresholds (HIGH 80%, EXTREME 100%) were too tight
     for AI-cycle continuation — kept refusing MU/ARM/INTC/MRAM/RKLB.
  2. AI Pass 2 outputs were buried in raw CSV fields; the dashboard
     didn't translate them to plain English an investor could read.

PR #89 fixes:
  - YAML: parabola_mom_30d_threshold raised MID 0.5→0.8, HIGH 0.8→1.5,
    EXTREME 1.0→2.0.
  - Verdict names: REFUSED-* → friendlier OVEREXTENDED / EV-NEGATIVE /
    MATH-CONFLICT / DOWNTREND / CLOSE-CALL (display-only, internal
    verdict_state unchanged for CSV backward-compat).
  - Detail row renders AI Pass 2 catalysts in plain English with
    verdict-contextual framing.
  - max-parallel default 2 → 4 (empirically safe per FMP burst test).
  - _fmp_get retries once on HTTP 429 / network glitch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# 1. Parabola thresholds raised
# =============================================================================

def test_parabola_thresholds_raised_for_ai_cycle():
    from src.config import SIGMA_CLASSES
    assert SIGMA_CLASSES["MID"].parabola_mom_30d_threshold == 0.80
    assert SIGMA_CLASSES["HIGH"].parabola_mom_30d_threshold == 1.50
    assert SIGMA_CLASSES["EXTREME"].parabola_mom_30d_threshold == 2.00


def test_parabola_passes_for_typical_ai_cycle_momentum():
    """Names that consistently triggered the OLD threshold should
    PASS the new threshold. Each ticker's mom_30d from cycle 17:56:"""
    from src.config import SIGMA_CLASSES
    high = SIGMA_CLASSES["HIGH"].parabola_mom_30d_threshold
    extreme = SIGMA_CLASSES["EXTREME"].parabola_mom_30d_threshold
    # MU at +92% HIGH: was tripping OLD 80%; now under 150%
    assert 0.92 < high
    # ARM at +99% HIGH: same
    assert 0.99 < high
    # INTC at +94% HIGH: same
    assert 0.94 < high
    # MRAM at +182% EXTREME: was over OLD 100%; under new 200%
    assert 1.82 < extreme


def test_parabola_still_refuses_true_blowoffs():
    """Sanity: extreme moves still trigger refusal. We loosened, didn't disable."""
    from src.config import SIGMA_CLASSES
    high = SIGMA_CLASSES["HIGH"].parabola_mom_30d_threshold
    extreme = SIGMA_CLASSES["EXTREME"].parabola_mom_30d_threshold
    # +200% in 30d (HIGH-class doubling+) → still tripping
    assert 2.00 > high
    # +250% in 30d (EXTREME doubling+) → still tripping
    assert 2.50 > extreme


# =============================================================================
# 2. Verdict display names (friendly renames)
# =============================================================================

def test_verdict_display_names_friendly():
    from src.orchestrator import _verdict_display_name
    # Internal → display
    assert _verdict_display_name("REFUSED-PARABOLA") == "OVEREXTENDED"
    assert _verdict_display_name("REFUSED-EV") == "EV-NEGATIVE"
    assert _verdict_display_name("REFUSED-METHOD") == "MATH-CONFLICT"
    assert _verdict_display_name("REFUSED-TREND") == "DOWNTREND"
    assert _verdict_display_name("BELOW-THRESHOLD") == "CLOSE-CALL"
    # BUY / WAIT unchanged (already operator-friendly)
    assert _verdict_display_name("BUY") == "BUY"
    assert _verdict_display_name("WAIT") == "WAIT"
    # Unknown verdict passes through
    assert _verdict_display_name("MYSTERY") == "MYSTERY"


def test_dashboard_renders_friendly_verdict_names():
    from src import orchestrator as orch
    decisions = [
        orch.TickerDecision(
            ticker="MU", sigma_class="HIGH", tier="T2",
            ambiguity=0.30, qualifies_for_t2_plus=True,
            spot=895.0, dip_target=878.0, rally_target=940.0,
            p_round_trip=0.40, ev_bps_of_dip=-150.0,
            verdict="REFUSED-PARABOLA", status_note="mom_30d high",
            ev_direct_bps=-150.0, ev_wait_bps=-100.0,
            p_dip_filled=0.80, p_rally_hit=0.55,
        )
    ]
    html_out = orch._render_dashboard_html(decisions, None)
    # Friendly display name renders in the verdict pill
    assert ">OVEREXTENDED<" in html_out
    # Old name not surfaced as the verdict pill text (it's still in data-verdict
    # attribute, but the visible label is the friendly one)
    # Check the pill specifically — the friendly name should appear inside
    # <span class="verdict">.
    import re
    pill_matches = re.findall(r'<span class="verdict"[^>]*>([^<]+)</span>', html_out)
    assert "OVEREXTENDED" in pill_matches


# =============================================================================
# 3. Plain-English AI surfacing
# =============================================================================

def test_ai_findings_translates_catalysts_to_plain_english():
    from src import orchestrator as orch
    d = orch.TickerDecision(
        ticker="MU", sigma_class="HIGH", tier="T2",
        ambiguity=0.30, qualifies_for_t2_plus=True,
        spot=895.0, dip_target=878.0, rally_target=945.0,
        p_round_trip=0.50, ev_bps_of_dip=45.0,
        verdict="BUY", status_note="",
        ev_direct_bps=45.0, ev_wait_bps=28.0,
        p_dip_filled=0.84, p_rally_hit=0.76,
        ai_catalysts=[
            {"name": "Q1 earnings beat", "magnitude": "high",
             "direction_risk": "bullish", "date_or_window": "2026-05-27"},
            {"name": "AI memory cycle", "magnitude": "high",
             "direction_risk": "bullish", "date_or_window": "ongoing"},
            {"name": "Sector rotation", "magnitude": "med",
             "direction_risk": "two-sided", "date_or_window": "2026-07"},
        ],
        ai_agreement="agree",
        ai_narrative="strong",
    )
    findings = orch._ai_findings_plain_english(d)

    # Catalysts converted to plain-English lines
    lines = " ".join(findings["catalyst_lines"])
    assert "Q1 earnings beat" in lines
    assert "strong signal" in lines     # "high" → "strong signal"
    assert "likely to push price UP" in lines   # bullish → plain English
    assert "could go either way" in lines       # two-sided → plain English

    # AI conclusion combines agreement + narrative
    assert "agrees" in findings["ai_conclusion"].lower()
    assert "credible sources" in findings["ai_conclusion"].lower()

    # Verdict framing is BUY-context
    assert "make money on average" in findings["verdict_framing"]


def test_ai_findings_framing_changes_per_verdict():
    """Framing line is verdict-contextual — BUY frames as confirmation,
    EV-NEGATIVE as math-disagrees-with-AI, OVEREXTENDED as parabola."""
    from src import orchestrator as orch
    base = dict(
        ticker="X", sigma_class="HIGH", tier="T2",
        ambiguity=0.30, qualifies_for_t2_plus=True,
        spot=100.0, dip_target=98.0, rally_target=105.0,
        p_round_trip=0.40, ev_bps_of_dip=0.0,
        status_note="",
    )

    buy = orch._ai_findings_plain_english(orch.TickerDecision(verdict="BUY", **base))
    assert "make money on average" in buy["verdict_framing"]

    ev_neg = orch._ai_findings_plain_english(orch.TickerDecision(verdict="REFUSED-EV", **base))
    assert "below the safety bar" in ev_neg["verdict_framing"]

    para = orch._ai_findings_plain_english(orch.TickerDecision(verdict="REFUSED-PARABOLA", **base))
    assert "rallied so much" in para["verdict_framing"]

    method = orch._ai_findings_plain_english(orch.TickerDecision(verdict="REFUSED-METHOD", **base))
    assert "don't agree" in method["verdict_framing"]


def test_ai_findings_empty_when_no_data():
    from src import orchestrator as orch
    d = orch.TickerDecision(
        ticker="X", sigma_class="HIGH", tier="T0",
        ambiguity=0.30, qualifies_for_t2_plus=False,
        spot=100.0, dip_target=98.0, rally_target=105.0,
        p_round_trip=0.40, ev_bps_of_dip=0.0,
        verdict="WAIT", status_note="",
        ai_catalysts=[],  # no AI data
        ai_agreement="",
        ai_narrative="",
    )
    findings = orch._ai_findings_plain_english(d)
    assert findings["catalyst_lines"] == []
    assert findings["ai_conclusion"] == ""


# =============================================================================
# 4. max-parallel default 2 → 4
# =============================================================================

def test_orchestrate_default_max_parallel_is_4():
    """PR #89: empirically safe per FMP burst test 2026-05-27."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "tools/orchestrate.py", "--help"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    # The argparse --help output shows the default
    assert "default 4" in result.stdout or "default: 4" in result.stdout


# =============================================================================
# 5. _fmp_get retries on HTTP 429
# =============================================================================

def test_fmp_get_retries_on_429():
    """First call returns 429, second returns 200 — _fmp_get should
    succeed and return the parsed JSON from the second call."""
    from src import data_fetch
    call_count = [0]

    def mock_get(url, params=None, timeout=None):
        call_count[0] += 1
        if call_count[0] == 1:
            resp = MagicMock()
            resp.status_code = 429
            resp.headers = {"Retry-After": "1"}
            return resp
        else:
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {}
            resp.raise_for_status = lambda: None
            resp.json = lambda: [{"symbol": "AMAT", "price": 100.0}]
            return resp

    with patch.object(data_fetch.requests, "get", side_effect=mock_get):
        result = data_fetch._fmp_get("quote", "test-key", {"symbol": "AMAT"})

    assert call_count[0] == 2     # made the retry
    assert isinstance(result, list)
    assert result[0]["symbol"] == "AMAT"


def test_fmp_get_returns_none_on_persistent_failure():
    """If both attempts fail, return None gracefully (don't crash the cycle)."""
    from src import data_fetch

    def mock_get(url, params=None, timeout=None):
        raise data_fetch.requests.exceptions.ConnectionError("network down")

    with patch.object(data_fetch.requests, "get", side_effect=mock_get):
        result = data_fetch._fmp_get("quote", "test-key", {"symbol": "AMAT"})

    assert result is None

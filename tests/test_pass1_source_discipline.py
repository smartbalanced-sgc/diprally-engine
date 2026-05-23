"""Tests for PR #50 — Pass 1 source-quality discipline.

The Pass 1 prompt must instruct the model to anchor high-magnitude
catalysts on institutional sources (SEC filings, IR sites, top-tier
sell-side, primary newswire) and explicitly forbid retail / influencer
/ aggregator anchors. Smoke audits during W6 repeatedly caught Pass 1
anchoring on TimothySykes, capitalstreetfx, phemex, etc. for
high-magnitude catalysts; Pass 2 critiqued the choice but the bad
catalyst was already in the blend.

These tests verify the prompt CONTENT (that the discipline section
exists with the right anchors and the right forbidden sources). The
actual model behavior is exercised by smoke runs and tracked via
D-W10-1 calibration once 30+ days of data accumulate.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ai_layer import build_ai_pass1_prompt


@dataclass
class _Snap:
    ticker: str = "INTC"
    timestamp: datetime = datetime(2026, 5, 23, 12, 0)
    spot: float = 119.84
    rsi: float = 66.2
    mom_30d: float = 0.92
    ytd_return: float = 2.04
    sector: str = "Technology"


@dataclass
class _Vol:
    blended_sigma: float = 0.83


@dataclass
class _Sig:
    name: str
    mu_annual: float
    confidence: str


def _prompt():
    snap = _Snap()
    vol = _Vol()
    sigs = [_Sig("historical", 0.10, "MEDIUM")]
    return build_ai_pass1_prompt(
        ticker="INTC", snapshot=snap, vol_profile=vol,
        horizon_days=60, base_signals=sigs,
        self_earnings_date=None, peer_tickers=["AMD", "AVGO"],
    )


def test_prompt_contains_institutional_anchor_requirement():
    """High-magnitude catalysts must require institutional anchor."""
    p = _prompt()
    assert "INSTITUTIONAL anchor" in p
    # SEC / IR / sell-side / newswire all named explicitly.
    for source in ("sec.gov", "Bloomberg", "Reuters", "WSJ", "FT",
                    "GS", "MS", "JPM"):
        assert source in p, f"institutional source {source!r} missing from Pass 1 prompt"


def test_prompt_explicitly_forbids_retail_anchors():
    """The specific retail sources Pass 1 was caught anchoring on
    during W6 smokes must be named in the forbidden list."""
    p = _prompt()
    for forbidden in ("TimothySykes", "capitalstreetfx", "phemex",
                       "TipRanks", "UnusualWhales", "MotleyFool",
                       "Seeking Alpha", "StocksToTrade", "247WallSt",
                       "heygotrade", "tradethepool"):
        assert forbidden in p, f"forbidden source {forbidden!r} missing from Pass 1 prompt"


def test_prompt_has_corporate_action_anchor_restriction():
    """M&A / secondary / spin-off catalysts have the strictest source
    requirement (SEC + IR + primary newswire ONLY)."""
    p = _prompt()
    assert "corporate-action" in p or "M&A" in p
    # Must explicitly mention 8-K / S-1 / S-3 filings.
    for filing in ("8-K", "S-1", "10-Q"):
        assert filing in p, f"SEC filing {filing!r} not referenced for corporate-action anchors"


def test_prompt_provides_downgrade_path_when_no_anchor():
    """If no institutional anchor found, Pass 1 must downgrade magnitude
    rather than dropping the catalyst entirely OR anchoring on retail."""
    p = _prompt()
    assert "DOWNGRADE" in p
    # Specifically: high → med/low when no institutional anchor.
    assert ("med" in p or "low" in p)


def test_prompt_still_returns_valid_json_structure():
    """The discipline section is additive — the core JSON contract
    must still be present and unchanged."""
    p = _prompt()
    assert "drift_estimate_annualized" in p
    assert "catalysts" in p
    assert "bull_factors" in p
    assert "bear_factors" in p
    # Sources field per catalyst still required.
    assert '"sources"' in p


def test_prompt_does_not_block_secondary_citations():
    """Retail sources can still be cited as SECONDARY alongside
    institutional anchors — the discipline forbids them as PRIMARY
    anchor only."""
    p = _prompt()
    assert "SECONDARY" in p or "secondary" in p


def test_prompt_includes_pr_marker_for_audit():
    """The discipline section is tagged with PR #50 so future
    engineers can trace why this guardrail exists."""
    p = _prompt()
    assert "PR #50" in p

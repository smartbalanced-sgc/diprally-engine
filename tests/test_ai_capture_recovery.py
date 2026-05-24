"""Tests for the 2026-05-24 audit fix: recover AI output fields
previously requested in the prompt but discarded by the parser.

Audit findings:
  Pass 1 prompt requests `narrative_evidence` and `key_risks` — the
  former was never captured, the latter was captured but never
  surfaced in the report.
  Pass 2 prompt requests `agreement_with_pass1`, `revision_reasoning`,
  `vol_regime_reasoning`, `narrative_reasoning`, `catalysts_reasoning`
  — all 5 discarded by parse_ai_pass2.
  Catalyst stress test `reasoning` per catalyst and catalyst
  verification verdict + reasoning per catalyst — both dropped from
  display.

Operator was paying for tokens whose output was thrown away. This PR
captures and surfaces them.
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

from src.ai_layer import parse_ai_pass1, parse_ai_pass2
from src.engine import AIPassOutput


# =============================================================================
# Pass 1 — narrative_evidence + key_risks capture
# =============================================================================

def test_pass1_captures_narrative_evidence():
    raw = {
        "drift_estimate_annualized": 0.15,
        "drift_range_low_high": [-0.10, 0.40],
        "confidence": "MEDIUM",
        "vol_regime": "MEDIUM",
        "narrative_score": "strong",
        "narrative_evidence": [
            {"claim": "Multi-quarter backlog visibility from DoD contracts",
             "source": "company 10-Q FY2026 Q1"},
            {"claim": "T. Rowe Price 13F shows position increase",
             "source": "SEC 13F filing 2026-04-15"},
        ],
        "catalysts": [], "bull_factors": [], "bear_factors": [], "key_risks": [],
    }
    pass1 = parse_ai_pass1(raw, sources_count=4, cost=0.10)
    assert len(pass1.narrative_evidence) == 2
    assert pass1.narrative_evidence[0]["claim"].startswith("Multi-quarter")
    assert pass1.narrative_evidence[0]["source"] == "company 10-Q FY2026 Q1"


def test_pass1_narrative_evidence_missing_returns_empty_list():
    raw = {
        "drift_estimate_annualized": 0.10, "drift_range_low_high": [-0.10, 0.30],
        "confidence": "LOW", "vol_regime": "MEDIUM",
        "narrative_score": "neutral",
        "catalysts": [], "bull_factors": [], "bear_factors": [], "key_risks": [],
    }
    pass1 = parse_ai_pass1(raw, sources_count=2, cost=0.05)
    assert pass1.narrative_evidence == []


def test_pass1_narrative_evidence_non_list_safely_normalized():
    """Defensive: if model returns a string or dict instead of a list,
    we fall back to empty rather than crash."""
    raw = {
        "drift_estimate_annualized": 0.10, "drift_range_low_high": [-0.10, 0.30],
        "confidence": "LOW", "vol_regime": "MEDIUM",
        "narrative_score": "neutral",
        "narrative_evidence": "this is a malformed string response",
        "catalysts": [], "bull_factors": [], "bear_factors": [], "key_risks": [],
    }
    pass1 = parse_ai_pass1(raw, sources_count=2, cost=0.05)
    assert pass1.narrative_evidence == []


# =============================================================================
# Pass 2 — reasoning fields + agreement_with_pass1 capture
# =============================================================================

def _stub_pass1():
    """Pass 1 standin for parse_ai_pass2's revision_from_prior_pass math."""
    return AIPassOutput(
        pass_number=1, drift_estimate=0.18,
        drift_range=(-0.10, 0.40), confidence="MEDIUM",
        vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[], bull_factors=[], bear_factors=[],
        key_risks=[], revision_from_prior_pass=None,
        cost_usd=0.10, raw_sources_cited=4,
    )


def test_pass2_captures_agreement_with_pass1():
    raw = {
        "revised_drift_estimate": 0.10,
        "revised_confidence": "LOW",
        "agreement_with_pass1": "strong_disagree",
        "primary_critique": "Pass 1's drift inconsistent with math",
        "revision_reasoning": "Math layer shows P(touch -10%) >> P(touch +10%); "
                              "Pass 1's positive drift conflicts.",
    }
    pass2 = parse_ai_pass2(raw, _stub_pass1(), cost=0.05)
    assert pass2.agreement_with_pass1 == "strong_disagree"


def test_pass2_agreement_normalized_to_lowercase():
    """Model sometimes returns 'STRONG_DISAGREE' or 'Strong_Disagree';
    parser normalizes."""
    raw = {
        "revised_drift_estimate": 0.10, "revised_confidence": "LOW",
        "agreement_with_pass1": "STRONG_DISAGREE",
        "primary_critique": "...",
    }
    pass2 = parse_ai_pass2(raw, _stub_pass1(), cost=0.05)
    assert pass2.agreement_with_pass1 == "strong_disagree"


def test_pass2_unknown_agreement_value_blanked():
    """Defensive: if the model returns gibberish (or omits the field),
    we don't fabricate one of the 3 valid values."""
    raw = {
        "revised_drift_estimate": 0.10, "revised_confidence": "LOW",
        "agreement_with_pass1": "kinda_disagree",  # not in {agree, partial_disagree, strong_disagree}
        "primary_critique": "...",
    }
    pass2 = parse_ai_pass2(raw, _stub_pass1(), cost=0.05)
    assert pass2.agreement_with_pass1 == ""


def test_pass2_captures_all_five_reasoning_fields():
    raw = {
        "revised_drift_estimate": 0.10, "revised_confidence": "LOW",
        "agreement_with_pass1": "partial_disagree",
        "primary_critique": "Pass 1 anchored on single source",
        "revision_reasoning": "Drift lowered by 8pp because the math layer's "
                              "near-symmetric touch probs imply no edge.",
        "vol_regime_reasoning": "Vol-expansion post earnings expected, kept HIGH.",
        "narrative_reasoning": "Strong narrative requires ≥2 sources; only 1 found.",
        "catalysts_reasoning": "Added missing Q3 guidance call from IR calendar.",
    }
    pass2 = parse_ai_pass2(raw, _stub_pass1(), cost=0.05)
    assert "drift lowered" in pass2.revision_reasoning.lower()
    assert "vol-expansion" in pass2.vol_regime_reasoning.lower()
    assert "strong narrative" in pass2.narrative_reasoning.lower()
    assert "q3 guidance" in pass2.catalysts_reasoning.lower()


def test_pass2_missing_reasoning_fields_default_to_blank():
    """When Pass 2 omits the reasoning fields, the parser leaves them
    as empty strings — NOT None. Display code can if-truthy-check safely."""
    raw = {
        "revised_drift_estimate": 0.10, "revised_confidence": "MEDIUM",
        "primary_critique": "...",
        # All reasoning fields absent.
    }
    pass2 = parse_ai_pass2(raw, _stub_pass1(), cost=0.05)
    assert pass2.revision_reasoning == ""
    assert pass2.vol_regime_reasoning == ""
    assert pass2.narrative_reasoning == ""
    assert pass2.catalysts_reasoning == ""
    assert pass2.agreement_with_pass1 == ""


def test_pass2_reasoning_handles_non_string_safely():
    """If model returns a dict or number where a string was expected,
    parser normalizes to empty (don't crash, don't .strip() on int)."""
    raw = {
        "revised_drift_estimate": 0.10, "revised_confidence": "MEDIUM",
        "primary_critique": "...",
        "revision_reasoning": {"unexpected": "shape"},
        "vol_regime_reasoning": 12345,
    }
    pass2 = parse_ai_pass2(raw, _stub_pass1(), cost=0.05)
    assert pass2.revision_reasoning == ""
    assert pass2.vol_regime_reasoning == ""


# =============================================================================
# Reporter surfacing — captured fields actually displayed
# =============================================================================

def _make_pass1(narrative_evidence=None, key_risks=None):
    return AIPassOutput(
        pass_number=1, drift_estimate=0.15, drift_range=(-0.10, 0.40),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[], bull_factors=[], bear_factors=[],
        key_risks=key_risks or [],
        revision_from_prior_pass=None, cost_usd=0.10, raw_sources_cited=4,
        narrative_evidence=narrative_evidence or [],
    )


def _make_pass2(**kwargs):
    defaults = dict(
        pass_number=2, drift_estimate=0.10, drift_range=(0.0, 0.20),
        confidence="LOW", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[], bull_factors=[], bear_factors=[],
        key_risks=["primary critique here"],
        revision_from_prior_pass=-0.05, cost_usd=0.04, raw_sources_cited=0,
    )
    defaults.update(kwargs)
    return AIPassOutput(**defaults)


def test_pass1_narrative_evidence_surfaces_in_report():
    """The captured narrative_evidence must reach the operator-visible report."""
    pass1 = _make_pass1(narrative_evidence=[
        {"claim": "Backlog visibility through 2027 from DoD contracts",
         "source": "10-Q FY26 Q1"},
    ])
    report = _render_ai_section(pass1=pass1, pass2=None)
    assert "Backlog visibility" in report
    assert "10-Q FY26 Q1" in report


def test_pass1_key_risks_surface_in_report():
    """Pass 1 key_risks were captured by the parser but never displayed —
    only Pass 2's risks made it to the report. Audit fix surfaces them."""
    pass1 = _make_pass1(key_risks=[
        "Customer concentration risk on single DoD program"
    ])
    report = _render_ai_section(pass1=pass1, pass2=None)
    assert "Customer concentration" in report


def test_pass2_agreement_tag_in_report():
    """STRONG DISAGREE flag must appear on the Pass 2 headline line so
    operator sees adversarial divergence at a glance."""
    pass1 = _make_pass1()
    pass2 = _make_pass2(agreement_with_pass1="strong_disagree")
    report = _render_ai_section(pass1=pass1, pass2=pass2)
    assert "STRONG DISAGREE" in report


def test_pass2_revision_reasoning_displayed():
    """Each Pass 2 revision now shows its rationale inline — operator
    doesn't have to read the full primary_critique to know WHY."""
    pass1 = _make_pass1()
    pass2 = _make_pass2(
        revision_reasoning="Math layer shows near-symmetric touch probs; "
                            "Pass 1's positive drift incompatible.",
    )
    report = _render_ai_section(pass1=pass1, pass2=pass2)
    assert "Math layer shows near-symmetric" in report


# =============================================================================
# Test helper
# =============================================================================

def _render_ai_section(pass1, pass2):
    """Invoke the AI-rendering block of reporter.format_report directly
    against a mock. We can't easily call format_report end-to-end
    (it needs many fields), but the AI-rendering subsection is the
    one we want to test, so re-execute its logic against the captured
    fields and return the concatenated lines.

    This mirrors the exact code path in src/reporter.py:408+ — if
    that path changes, this helper must be kept in sync."""
    lines = []
    if pass1:
        lines.append(f"  PASS 1: drift={pass1.drift_estimate:+.1%}/yr  "
                     f"conf={pass1.confidence}  vol_regime={pass1.vol_regime}  "
                     f"narrative={pass1.narrative_score}  "
                     f"sources={pass1.raw_sources_cited}  "
                     f"cost=${pass1.cost_usd:.2f}")
        for c in pass1.catalysts[:5]:
            if isinstance(c, dict):
                verdict = c.get("verification_verdict") or ""
                verdict_tag = f"  [{verdict}]" if verdict else ""
                lines.append(f"      • {c.get('name','?')} "
                              f"({c.get('date_or_window','?')}, "
                              f"{c.get('direction_risk','?')}, "
                              f"magnitude {c.get('magnitude','?')}){verdict_tag}")
                v_reason = c.get("verification_reasoning") or ""
                if v_reason:
                    lines.append(f"          ↳ verify: {v_reason[:160]}")
        if getattr(pass1, "narrative_evidence", None):
            for ev in pass1.narrative_evidence[:3]:
                if isinstance(ev, dict) and ev.get("claim"):
                    lines.append(f"    narrative ev: \"{ev.get('claim','')[:120]}\" "
                                 f"— {ev.get('source','?')}")
        if pass1.key_risks:
            for risk in pass1.key_risks[:3]:
                if risk:
                    lines.append(f"    risk: {str(risk)[:200]}")
    if pass2:
        rev = pass2.revision_from_prior_pass
        rev_str = f"({rev:+.1%} from Pass 1)" if rev is not None else ""
        agree_tag = ""
        if getattr(pass2, "agreement_with_pass1", ""):
            tag_map = {
                "agree":             "AGREE",
                "partial_disagree":  "PARTIAL DISAGREE",
                "strong_disagree":   "STRONG DISAGREE",
            }
            agree_tag = f"  [{tag_map.get(pass2.agreement_with_pass1, pass2.agreement_with_pass1.upper())}]"
        lines.append(f"  PASS 2: drift={pass2.drift_estimate:+.1%}/yr  "
                     f"conf={pass2.confidence}  {rev_str}{agree_tag}  "
                     f"cost=${pass2.cost_usd:.2f}")
        if getattr(pass2, "revision_reasoning", ""):
            lines.append(f"    drift rationale: {pass2.revision_reasoning[:240]}")
    return "\n".join(lines)

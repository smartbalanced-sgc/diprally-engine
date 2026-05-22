"""AI layer — two-pass adversarial critique with numeric outputs.

Every AI output is an arithmetic input to the model, never display prose
(sacred decision #10). Pass 2 wins (#7).

W0 keeps the v2 prompts unchanged. W1 swaps Opus for Sonnet/Haiku on Pass 2
and stress, adds caching, dispatches via the budget broker.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from src.config import (
    MODEL_OPUS,
    WEB_SEARCH_PER_USE,
    pricing_for_model,
)


# =============================================================================
# Anthropic client + cost computation
# =============================================================================

def _anthropic_client():
    """Lazy init."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)
    except Exception as e:
        print(f"   WARNING: anthropic client init failed: {e}")
        return None


def compute_ai_cost(response, model_id: str = MODEL_OPUS, had_web_search: bool = False) -> float:
    """Cost of an Anthropic API call. Dispatches pricing on model_id.

    Honest fallback: returns 0.0 when usage data is unreadable. Earlier code
    returned 0.30 / 0.05 as a guess, which over-reported a $0 outcome (no
    actual API call) and under-reported a $1+ outcome — neither is acceptable
    for the $2/day budget broker. 0.0 is the correct null value; the caller
    should log a warning when usage is missing and treat it as a soft error.
    """
    try:
        u = response.usage
        in_rate, out_rate = pricing_for_model(model_id)
        cost = u.input_tokens * in_rate + u.output_tokens * out_rate
        ws_uses = 0
        stu = getattr(u, "server_tool_use", None)
        if stu is not None:
            ws_uses = getattr(stu, "web_search_requests", 0) or 0
        if not ws_uses and had_web_search:
            ws_uses = 1
        cost += ws_uses * WEB_SEARCH_PER_USE
        return float(cost)
    except Exception as e:
        print(f"   WARNING: compute_ai_cost could not read response.usage: {e}")
        return 0.0


# =============================================================================
# Pass 1 — data gathering + multi-hypothesis catalysts
# =============================================================================

def build_ai_pass1_prompt(
    ticker,
    snapshot,
    vol_profile,
    horizon_days,
    base_signals,
    self_earnings_date,
    peer_tickers,
):
    """Pass 1: data gathering + multi-hypothesis catalyst identification.

    Hard mandate: STRUCTURED JSON with ≥5 catalyst candidates, each cited from
    ≥2 distinct sources.
    """
    today = snapshot.timestamp.strftime("%Y-%m-%d")
    base_signal_summary = "\n".join(
        f"  - {s.name}: mu={s.mu_annual:+.1%}/yr conf={s.confidence}"
        for s in base_signals
    )
    earnings_str = (
        self_earnings_date.strftime("%Y-%m-%d")
        if self_earnings_date else "unknown"
    )

    return f"""Analyse {ticker} for a 60-day round-trip swing trade.
Today: {today}. Spot: ${snapshot.spot:.2f}. Sector: {snapshot.sector}.
σ blended: {vol_profile.blended_sigma:.1%}. RSI: {snapshot.rsi:.1f}. 30d mom: {snapshot.mom_30d:+.1%}. YTD: {snapshot.ytd_return:+.1%}.
Next own earnings: {earnings_str}. Peers: {', '.join(peer_tickers)}.

Base signal blend (math-derived):
{base_signal_summary}

OUTPUT — single JSON object. NO PROSE BEFORE OR AFTER. NO MARKDOWN FENCES. STRINGS MUST NOT CONTAIN UNESCAPED NEWLINES. Keep each string < 250 chars.

{{
"drift_estimate_annualized": 0.20,
"drift_range_low_high": [-0.20, 0.50],
"confidence": "MEDIUM",
"vol_regime": "MEDIUM",
"narrative_score": "neutral",
"narrative_evidence": [{{"claim": "short", "source": "publisher"}}],
"catalysts": [{{"name": "short name", "type": "earnings", "date_or_window": "YYYY-MM-DD", "magnitude": "med", "direction_risk": "two-sided", "sources": ["src1", "src2"]}}],
"bull_factors": [{{"factor": "concise factor", "weight": "med", "sources": ["src1", "src2"]}}],
"bear_factors": [{{"factor": "concise factor", "weight": "med", "sources": ["src1", "src2"]}}],
"key_risks": ["short risk 1", "short risk 2"]
}}

RULES:
- vol_regime: HIGH if post-event vol expansion expected; LOW if vol-collapse signal; MEDIUM otherwise.
- narrative_score: "strong" only if ≥2 sources defend a structural multi-quarter story; else "neutral".
- catalysts: list 3-5 candidates, each with ≥2 sources. Concise names.
- bull_factors and bear_factors: each list 2-4 items, concise (<200 chars).
- key_risks: 2-3 risks, one short sentence each.
- Return ONLY the JSON object. No preamble. No explanation. No markdown.
"""


# =============================================================================
# Pass 2 — adversarial critique. Pass 2 wins.
# =============================================================================

def build_ai_pass2_prompt(
    ticker,
    snapshot,
    pass1,
    mc_marginal_summary,
    sigma_triangulation_summary,
    prior_posterior_drift,
):
    """Pass 2: ADVERSARIAL critique of Pass 1. Pass 2 drift REPLACES Pass 1
    in the signal blend (#7).
    """
    from src.signals import _factor_weight

    def _safe_name(c):
        return c.get("name", "?") if isinstance(c, dict) else str(c)

    def _safe_factor(f):
        return f.get("factor", str(f)) if isinstance(f, dict) else str(f)

    pass1_summary = {
        "drift_estimate": pass1.drift_estimate,
        "drift_range": list(pass1.drift_range),
        "confidence": pass1.confidence,
        "vol_regime": pass1.vol_regime,
        "narrative_score": pass1.narrative_score,
        "catalysts_count": len(pass1.catalysts),
        "catalyst_names": [_safe_name(c) for c in pass1.catalysts],
        "bull_factors_high": [_safe_factor(f) for f in pass1.bull_factors if _factor_weight(f) == "high"],
        "bear_factors_high": [_safe_factor(f) for f in pass1.bear_factors if _factor_weight(f) == "high"],
    }
    prior_str = f"{prior_posterior_drift:+.1%}/yr" if prior_posterior_drift is not None else "n/a (no history)"
    return f"""You are PASS 2 — an adversarial critic of Pass 1's analysis of {ticker}.

PASS 1 PRODUCED:
{json.dumps(pass1_summary, indent=2)}

INDEPENDENT MATH LAYER SAYS:
- σ blended (5-anchor): {sigma_triangulation_summary['blended']:.1%}
- σ divergence: {sigma_triangulation_summary['divergence']:.1f}pp ({'tight' if sigma_triangulation_summary['divergence'] < 5 else 'wide'})
- Closed-form P(touch +10% from spot in horizon): {mc_marginal_summary.get('p_up_10pct', 'n/a')}
- Closed-form P(touch -10% from spot in horizon): {mc_marginal_summary.get('p_down_10pct', 'n/a')}
- Prior posterior drift (yesterday): {prior_str}

YOUR JOB: critique Pass 1. Find the most likely error. Return JSON:

{{
  "agreement_with_pass1": "agree" | "partial_disagree" | "strong_disagree",
  "primary_critique": "Specific error or weakness in Pass 1",
  "missing_catalysts": ["catalysts Pass 1 missed, if any"],
  "revised_drift_estimate": <float, your corrected annualised drift>,
  "revised_confidence": "LOW" | "MEDIUM" | "HIGH",
  "revision_reasoning": "Why you revised (or kept) Pass 1's estimate",
  "vol_regime_concur": true | false,
  "narrative_score_concur": true | false
}}

ADVERSARIAL POSTURE:
- DO NOT rubber-stamp Pass 1. If Pass 1 is right, say so explicitly with reasoning.
- If Pass 1's drift estimate is inconsistent with the math (e.g., very bullish but stock has touched dip more than rally in MC), critique it.
- If Pass 1 missed a known catalyst in the horizon, flag it.
- If Pass 1 anchored on single source where multiple were available, critique it.
- Return ONLY valid JSON.
"""


# =============================================================================
# AI dispatch + JSON extraction
# =============================================================================

def call_ai_pass(prompt, max_tokens=3000, pass_label="Pass"):
    """Call Claude Opus, parse JSON, return (parsed, cost, sources_cited).

    Returns (None, 0.0, 0) on failure.
    """
    client = _anthropic_client()
    if client is None:
        print(f"⚠️  No Anthropic client — {pass_label} skipped")
        return None, 0.0, 0

    try:
        response = client.messages.create(
            model=MODEL_OPUS,
            max_tokens=max_tokens,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            messages=[{"role": "user", "content": prompt}],
        )
        cost = compute_ai_cost(response, model_id=MODEL_OPUS, had_web_search=True)
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        full_text = "\n".join(text_parts).strip()

        start = full_text.find("{")
        end = full_text.rfind("}")
        if start < 0 or end < 0:
            print(f"⚠️  {pass_label}: no JSON found in response")
            return None, cost, 0
        json_text = full_text[start:end + 1]

        try:
            parsed = json.loads(json_text, strict=False)
        except json.JSONDecodeError as e:
            import re
            sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', json_text)
            try:
                parsed = json.loads(sanitized, strict=False)
            except json.JSONDecodeError as e2:
                print(f"⚠️  {pass_label}: JSON parse error after sanitisation: {e2}")
                print(f"   (first 400 chars of response): {full_text[:400]}")
                return None, cost, 0

        sources = set()

        def collect_sources(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("sources", "url_or_publication"):
                        if isinstance(v, list):
                            sources.update(str(s) for s in v if s)
                        elif v:
                            sources.add(str(v))
                    else:
                        collect_sources(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect_sources(item)

        collect_sources(parsed)
        return parsed, cost, len(sources)
    except Exception as e:
        print(f"⚠️  {pass_label} call failed: {e}")
        return None, 0.0, 0


def call_ai_catalyst_stress_test(ticker, spot, dip_price, rally_price,
                                  catalysts, horizon_days):
    """Top-3 catalyst impact: directional drift if disappoints by 20%."""
    client = _anthropic_client()
    if client is None or not catalysts:
        return [], 0.0

    top = [c for c in catalysts[:3] if isinstance(c, dict)]
    if not top:
        return [], 0.0
    prompt = f"""For {ticker} at spot ${spot:.2f}, dip target ${dip_price:.0f}, rally target ${rally_price:.0f},
60-day horizon. For each catalyst below, estimate the directional drift impact
(annualised pp) if the catalyst disappoints by 20% on its key metric.

Catalysts:
{json.dumps([{'name': c.get('name'), 'date': c.get('date_or_window'), 'direction': c.get('direction_risk')} for c in top], indent=2)}

Return JSON list, one per catalyst:
[
  {{"catalyst_name": "...", "drift_shock_pp_on_disappointment": <float, signed pp e.g. -8.0 for -8pp>, "reasoning": "..."}},
  ...
]
Return ONLY valid JSON list.
"""
    try:
        response = client.messages.create(
            model=MODEL_OPUS,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        cost = compute_ai_cost(response, model_id=MODEL_OPUS, had_web_search=False)
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        text = "\n".join(text_parts).strip()
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end < 0:
            return [], cost
        parsed = json.loads(text[start:end + 1])
        return parsed if isinstance(parsed, list) else [], cost
    except Exception as e:
        print(f"⚠️  Catalyst stress test failed: {e}")
        return [], 0.0


# =============================================================================
# Parsers — JSON → dataclass.
# These import the dataclass from engine.py to keep the dataclass alongside
# its other engine-side users; we resolve circularity by deferring the import.
# =============================================================================

def parse_ai_pass1(raw, sources_count, cost):
    """Convert Pass 1 JSON to AIPassOutput."""
    from src.engine import AIPassOutput
    drift_range = raw.get("drift_range_low_high", [0.0, 0.0])
    return AIPassOutput(
        pass_number=1,
        drift_estimate=float(raw.get("drift_estimate_annualized", 0.0)),
        drift_range=(float(drift_range[0]), float(drift_range[1])) if len(drift_range) == 2 else (0.0, 0.0),
        confidence=str(raw.get("confidence", "LOW")).upper(),
        vol_regime=str(raw.get("vol_regime", "MEDIUM")).upper(),
        narrative_score=str(raw.get("narrative_score", "neutral")).lower(),
        catalysts=raw.get("catalysts", []) or [],
        bull_factors=raw.get("bull_factors", []) or [],
        bear_factors=raw.get("bear_factors", []) or [],
        key_risks=raw.get("key_risks", []) or [],
        revision_from_prior_pass=None,
        cost_usd=cost,
        raw_sources_cited=sources_count,
    )


def parse_ai_pass2(raw, pass1_drift, cost):
    """Convert Pass 2 JSON to AIPassOutput. Pass 2's drift replaces Pass 1 in blend."""
    from src.engine import AIPassOutput
    revised = float(raw.get("revised_drift_estimate", pass1_drift))
    return AIPassOutput(
        pass_number=2,
        drift_estimate=revised,
        drift_range=(revised - 0.10, revised + 0.10),
        confidence=str(raw.get("revised_confidence", "LOW")).upper(),
        vol_regime="MEDIUM",
        narrative_score="neutral",
        catalysts=[],
        bull_factors=[],
        bear_factors=[],
        key_risks=[raw.get("primary_critique", "")] + raw.get("missing_catalysts", []),
        revision_from_prior_pass=revised - pass1_drift,
        cost_usd=cost,
        raw_sources_cited=0,
    )

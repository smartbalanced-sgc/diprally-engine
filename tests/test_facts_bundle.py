"""Regression tests for the 2026-05-30 Stage 0 facts bundle.

DEFECT: AI Pass 1/Pass 2 prompts were given only collapsed signal SUMMARIES
(e.g. "pt_revision: mu=+1.2% conf=MEDIUM") instead of the structured FMP
data the engine had already fetched. AI had to web_search to re-discover
what FMP already knew, and routinely hallucinated plausible-looking
analyst names / catalysts that didn't match reality (smoking gun: Pass 2
named "BofA, Citi, HSBC, Melius" as the MU PT-raise cluster when the real
firms were DA Davidson, Mizuho, Barclays, UBS, Melius — entirely
different names except for Melius).

FIX: src/facts_bundle.py reshapes the engine's pre-fetched data into a
structured ground-truth envelope that's included verbatim in Pass 1 and
Pass 2 prompts. AI must cite bundle fields and reserve web_search for
incremental info only.

These tests guard:
  1. Bundle construction from synthetic pre-fetched data
  2. bundle_to_prompt_block serialization + char-cap trim behavior
  3. Pass 1 prompt includes the bundle when given
  4. Pass 2 prompt includes the bundle when given
  5. Backward compat: both prompts still work without bundle (default "")
"""
from __future__ import annotations

from datetime import datetime

import pytest

from src.facts_bundle import build_facts_bundle, bundle_to_prompt_block


# ---------------------------------------------------------------------------
# 1. Bundle construction
# ---------------------------------------------------------------------------

def _synth_inputs():
    """Inputs matching the engine's call signature; shaped after MU 2026-05-30."""
    return dict(
        ticker="MU", spot=971.00, sigma_blended=0.8994, sigma_class="HIGH",
        rsi=68.0, mom_5d=0.04, mom_30d=0.45, ytd_return=1.0,
        horizon_days=20, peer_tickers=["SNDK", "WDC", "STX"],
        self_earnings_date=datetime(2026, 6, 24),
        peer_earnings_dates=[],
        profile={"sector": "Technology", "industry": "Semiconductors",
                 "mktCap": 1_095_000_000_000, "beta": 1.4},
        analyst_targets={"targetHigh": 1625, "targetLow": 500,
                          "targetMean": 1100, "targetMedian": 1150,
                          "targetConsensus": 1150},
        analyst_summary={"lastMonth": 12, "lastQuarter": 18},
        pt_news=[
            {"publishedDate": "2026-05-28T10:25:14.000Z",
             "priceTarget": 1500, "company": "DA Davidson",
             "title": "Micron price target raised to $1,500 from $1,000 at DA Davidson"},
            {"publishedDate": "2026-05-26T09:14:00.000Z",
             "priceTarget": 1625, "company": "UBS",
             "title": "Micron Technology (MU) PT Raised to $1,625 at UBS"},
        ],
        grades_history=[
            {"date": "2026-05-28", "gradingCompany": "Mizuho",
             "previousGrade": "Outperform", "newGrade": "Outperform",
             "action": "maintain"},
        ],
        fundamentals={"ttm_fcf": 8_500_000_000, "fcf_yield": 0.0078,
                       "net_debt_to_ebitda": -0.05, "margin_trend": 0.04},
        sector_perf={"sector": "Technology", "cum_return": 0.085, "days": 30},
        macro={"vix": 14.2, "spy_trend": 0.025, "regime": "risk_on"},
        short_data={"short_percent_of_float": 0.018,
                     "days_to_cover": 2.1, "source": "yfinance"},
        iv_data={"iv": 0.62, "dte": 21, "is_liquid": True},
        recent_news=[
            {"date": "2026-05-30", "publisher": "Investopedia",
             "title": "Micron Joined The $1 Trillion Club"},
            {"date": "2026-05-29", "publisher": "Benzinga",
             "title": "Micron's Best Month Since 1985"},
        ],
    )


def test_bundle_includes_all_critical_sections():
    """Smoke: bundle must surface every category the AI prompt needs."""
    b = build_facts_bundle(**_synth_inputs())
    required = (
        "ticker", "spot", "sigma_blended_annual_pct", "sigma_class",
        "rsi_14", "mom_30d_pct", "horizon_trading_days",
        "next_earnings_date", "next_earnings_in_horizon",
        "analyst_consensus", "pt_revisions_90d", "grade_changes_90d",
        "recent_news_30d", "fundamentals", "sector_perf", "macro",
        "short_interest", "options_iv", "peer_tickers", "peer_earnings_in_horizon",
    )
    for k in required:
        assert k in b, f"facts bundle missing required field: {k}"
    assert b["ticker"] == "MU"
    assert b["spot"] == 971.0
    assert b["sigma_class"] == "HIGH"


def test_pt_revisions_carry_real_analyst_names_not_just_numbers():
    """Anti-hallucination guard: bundle's pt_revisions_90d must preserve
    the analyst firm names so AI cites real firms instead of inventing
    'BofA / Citi / HSBC' style placeholders."""
    b = build_facts_bundle(**_synth_inputs())
    revs = b["pt_revisions_90d"]
    assert len(revs) == 2
    firms = {r["firm"] for r in revs}
    assert "DA Davidson" in firms
    assert "UBS" in firms
    # The fabricated names that triggered this defect MUST NOT appear
    # because they're not in the input data.
    assert "BofA" not in firms
    assert "Citi" not in firms
    assert "HSBC" not in firms


def test_pt_revisions_parse_prior_pt_from_title():
    """The parser extracts the prior target from the news title (e.g.
    'raised to $1,500 from $1,000' → prior_pt=1000). Used by AI to
    quantify the magnitude of revision."""
    b = build_facts_bundle(**_synth_inputs())
    da_davidson = next(r for r in b["pt_revisions_90d"] if r["firm"] == "DA Davidson")
    assert da_davidson["prior_pt"] == 1000.0
    assert da_davidson["new_pt"] == 1500.0
    assert da_davidson["action"] == "raise"


def test_recent_news_30d_carries_real_headlines():
    """News must surface, not be silently dropped."""
    b = build_facts_bundle(**_synth_inputs())
    news = b["recent_news_30d"]
    assert len(news) == 2
    titles = [n["title"] for n in news]
    assert any("$1 Trillion Club" in t for t in titles)
    assert any("Best Month Since 1985" in t for t in titles)


def test_missing_inputs_degrade_gracefully():
    """If the caller passes None for a fetched source (FMP fetch failed),
    bundle must omit that section rather than crashing."""
    inputs = _synth_inputs()
    inputs["fundamentals"] = None
    inputs["recent_news"] = None
    b = build_facts_bundle(**inputs)
    # Required headline fields still present
    assert "spot" in b
    assert "pt_revisions_90d" in b
    # Optional sections gracefully omitted
    assert "fundamentals" not in b
    assert "recent_news_30d" not in b


# ---------------------------------------------------------------------------
# 2. Serialization + char-cap
# ---------------------------------------------------------------------------

def test_bundle_to_prompt_block_serializes_to_valid_json():
    import json
    b = build_facts_bundle(**_synth_inputs())
    s = bundle_to_prompt_block(b)
    # Must round-trip cleanly.
    parsed = json.loads(s)
    assert parsed["ticker"] == "MU"


def test_bundle_to_prompt_block_caps_at_max_chars():
    """Inflate the bundle with synthetic lists, verify trim respects max_chars."""
    inputs = _synth_inputs()
    # Inflate pt_revisions and news to force trim path
    inputs["pt_news"] = [
        {"publishedDate": "2026-05-28T10:25:14.000Z",
         "priceTarget": 1500 + i, "company": f"Firm_{i}",
         "title": f"PT raised to ${1500+i} from ${1000+i}"}
        for i in range(80)  # 80 entries — over the cap
    ]
    inputs["recent_news"] = [
        {"date": "2026-05-29", "publisher": "Pub",
         "title": "X" * 150}
        for _ in range(50)
    ]
    b = build_facts_bundle(**inputs)
    s = bundle_to_prompt_block(b, max_chars=3000)
    assert len(s) <= 3000, f"bundle exceeded char cap: {len(s)} > 3000"


# ---------------------------------------------------------------------------
# 3. Pass 1 prompt integration
# ---------------------------------------------------------------------------

class _Snap:
    sector = "Technology"
    rsi = 68.0
    mom_30d = 0.45
    ytd_return = 1.0
    spot = 971.0
    def __init__(self):
        self.timestamp = datetime(2026, 5, 30)


class _Vol:
    blended_sigma = 0.8994


class _Sig:
    name = "historical"; mu_annual = 0.10; confidence = "MEDIUM"


def test_pass1_prompt_includes_bundle_when_provided():
    from src.ai_layer import build_ai_pass1_prompt
    bundle_json = '{"ticker":"MU","pt_revisions_90d":[{"firm":"DA Davidson"}]}'
    prompt = build_ai_pass1_prompt(
        ticker="MU", snapshot=_Snap(), vol_profile=_Vol(),
        horizon_days=20, base_signals=[_Sig()],
        self_earnings_date=datetime(2026, 6, 24),
        peer_tickers=["SNDK", "STX"],
        facts_bundle_json=bundle_json,
    )
    assert "GROUND-TRUTH FACTS BUNDLE" in prompt
    assert bundle_json in prompt
    # The anti-hallucination directive must be there
    assert "DO NOT fabricate" in prompt or "Do NOT fabricate" in prompt


def test_pass1_prompt_omits_bundle_block_when_empty():
    """Backward compat: omitting facts_bundle_json (default "") produces
    a prompt without the bundle section, matching pre-2026-05-30 behavior."""
    from src.ai_layer import build_ai_pass1_prompt
    prompt = build_ai_pass1_prompt(
        ticker="MU", snapshot=_Snap(), vol_profile=_Vol(),
        horizon_days=20, base_signals=[_Sig()],
        self_earnings_date=datetime(2026, 6, 24),
        peer_tickers=["SNDK", "STX"],
        # facts_bundle_json default ""
    )
    assert "GROUND-TRUTH FACTS BUNDLE" not in prompt


# ---------------------------------------------------------------------------
# 4. Pass 2 prompt integration
# ---------------------------------------------------------------------------

def test_pass2_prompt_includes_bundle_when_provided():
    from src.ai_layer import build_ai_pass2_prompt
    from src.engine import AIPassOutput  # use real dataclass for shape compat
    p1 = AIPassOutput(
        pass_number=1, drift_estimate=0.32, drift_range=(-0.10, 0.50),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="strong",
        catalysts=[], bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.04, raw_sources_cited=5,
    )
    bundle_json = '{"ticker":"MU","pt_revisions_90d":[{"firm":"DA Davidson","new_pt":1500}]}'
    prompt = build_ai_pass2_prompt(
        ticker="MU", snapshot=_Snap(), pass1=p1,
        mc_marginal_summary={"p_up": "60%", "p_down": "40%", "bracket_pct_str": "10%"},
        sigma_triangulation_summary={"blended": 0.9, "divergence": 3.5},
        prior_posterior_drift=0.12,
        facts_bundle_json=bundle_json,
    )
    assert "GROUND-TRUTH FACTS BUNDLE" in prompt
    assert "DA Davidson" in prompt
    assert "PASS 2 FACT-CHECK MANDATE" in prompt


def test_pass2_prompt_handles_missing_bundle_gracefully():
    """When bundle omitted, Pass 2 prompt still well-formed; bundle slot
    shows '(empty - bundle was not provided to Pass 1; cannot cross-check)'."""
    from src.ai_layer import build_ai_pass2_prompt
    from src.engine import AIPassOutput
    p1 = AIPassOutput(
        pass_number=1, drift_estimate=0.10, drift_range=(-0.20, 0.40),
        confidence="LOW", vol_regime="MEDIUM", narrative_score="neutral",
        catalysts=[], bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.04, raw_sources_cited=3,
    )
    prompt = build_ai_pass2_prompt(
        ticker="MU", snapshot=_Snap(), pass1=p1,
        mc_marginal_summary={"p_up": "50%", "p_down": "50%", "bracket_pct_str": "10%"},
        sigma_triangulation_summary={"blended": 0.9, "divergence": 3.5},
        prior_posterior_drift=None,
    )
    # When bundle empty, the prompt notes the absence (NOT silent).
    assert "(empty" in prompt

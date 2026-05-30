"""Stage 0 — structured FMP facts bundle for AI prompts.

PROBLEM solved (2026-05-30 AI-overlay audit): the engine fetches rich
structured data from FMP (per-analyst PT revisions, grade changes,
fundamentals, sector perf, etc.) but COLLAPSES each to a single signal
NUMBER before showing it to AI Pass 1 / Pass 2. So the AI has to
web_search to re-discover what FMP already knows, and routinely
hallucinates plausible-looking facts that don't match reality (e.g.
Pass 2's "Multi-Broker PT Upgrade Cluster (BofA, Citi, HSBC, Melius)"
on MU when the real cluster was DA Davidson / Mizuho / Barclays / UBS /
Melius — completely different names).

THIS MODULE builds a structured "facts bundle" from the engine's
already-fetched FMP data — preserving the raw lists and numbers rather
than collapsing them into single signal contributions. The bundle is
then included verbatim in Pass 1 and Pass 2 prompts as ground truth.

Pure function — no I/O. The engine pre-fetches every data source as
part of its normal signal pipeline; this module just reshapes what's
already in memory into AI-prompt-ready form. Only NEW data the engine
didn't previously fetch is `recent_news_30d`, which the engine wires
in via a single fetch_recent_news call (previously dead code).

Token cost: ~1-3 KB per bundle (≈ 300-800 tokens). At Sonnet input
pricing $3/M, ~$0.001-0.0025 per Pass call — negligible vs the
catalyst-fabrication cost it eliminates.

Sacred-decision alignment:
  - #10 "AI outputs are arithmetic inputs, not display prose" — the
    bundle is structured INPUT to the AI, and AI's structured output
    still flows back into the math signals downstream.
  - #11 same-day cache compatible — bundle is deterministic from
    fetched data; cached AI payload still serves same-day replay.
  - CLAUDE.md hard constraint "AI output must never be silently
    dropped" — this module is the upstream mirror: structured FMP
    data must never be silently HIDDEN from the AI that needs it.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional


_NEWS_LOOKBACK_DAYS = 30
_PT_REVISION_LOOKBACK_DAYS = 90
_GRADES_LOOKBACK_DAYS = 90
_MAX_PT_REVISIONS = 30          # cap bundle size on prolific names
_MAX_GRADE_CHANGES = 25
_MAX_NEWS_HEADLINES = 15


def _filter_recent(rows: Optional[list], date_field: str, days: int) -> list:
    """Return rows whose date_field falls in the last `days` calendar days."""
    if not rows:
        return []
    cutoff = (datetime.now() - timedelta(days=days)).date()
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        dstr = (r.get(date_field) or "")[:10]
        try:
            d = datetime.strptime(dstr, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if d >= cutoff:
            out.append(r)
    return out


def _parse_prior_pt_from_title(title: str) -> Optional[float]:
    """Extract the prior PT from titles like 'raised to $1,500 from $1,000'."""
    import re
    if not title:
        return None
    m = re.search(r"from\s*\$?([\d,]+(?:\.\d+)?)", title, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except (ValueError, TypeError):
        return None


def _classify_pt_action(prior: Optional[float], new: Optional[float]) -> str:
    """raise / cut / unchanged / unknown given prior + new PT."""
    if prior is None or new is None:
        return "unknown"
    if new > prior * 1.005:
        return "raise"
    if new < prior * 0.995:
        return "cut"
    return "unchanged"


def build_facts_bundle(
    *,
    ticker: str,
    spot: float,
    sigma_blended: float,
    sigma_class: str,
    rsi: float,
    mom_5d: float,
    mom_30d: float,
    ytd_return: float,
    horizon_days: int,
    peer_tickers: list,
    self_earnings_date: Optional[datetime] = None,
    peer_earnings_dates: Optional[list] = None,
    # Pre-fetched data (caller supplies — engine already has these in memory)
    profile: Optional[dict] = None,
    analyst_targets: Optional[dict] = None,
    analyst_summary: Optional[dict] = None,
    pt_news: Optional[list] = None,
    grades_history: Optional[list] = None,
    fundamentals: Optional[dict] = None,
    sector_perf: Optional[dict] = None,
    macro: Optional[dict] = None,
    short_data: Optional[dict] = None,
    iv_data: Optional[dict] = None,
    recent_news: Optional[list] = None,
) -> dict:
    """Build the structured what-FMP-knows envelope for one ticker.

    PURE FUNCTION — no I/O. Takes pre-fetched data the engine already
    pulled for its signal pipeline, reshapes into AI-prompt-ready form.

    Returns a serializable dict suitable for json.dumps. Fields missing
    from the caller's pre-fetched data appear as None / empty list /
    "(not available)" so AI knows what's absent rather than silently
    dropping the section.
    """
    today = datetime.now().date()
    bundle: dict[str, Any] = {
        "ticker": ticker,
        "as_of": today.isoformat(),
        "spot": round(spot, 4),
        "sigma_blended_annual_pct": round(sigma_blended * 100, 2),
        "sigma_class": sigma_class,
        "rsi_14": round(rsi, 1),
        "mom_5d_pct": round(mom_5d * 100, 2),
        "mom_30d_pct": round(mom_30d * 100, 2),
        "ytd_return_pct": round(ytd_return * 100, 2),
        "horizon_trading_days": horizon_days,
        "peer_tickers": list(peer_tickers or []),
    }

    # --- Company profile ---
    if profile:
        bundle["sector"] = profile.get("sector") or "Unknown"
        bundle["industry"] = profile.get("industry") or "Unknown"
        for cap_key in ("mktCap", "marketCap", "mcap", "market_cap"):
            v = profile.get(cap_key)
            if v:
                try:
                    bundle["market_cap_usd"] = int(float(v))
                    break
                except (ValueError, TypeError):
                    continue
        try:
            bundle["beta"] = (
                float(profile.get("beta")) if profile.get("beta") else None
            )
        except (ValueError, TypeError):
            bundle["beta"] = None

    # --- Self earnings ---
    bundle["next_earnings_date"] = (
        self_earnings_date.strftime("%Y-%m-%d")
        if self_earnings_date else None
    )
    if self_earnings_date is not None:
        try:
            d = (self_earnings_date.date()
                 if hasattr(self_earnings_date, "date")
                 else self_earnings_date)
            from src.market_calendar import trading_days_after as _tda
            td_to_earnings = _tda(today, d)
            bundle["next_earnings_in_horizon"] = (
                0 <= td_to_earnings <= horizon_days
            )
            bundle["next_earnings_trading_days_away"] = td_to_earnings
        except Exception:
            bundle["next_earnings_in_horizon"] = None

    # --- Peer earnings in horizon ---
    bundle["peer_earnings_in_horizon"] = [
        {
            "date": (d.date() if hasattr(d, "date") else d).isoformat(),
        }
        for d in (peer_earnings_dates or [])
    ]

    # --- Analyst consensus + per-revision history ---
    if analyst_targets:
        bundle["analyst_consensus"] = {
            k: analyst_targets.get(k) for k in (
                "targetHigh", "targetLow", "targetMean",
                "targetMedian", "targetConsensus",
            )
        }
    if analyst_summary:
        bundle["analyst_coverage"] = {
            "last_month_analyst_count": analyst_summary.get("lastMonth"),
            "last_quarter_analyst_count": analyst_summary.get("lastQuarter"),
        }

    if pt_news is not None:
        rev = []
        for n in pt_news[:_MAX_PT_REVISIONS]:
            new_pt = n.get("priceTarget")
            try:
                new_pt = float(new_pt) if new_pt is not None else None
            except (ValueError, TypeError):
                new_pt = None
            prior_pt = _parse_prior_pt_from_title(n.get("title", ""))
            rev.append({
                "date": (n.get("publishedDate") or "")[:10],
                "firm": n.get("company") or "(unattributed)",
                "new_pt": new_pt,
                "prior_pt": prior_pt,
                "action": _classify_pt_action(prior_pt, new_pt),
                "title": (n.get("title") or "")[:160],
            })
        bundle["pt_revisions_90d"] = rev
        bundle["pt_revisions_90d_count"] = len(rev)
        if rev:
            raises = sum(1 for r in rev if r["action"] == "raise")
            cuts = sum(1 for r in rev if r["action"] == "cut")
            bundle["pt_revisions_90d_raise_cut_ratio"] = (
                f"{raises} raises / {cuts} cuts"
            )

    # --- Grade changes ---
    if grades_history is not None:
        recent_grades = _filter_recent(
            grades_history, "date", _GRADES_LOOKBACK_DAYS,
        )
        gc = []
        for g in recent_grades[:_MAX_GRADE_CHANGES]:
            gc.append({
                "date": (g.get("date") or "")[:10],
                "firm": g.get("gradingCompany") or "(unattributed)",
                "from_grade": g.get("previousGrade"),
                "to_grade": g.get("newGrade"),
                "action": g.get("action"),
            })
        bundle["grade_changes_90d"] = gc
        if gc:
            bundle["grade_actions_summary"] = {
                "upgrade": sum(1 for x in gc if (x.get("action") or "").lower() == "upgrade"),
                "downgrade": sum(1 for x in gc if (x.get("action") or "").lower() == "downgrade"),
                "maintain": sum(1 for x in gc if (x.get("action") or "").lower() == "maintain"),
                "initiate": sum(1 for x in gc if "init" in (x.get("action") or "").lower()),
            }

    # --- Recent news (was dead code in engine pre-2026-05-30) ---
    if recent_news is not None:
        rn = _filter_recent(recent_news, "date", _NEWS_LOOKBACK_DAYS)
        bundle["recent_news_30d"] = [
            {
                "date": n.get("date") or "",
                "publisher": (n.get("publisher") or "")[:40],
                "title": (n.get("title") or "")[:160],
            }
            for n in rn[:_MAX_NEWS_HEADLINES]
        ]
        bundle["recent_news_30d_count"] = len(bundle["recent_news_30d"])

    # --- Fundamentals (full numbers) ---
    if fundamentals is not None:
        bundle["fundamentals"] = {
            "ttm_fcf_usd": fundamentals.get("ttm_fcf"),
            "fcf_yield_pct": (
                round(fundamentals["fcf_yield"] * 100, 2)
                if fundamentals.get("fcf_yield") is not None else None
            ),
            "net_debt_to_ebitda": fundamentals.get("net_debt_to_ebitda"),
            "operating_margin_trend_pp": (
                round(fundamentals["margin_trend"] * 100, 2)
                if fundamentals.get("margin_trend") is not None else None
            ),
        }

    # --- Sector performance ---
    if sector_perf is not None:
        bundle["sector_perf"] = {
            "sector": sector_perf.get("sector"),
            "cum_return_pct": (
                round(sector_perf["cum_return"] * 100, 2)
                if sector_perf.get("cum_return") is not None else None
            ),
            "lookback_days": sector_perf.get("days"),
        }

    # --- Macro (VIX + SPY) ---
    if macro is not None:
        bundle["macro"] = {
            "vix": macro.get("vix"),
            "spy_trend_pct": (
                round(macro["spy_trend"] * 100, 2)
                if macro.get("spy_trend") is not None else None
            ),
            "regime": macro.get("regime"),
        }

    # --- Short interest ---
    if short_data is not None:
        bundle["short_interest"] = {
            "pct_of_float": (
                round(short_data["short_percent_of_float"] * 100, 2)
                if short_data.get("short_percent_of_float") is not None else None
            ),
            "days_to_cover": short_data.get("days_to_cover"),
            "source": short_data.get("source"),
        }

    # --- Options IV ---
    if iv_data is not None:
        bundle["options_iv"] = {
            "iv_annual_pct": (
                round(iv_data["iv"] * 100, 2)
                if iv_data.get("iv") is not None else None
            ),
            "dte": iv_data.get("dte"),
            "is_liquid": iv_data.get("is_liquid"),
        }

    return bundle


def bundle_to_prompt_block(bundle: dict, max_chars: int = 8000) -> str:
    """Serialize bundle as a compact JSON block for inclusion in an AI prompt.

    Caps total chars (default 8K) by trimming the longest list fields
    (pt_revisions, news, grade_changes) when the bundle is unusually
    large. AI sees a stable shape regardless of ticker activity level.
    """
    import json
    s = json.dumps(bundle, default=str, separators=(",", ":"))
    if len(s) <= max_chars:
        return s
    trimmed = dict(bundle)
    for k, half in (
        ("recent_news_30d", 8),
        ("grade_changes_90d", 12),
        ("pt_revisions_90d", 15),
    ):
        if k in trimmed and isinstance(trimmed[k], list):
            trimmed[k] = trimmed[k][:half]
        s = json.dumps(trimmed, default=str, separators=(",", ":"))
        if len(s) <= max_chars:
            return s
    return s[:max_chars]

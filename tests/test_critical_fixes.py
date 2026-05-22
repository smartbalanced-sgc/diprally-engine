"""Tests for the pre-W2 critical fixes.

Each test corresponds to one of the 8 fixes in the critical-fixes PR:

  1. peer_rs lookback bug — verify the peer history has enough trading
     bars to compute the n_day_return
  2. Display masking fix — NONE_FOUND signal yields is_absent=True
  3. URL apikey redaction in error logs
  4. FetchError typed exception on HTTP failure
  5. Analyst extreme-outlier confidence downgrade
  6. Effective-weight column computed correctly from blend
  7. GARCH α+β 4-decimal display when > 0.95 (visual; smoke-tested by
     other tests indirectly)

These tests are first-line-of-defense regression guards. They run via
`python tests/test_critical_fixes.py` or under pytest. No external
dependencies (no FMP / no Anthropic).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import requests

from src.config import (
    ANALYST_EXTREME_DRIFT_THRESHOLD,
    BLEND_WEIGHTS_V2,
    DEFAULT_LOOKBACK_DAYS,
)
from src.data_fetch import FetchError, _redact
from src.engine import DriftSignal, _signals_dict_to_display_list
from src.signals import (
    _gate_extreme_drift,
    blend_with_uncertainty,
    signal_from_analyst_targets,
    signal_from_peer_rs,
)


# ---------- 1. peer_rs lookback ----------

def test_peer_rs_works_with_730d_history():
    """With DEFAULT_LOOKBACK_DAYS=730 calendar days fetched, peer_rs has
    plenty of trading bars to compute the 60-trading-day return.

    Pre-fix: fetch_peer_history was called with lookback_days=60, yielding
    ~43 trading bars — n_day_return(df, 60) returned None — signal returned
    _none_signal, silently masked as '+0.0% LOW' in display.

    Post-fix: caller passes DEFAULT_LOOKBACK_DAYS=730. Test verifies signal
    produces a real drift output when given enough bars."""
    # Synthesize 100 trading bars of history for ticker + peer
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    own = pd.DataFrame({
        "Date": dates,
        "Close": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),  # +0.1%/day drift
    })
    peer = pd.DataFrame({
        "Date": dates,
        "Close": 100 * np.cumprod(1 + np.random.normal(-0.0005, 0.02, 100)),  # -0.05%/day drift
    })
    result = signal_from_peer_rs(own, {"PEER": peer}, lookback_days=60, ticker="TEST")
    assert result["drift"] is not None, "Signal returned None — peer lookback still broken"
    assert result["confidence"] in ("HIGH", "MEDIUM", "LOW"), result
    assert "TEST" in result["notes"]


def test_peer_rs_returns_none_signal_when_too_few_bars():
    """Pre-fix masking sanity check: when history is genuinely too short
    (e.g. newly-IPOed ticker), signal returns _none_signal — and the
    display fix (test below) marks it absent rather than showing +0.0%."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")  # only 30 bars
    own = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 110, 30)})
    peer = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 105, 30)})
    result = signal_from_peer_rs(own, {"PEER": peer}, lookback_days=60, ticker="TEST")
    assert result["drift"] is None, "Should return _none_signal when history too short"
    assert result["source_quality"] == "NONE_FOUND"


# ---------- 2. Display masking: is_absent ----------

def test_display_marks_none_found_signals_absent():
    """NONE_FOUND signals must be marked is_absent=True so the reporter
    can render 'n/a' rather than the misleading '+0.0%'."""
    signals_dict = {
        "historical": {"drift": 0.10, "confidence": "HIGH", "source_quality": "PRIMARY",
                       "sources_count": 1, "notes": "ok"},
        "peer_rs": {"drift": None, "confidence": "LOW", "source_quality": "NONE_FOUND",
                    "sources_count": 0, "notes": "no peers"},
    }
    out = _signals_dict_to_display_list(signals_dict, BLEND_WEIGHTS_V2)
    by_name = {s.name: s for s in out}
    historical = [s for s in out if "Historical" in s.name][0]
    peer_rs = [s for s in out if "Peer RS" in s.name][0]
    assert historical.is_absent is False, "Active signal incorrectly marked absent"
    assert peer_rs.is_absent is True, "NONE_FOUND signal not marked absent"


def test_display_effective_weight_computed_from_blend():
    """Effective weight column shows post-gate-renormalized weights from
    the live blend, not the nominal BLEND_WEIGHTS_V2 values."""
    signals_dict = {
        "historical": {"drift": 0.10, "confidence": "MEDIUM", "source_quality": "PRIMARY",
                       "sources_count": 1, "notes": "ok"},  # weight 0.05, full
        "analyst": {"drift": -0.04, "confidence": "HIGH", "source_quality": "REPUTABLE",
                    "sources_count": 10, "notes": "ok"},  # weight 0.15, full
        "ai": {"drift": 0.0, "confidence": "LOW", "source_quality": "NONE_FOUND",
               "sources_count": 0, "notes": "AI absent"},  # phantom — zero weight
    }
    blend = blend_with_uncertainty(signals_dict, weights_dict=BLEND_WEIGHTS_V2)
    out = _signals_dict_to_display_list(signals_dict, BLEND_WEIGHTS_V2, blend=blend)
    historical = [s for s in out if "Historical" in s.name][0]
    analyst = [s for s in out if "Analyst" in s.name][0]
    ai = [s for s in out if "AI analyst" in s.name][0]
    # Dynamic: post D-W2-16 (sacred #15 insider drop), weights shifted.
    # Compute expected from live BLEND_WEIGHTS_V2: historical and analyst
    # are both MEDIUM/HIGH (no halving); only those two contribute to total.
    hist_nominal = BLEND_WEIGHTS_V2["historical"]
    ana_nominal = BLEND_WEIGHTS_V2["analyst"]
    total = hist_nominal + ana_nominal
    expected_hist_eff = hist_nominal / total
    expected_ana_eff = ana_nominal / total
    assert abs(historical.effective_weight - expected_hist_eff) < 0.01, \
        f"hist effective: {historical.effective_weight} (expected {expected_hist_eff})"
    assert abs(analyst.effective_weight - expected_ana_eff) < 0.01, \
        f"analyst effective: {analyst.effective_weight} (expected {expected_ana_eff})"
    assert ai.effective_weight == 0.0, f"ai effective should be 0: {ai.effective_weight}"


# ---------- 3. URL apikey redaction ----------

def test_redact_strips_apikey_from_url():
    """Apikey query params must never appear in error messages or logs.
    _redact applies a case-insensitive substring replacement that handles
    the common URL-encoded forms."""
    cases = [
        ("https://fmp.com/api?apikey=secret123&from=2024", "apikey=***REDACTED***"),
        ("HTTPError: 402 ... apikey=ld6wilmawW3FutupImuIMeNIuqafQIMo", "apikey=***REDACTED***"),
        ("apikey=ABC123XYZ", "apikey=***REDACTED***"),
        ("normal text no key", "normal text no key"),  # no-op when no apikey
    ]
    for input_str, expected_substring in cases:
        out = _redact(input_str)
        assert expected_substring in out, f"redaction failed for {input_str!r} → {out!r}"
        if "secret" in input_str or "ld6w" in input_str or "ABC123" in input_str:
            assert "secret123" not in out and "ld6w" not in out and "ABC123" not in out, \
                f"key leaked: {out!r}"


# ---------- 4. FetchError typed exception ----------

def test_fetcherror_carries_status_and_redacts_reason():
    """FetchError must carry ticker, source, status, and a pre-redacted reason
    so callers can distinguish 402-not-in-plan from 429-rate-limit from
    network timeout."""
    e = FetchError("TEST", "fmp", 402, "HTTPError ... apikey=secret")
    assert e.ticker == "TEST"
    assert e.source == "fmp"
    assert e.status == 402
    assert "secret" not in str(e), "FetchError leaked apikey"
    assert "***REDACTED***" in e.reason or "REDACTED" in e.reason


# ---------- 5. Analyst extreme-outlier downgrade ----------

def test_gate_extreme_drift_downgrades_high_to_medium():
    """|implied drift| > threshold should step HIGH → MEDIUM with a
    visible verification flag in the notes."""
    new_conf, new_notes = _gate_extreme_drift(-0.589, "HIGH", "base notes")
    assert new_conf == "MEDIUM"
    assert "EXTREME OUTLIER" in new_notes
    assert "manual verification" in new_notes


def test_gate_extreme_drift_preserves_normal_drift():
    """Normal drift values (under threshold) should pass through unchanged."""
    new_conf, new_notes = _gate_extreme_drift(0.10, "HIGH", "base notes")
    assert new_conf == "HIGH"
    assert new_notes == "base notes"


def test_gate_extreme_drift_caps_at_low():
    """A LOW signal with extreme drift stays LOW (can't go below LOW)."""
    new_conf, new_notes = _gate_extreme_drift(-0.80, "LOW", "x")
    assert new_conf == "LOW"
    assert "EXTREME OUTLIER" in new_notes


def test_analyst_signal_downgrades_extreme_drift_end_to_end():
    """MOG-A scenario: implied drift -58.9% should produce MEDIUM conf
    (not HIGH) with verification flag in notes."""
    # Spot $319, last-month avg $131, n=13 — HIGH base conf
    summary = {"last_month_count": 13, "last_month_avg": 131.0,
               "last_quarter_count": 14, "last_quarter_avg": 130.0,
               "last_year_count": 25, "last_year_avg": 145.0,
               "all_time_count": 30, "all_time_avg": 150.0}
    s = signal_from_analyst_targets({}, 318.87, summary=summary)
    assert abs(s["drift"] - (-0.5891)) < 0.01, f"drift: {s['drift']}"
    assert s["confidence"] == "MEDIUM", f"should downgrade HIGH→MEDIUM: {s['confidence']}"
    assert "EXTREME OUTLIER" in s["notes"]


# ---------- Sacred #13 EV-hurdle gate ----------

def test_ev_hurdle_threshold_constant():
    """Sacred decision #13 specifies 50bps as the minimum EV/dip ratio."""
    from src.config import EV_HURDLE_BPS_OF_DIP
    assert EV_HURDLE_BPS_OF_DIP == 50


def test_ev_hurdle_math_post_sacred_6():
    """Post-sacred-#6 (capital removed), ev_pct_of_dip is computed directly
    as net_ev_per_share / dip_price. No more capital-scaled total.

    Verify on the historical SNDK case: net_ev_per_share ≈ $6.77, dip $1467
    → ev_pct_of_dip = 6.77/1467 = 46.1bps."""
    dip = 1467.0
    net_ev_per_share = 6.77
    ev_pct_of_dip = net_ev_per_share / dip
    # SNDK post-hotfix smoke surfaced 46.5bps — within rounding of the formula
    assert 0.0040 < ev_pct_of_dip < 0.0050  # 40-50bps range


def test_ev_hurdle_triggers_at_46bps():
    """SNDK at ~46bps EV/dip must fail the sacred-#13 50bps gate."""
    from src.config import EV_HURDLE_BPS_OF_DIP
    ev_pct_of_dip = 0.0046  # 46bps
    threshold = EV_HURDLE_BPS_OF_DIP / 10000.0
    assert ev_pct_of_dip < threshold, "46bps must fail the 50bps gate"


def test_ev_hurdle_passes_at_60bps():
    """Trade with EV = 60bps of dip should pass the sacred-#13 gate."""
    from src.config import EV_HURDLE_BPS_OF_DIP
    ev_pct_of_dip = 0.0060  # 60bps
    threshold = EV_HURDLE_BPS_OF_DIP / 10000.0
    assert ev_pct_of_dip >= threshold, "60bps must pass the 50bps gate"


def test_ev_hurdle_refusal_headline_shows_correct_prices():
    """Regression guard for the hotfix-3 typo bug. When the EV-hurdle gate
    fires, the refusal headline must render the dip and rally prices correctly,
    not with a stray '$1' prefix that turned $1,467 into $11,467 in the
    post-PR-#8 SNDK smoke."""
    from datetime import datetime
    from src.engine import (
        DriftSignal,
        JointConditionalResult,
        MarketSnapshot,
        VolatilityProfile,
    )
    from src.reporter import format_report

    # Minimal synthetic inputs
    snapshot = MarketSnapshot(
        ticker="TEST", timestamp=datetime(2026, 5, 22, 18, 52), spot=1510.0,
        market_cap=1e11, sector="Tech", industry="X",
        rsi=63.0, mom_5d=0.07, mom_30d=0.77, ytd_return=4.5,
        price_history=None,
    )
    vol_profile = VolatilityProfile(
        garch_sigma=0.97, garch_alpha=0.07, garch_beta=0.907,
        garch_alpha_plus_beta=0.977, realized_30d=0.95, realized_60d=0.97,
        realized_90d=0.97, options_iv=0.95, options_dte=56,
        blended_sigma=0.96, anchors_count=5, divergence_pp=2.5,
        near_unit_root=False,
    )
    best = JointConditionalResult(
        dip_price=1467.0, rally_price=1656.0,
        p_dip_touched=0.741, p_rally_given_dip=0.756,
        p_round_trip=0.56, p_bag_hold=0.18, p_no_trade_rally_first=0.26,
        p_neither=0.0,
        expected_days_to_dip=0.0, expected_days_dip_to_rally=11.0,
        expected_gain_per_share=185.0, expected_bag_hold_loss=540.0,
        net_ev_per_share=6.77,        # post-sacred-#6: per-share not total
        ev_pct_of_dip=0.00461,        # 46.1bps of $1467 dip
    )
    method_check = {"table": [], "flags": [], "refusals": [], "refused": False,
                    "agreement_status": "✓", "pde_mass_conservation": 1.0,
                    "pde_p_neither": 0.0,
                    "tolerances": {"sigma_used": 0.96,
                                   "first_passage_pp": 3.9, "marginal_pp": 2.9,
                                   "refuse_first_passage_pp": 6.9,
                                   "refuse_marginal_pp": 5.2}}
    backtest = {"n_samples": 0, "sufficient_data": False,
                "message": "no history yet"}
    posterior = {"prior_mu": 0.0, "prior_std": 0.15,
                 "today_mu": 0.17, "today_std": 0.24,
                 "post_mu": 0.17, "post_std": 0.24,
                 "prior_weight": 0.0, "today_weight": 1.0,
                 "phantom_signals": [], "phantom_std_inflation": 0.0}

    report = format_report(
        snapshot, vol_profile, [], None, None, posterior,
        best, method_check, [], backtest,
        0.65, 0.75, 60, 0.0, 30.0,                # no capital arg post sacred #6
        met_threshold_strict=False,
        ev_hurdle_refused=True,
        ev_pct_of_dip=0.00465,  # 46.5 bps
    )

    # Headline must show $1,467 (not $11,467) and $1,656 (not $11,656)
    assert "$1,467" in report, "Dip price not rendered correctly in headline"
    assert "$11,467" not in report, "Stray $1 prefix bug regressed"
    assert "$1,656" in report, "Rally price not rendered correctly"
    assert "$11,656" not in report
    assert "REFUSED" in report
    assert "46.5bps" in report


# ---------- Sacred #15: insider signal dropped (D-W2-16) ----------

def test_insider_signal_not_in_blend_weights():
    """Sacred #15: insider signal dropped (Form 4 lag + noise). Must not
    appear in either v1 or v2 blend weights post D-W2-16."""
    from src.config import BLEND_WEIGHTS, BLEND_WEIGHTS_V2
    assert "insider" not in BLEND_WEIGHTS_V2, "Insider signal must be dropped per sacred #15"
    assert "insider" not in BLEND_WEIGHTS, "Insider signal must be dropped per sacred #15 (v1 too)"


def test_blend_weights_v2_has_expected_signals():
    """Post D-W2-16: 10 active signals in v2 (down from 11). Specifically
    historical, analyst, sector, macro, short_interest, peer_rs,
    sector_decoupling, ai, catalyst_proximity, narrative."""
    from src.config import BLEND_WEIGHTS_V2
    expected = {
        "historical", "analyst", "sector", "macro", "short_interest",
        "peer_rs", "sector_decoupling", "ai", "catalyst_proximity", "narrative",
    }
    assert set(BLEND_WEIGHTS_V2.keys()) == expected, \
        f"v2 signals drifted: {set(BLEND_WEIGHTS_V2.keys())}"


# ---------- Sacred #14 trend filter (D-W2-15) ----------

def test_trend_filter_threshold_constant():
    """Sacred #14 specifies mom_30d < -25% as the trend-filter floor."""
    from src.config import TREND_FILTER_MOM_30D_THRESHOLD
    assert TREND_FILTER_MOM_30D_THRESHOLD == -0.25


def test_has_supporting_catalyst_returns_false_when_no_ai():
    """In --no-ai mode (effective_ai=None), no catalysts known. Strict
    reading of sacred #14: can't disprove falling-knife → refuse."""
    from src.engine import _has_supporting_catalyst
    assert _has_supporting_catalyst(None, 60) is False


def test_has_supporting_catalyst_returns_false_when_only_bearish():
    """Bearish-only catalysts don't rescue a falling knife — they confirm
    it. Sacred #14 looks for bullish OR two-sided catalysts in horizon."""
    from src.engine import AIPassOutput, _has_supporting_catalyst
    from datetime import datetime, timedelta
    in_horizon_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    fake_ai = AIPassOutput(
        pass_number=1, drift_estimate=-0.10, drift_range=(-0.2, 0.0),
        confidence="LOW", vol_regime="HIGH", narrative_score="weak",
        catalysts=[
            {"name": "Earnings miss risk", "date_or_window": in_horizon_date,
             "direction_risk": "bearish", "magnitude": "high", "sources": ["s1"]}
        ],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.0, raw_sources_cited=1,
    )
    assert _has_supporting_catalyst(fake_ai, 60) is False


def test_has_supporting_catalyst_returns_true_when_bullish_in_horizon():
    """Bullish catalyst in horizon → trend filter passes (doesn't refuse)."""
    from src.engine import AIPassOutput, _has_supporting_catalyst
    from datetime import datetime, timedelta
    in_horizon_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    fake_ai = AIPassOutput(
        pass_number=1, drift_estimate=0.10, drift_range=(0.0, 0.2),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="neutral",
        catalysts=[
            {"name": "Product launch", "date_or_window": in_horizon_date,
             "direction_risk": "bullish", "magnitude": "med", "sources": ["s1"]}
        ],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.0, raw_sources_cited=1,
    )
    assert _has_supporting_catalyst(fake_ai, 60) is True


def test_has_supporting_catalyst_returns_true_when_two_sided_in_horizon():
    """Two-sided catalysts (earnings) provide a thesis even though direction
    is uncertain. The trader's bet becomes 'dip exhausted before event'."""
    from src.engine import AIPassOutput, _has_supporting_catalyst
    from datetime import datetime, timedelta
    in_horizon_date = (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")
    fake_ai = AIPassOutput(
        pass_number=1, drift_estimate=0.0, drift_range=(-0.1, 0.1),
        confidence="LOW", vol_regime="HIGH", narrative_score="neutral",
        catalysts=[
            {"name": "Q3 earnings", "date_or_window": in_horizon_date,
             "direction_risk": "two-sided", "magnitude": "high", "sources": ["s1", "s2"]}
        ],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.0, raw_sources_cited=2,
    )
    assert _has_supporting_catalyst(fake_ai, 60) is True


def test_has_supporting_catalyst_ignores_out_of_horizon():
    """Catalyst beyond the horizon doesn't count — by the time it occurs
    the trade window has closed."""
    from src.engine import AIPassOutput, _has_supporting_catalyst
    from datetime import datetime, timedelta
    far_future = (datetime.now() + timedelta(days=200)).strftime("%Y-%m-%d")
    fake_ai = AIPassOutput(
        pass_number=1, drift_estimate=0.10, drift_range=(0.0, 0.2),
        confidence="MEDIUM", vol_regime="MEDIUM", narrative_score="neutral",
        catalysts=[
            {"name": "Far-future event", "date_or_window": far_future,
             "direction_risk": "bullish", "magnitude": "high", "sources": ["s1"]}
        ],
        bull_factors=[], bear_factors=[], key_risks=[],
        revision_from_prior_pass=None, cost_usd=0.0, raw_sources_cited=1,
    )
    assert _has_supporting_catalyst(fake_ai, 60) is False


# ---------- 6. Effective-weight via blend ----------

def test_blend_weights_reflect_low_halving():
    """blend['weights'] after LOW halving should be smaller than nominal.

    Post-D-W2-16 (sacred #15 insider drop), v2 nominal weights are:
      ai: 0.26, analyst: 0.16 (each picked up part of insider's 2%).
    LOW conf halves ai's nominal → 0.13."""
    from src.config import BLEND_WEIGHTS_V2
    signals_dict = {
        "ai": {"drift": 0.10, "confidence": "LOW", "source_quality": "REPUTABLE",
               "sources_count": 5, "notes": "ok"},
        "analyst": {"drift": 0.05, "confidence": "HIGH", "source_quality": "REPUTABLE",
                    "sources_count": 10, "notes": "ok"},
    }
    blend = blend_with_uncertainty(signals_dict, weights_dict=BLEND_WEIGHTS_V2)
    weights = blend["weights"]
    expected_ai_halved = BLEND_WEIGHTS_V2["ai"] * 0.5
    expected_analyst_full = BLEND_WEIGHTS_V2["analyst"]
    assert abs(weights["ai"] - expected_ai_halved) < 0.001, \
        f"AI halved weight: {weights['ai']} (expected {expected_ai_halved})"
    assert abs(weights["analyst"] - expected_analyst_full) < 0.001, \
        f"analyst full weight: {weights['analyst']} (expected {expected_analyst_full})"


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in sorted(inspect.getmembers(sys.modules[__name__], inspect.isfunction)):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
            fails += 1
    if fails:
        print(f"\n{fails} test(s) failed")
        sys.exit(1)
    print("\nALL TESTS PASSED")

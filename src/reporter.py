"""Text report + per-ticker HTML dashboard.

W0 keeps the v2 layout. W5 adds the multi-ticker aggregate index.html
ranked-dashboard generator.
"""
from __future__ import annotations

import base64
import io

import numpy as np

from src.config import CATALYST_Z_THRESHOLD, V2_VERSION, BACKTEST_MIN_SAMPLES
from src.signals import _factor_weight


def hr(title: str = "") -> str:
    line = "=" * 78
    return f"\n{line}\n{title}\n{line}" if title else line


def format_report(
    snapshot,
    vol_profile,
    base_signals,
    pass1,
    pass2,
    posterior,
    best,
    method_check,
    catalyst_stress,
    backtest,
    conviction_dip,
    conviction_rally_cond,
    horizon_days,
    capital_usd,
    total_ai_cost,
    runtime_seconds,
    met_threshold_strict=True,
    unusual_move=None,
    sensitivity=None,
    path_metrics=None,
) -> str:
    lines: list[str] = []
    lines.append(hr(f"DIPRALLY ENGINE ({V2_VERSION}) — {snapshot.ticker} — {snapshot.timestamp:%Y-%m-%d %H:%M}"))
    lines.append(f"  Ticker: {snapshot.ticker}")
    lines.append(f"  Spot: ${snapshot.spot:.2f}   Market cap: ${snapshot.market_cap/1e9:.1f}B")
    lines.append(f"  Sector / Industry: {snapshot.sector} / {snapshot.industry}")
    lines.append(f"  RSI: {snapshot.rsi:.1f}   5d mom: {snapshot.mom_5d:+.1%}   30d mom: {snapshot.mom_30d:+.1%}   YTD: {snapshot.ytd_return:+.1%}")
    lines.append(f"  Conviction thresholds: dip {conviction_dip:.0%} marginal, rally-cond {conviction_rally_cond:.0%}")
    lines.append(f"  Horizon: {horizon_days} trading days   Capital: ${capital_usd:,.0f}")

    # HEADLINE RECOMMENDATION
    lines.append(hr("ROUND-TRIP RECOMMENDATION"))
    if best is None:
        lines.append("  No dip/rally pair meets the conviction thresholds at current spot/vol/drift.")
        lines.append("  Action: WAIT — re-run after next close.")
    else:
        if not met_threshold_strict:
            lines.append("  ⚠ BELOW THRESHOLD — no pair met dip ≥{:.0%} AND rally-cond ≥{:.0%}.".format(
                conviction_dip, conviction_rally_cond))
            lines.append("  ⚠ Showing best-by-EV fallback. DO NOT TRADE this pair without re-evaluating.")
            lines.append("  ⚠ Action: WAIT for a higher-conviction setup OR adjust thresholds with --conviction-dip / --conviction-rally-cond.")
            lines.append("")
        if best.net_expected_value < 0 and met_threshold_strict:
            lines.append("  ⚠ NEGATIVE EXPECTED VALUE — thresholds met BUT average outcome loses money.")
            lines.append(f"  ⚠ Bag-hold scenario (P={best.p_bag_hold:.0%}, ${best.expected_bag_hold_loss:,.0f}/share loss) dominates the gain.")
            lines.append("  ⚠ Consider waiting for a higher-EV setup or skipping this trade.")
            lines.append("")
        shares = capital_usd / best.dip_price
        lines.append(f"  Dip buy-limit:    ${best.dip_price:,.0f}  (P(touch within {horizon_days}d) = {best.p_dip_touched:.1%}, expected day {best.expected_days_to_dip:.0f})")
        lines.append(f"  Rally sell-limit: ${best.rally_price:,.0f}  (P(rally | dip touched) = {best.p_rally_given_dip:.1%}, expected day +{best.expected_days_dip_to_rally:.0f})")
        lines.append(f"  Joint P(round-trip): {best.p_round_trip:.1%}")
        lines.append(f"  Expected gain/share if completed: +${best.expected_gain_per_share:,.0f}")
        lines.append(f"  Expected $ loss if bag-hold: ${best.expected_bag_hold_loss:,.0f}/share at horizon")
        lines.append(f"  Net expected $/trade: ${best.net_expected_value:,.0f}  (capital ${capital_usd:,.0f} → ~{shares:.1f} shares)")

        lines.append(hr("SCENARIO BREAKDOWN (sum to 100%)"))
        lines.append(f"  A. Round-trip completed:     {best.p_round_trip:6.1%}  → profit")
        lines.append(f"  B. Bag-hold at horizon:      {best.p_bag_hold:6.1%}  → paper loss")
        lines.append(f"  C. Rally-first, no entry:    {best.p_no_trade_rally_first:6.1%}  → missed trade, no P&L")
        lines.append(f"  D. Neither touched:          {best.p_neither:6.1%}  → no trade, no P&L")

    # THREE-METHOD CROSS-CHECK
    lines.append(hr("THREE-METHOD MATH CROSS-CHECK (MC / PDE / closed-form)"))
    lines.append(f"  {method_check['agreement_status']}")
    lines.append(f"  {'Quantity':<25} {'MC':>10} {'PDE':>10} {'Δ pp':>8}")
    for q, mc, pde, delta in method_check["table"]:
        lines.append(f"  {q:<25} {mc:>9.1f}% {pde:>9.1f}% {delta:>7.2f}")
    if method_check["flags"]:
        for flag in method_check["flags"]:
            lines.append(f"  ⚠ {flag}")
    lines.append(f"  PDE mass conservation: {method_check['pde_mass_conservation']:.5f} (should be ~1.0)")

    # UNUSUAL MOVE Z-SCORE
    if unusual_move:
        lines.append(hr("UNUSUAL MOVE DETECTION (beta-adjusted Z-score)"))
        z = unusual_move["z_score"]
        ret_pct = unusual_move["return_pct"]
        beta = unusual_move["beta"]
        trigger = unusual_move["triggered"]
        flag_str = "  ⚠ TRIGGERED — investigate possible hidden catalyst" if trigger else "  ✓ within normal range"
        lines.append(f"  Today's return: {ret_pct:+.2f}%  |  beta: {beta:.2f}  |  Z (β-adj): {z:.2f}")
        lines.append(f"  Threshold: |Z| ≥ {CATALYST_Z_THRESHOLD:.1f} for high-vol regime")
        lines.append(flag_str)
        if trigger:
            lines.append("  (Pattern from src/sentiment.py — abnormal moves often precede / signal catalysts)")

    # SIGMA TRIANGULATION
    lines.append(hr(f"SIGMA TRIANGULATION ({vol_profile.anchors_count} anchors)"))
    if vol_profile.garch_alpha_plus_beta > 0:
        alpha_beta_str = (
            f"α={vol_profile.garch_alpha:.3f}, β={vol_profile.garch_beta:.3f}, "
            f"α+β={vol_profile.garch_alpha_plus_beta:.3f}"
        )
    else:
        alpha_beta_str = "α+β fit failed"
    lines.append(f"  GARCH spot:       {vol_profile.garch_sigma:.1%}  ({alpha_beta_str})")
    lines.append(f"  Realized 30d:     {vol_profile.realized_30d:.1%}")
    lines.append(f"  Realized 60d:     {vol_profile.realized_60d:.1%}")
    lines.append(f"  Realized 90d:     {vol_profile.realized_90d:.1%}")
    iv_str = f"{vol_profile.options_iv:.1%} (DTE {vol_profile.options_dte})" if vol_profile.options_iv else "n/a"
    lines.append(f"  Options IV:       {iv_str}")
    lines.append(f"  BLENDED:          {vol_profile.blended_sigma:.1%}   Divergence: {vol_profile.divergence_pp:.1f}pp")
    if vol_profile.near_unit_root:
        lines.append(f"  ⚠ GARCH α+β > 0.98 — near-IGARCH, vol shocks highly persistent")
    elif 0.95 < vol_profile.garch_alpha_plus_beta <= 0.98:
        lines.append(f"  ⚠ GARCH α+β > 0.95 — high vol persistence, multi-step forecasts unreliable")

    # 11-SIGNAL DRIFT BLEND
    lines.append(hr(f"DRIFT INTELLIGENCE ({len(base_signals)} signals)"))
    lines.append(f"  {'Signal':<35} {'mu (ann)':>10} {'Conf':>8} {'Weight':>8}")
    for s in base_signals:
        lines.append(f"  {s.name:<35} {s.mu_annual:>+9.1%} {s.confidence:>8} {s.weight:>7.0%}")

    # BAYESIAN POSTERIOR
    lines.append(hr("BAYESIAN BELIEF UPDATE"))
    lines.append(f"  Prior posterior (from CSV): mu={posterior.get('prior_mu', 0):+.1%}/yr, std={posterior.get('prior_std', 0.15)*100:.1f}pp")
    lines.append(f"  Today's blend:              mu={posterior.get('today_mu', 0):+.1%}/yr, std={posterior.get('today_std', 0.20)*100:.1f}pp")
    lines.append(f"  Posterior (used in MC):     mu={posterior.get('post_mu', 0):+.1%}/yr, std={posterior.get('post_std', 0.10)*100:.1f}pp")
    lines.append(f"  Prior weight: {posterior.get('prior_weight', 0):.0%}, today weight: {posterior.get('today_weight', 0):.0%}")

    # SENSITIVITY TABLE
    if sensitivity and best:
        lines.append(hr("SENSITIVITY at recommended pair"))
        lines.append(f"  {'Scenario':<35} {'μ':>7} {'σ':>7} {'P(RT)':>7} {'P(BH)':>7} {'Net EV/sh':>11}")
        for row in sensitivity:
            lines.append(
                f"  {row['label']:<35} "
                f"{row['mu']*100:>+6.0f}% "
                f"{row['sigma']*100:>6.0f}% "
                f"{row['p_round_trip']*100:>6.0f}% "
                f"{row['p_bag_hold']*100:>6.0f}% "
                f"{'$' + format(int(round(row['net_ev_per_share'])), '+,d'):>11}"
            )
        lines.append("  (P(RT)=round-trip, P(BH)=bag-hold; Net EV in $/share at recommended pair)")

    # PATH METRICS
    if path_metrics:
        lines.append(hr("PATH-DEPENDENT RISK METRICS"))
        lines.append(f"  Max drawdown from spot ${snapshot.spot:,.0f}:")
        lines.append(f"    median: {path_metrics['max_dd_p50']*100:5.1f}% (${path_metrics['max_dd_price_p50']:,.0f} touched)")
        lines.append(f"    p75:    {path_metrics['max_dd_p75']*100:5.1f}% (${path_metrics['max_dd_price_p75']:,.0f} touched)")
        lines.append(f"    p90:    {path_metrics['max_dd_p90']*100:5.1f}% (${path_metrics['max_dd_price_p90']:,.0f} touched)")
        lines.append(f"  Panic floor ${path_metrics['panic_floor_price']:,.0f} (30% below spot) touched: P = {path_metrics['p_panic_touched']*100:.0f}%")
        if path_metrics.get("time_to_dip_p50") is not None:
            lines.append(
                f"  Time-to-dip (paths that touched): median {path_metrics['time_to_dip_p50']:.0f}d, "
                f"p25/p75 {path_metrics['time_to_dip_p25']:.0f}d/{path_metrics['time_to_dip_p75']:.0f}d"
            )
        if path_metrics.get("time_to_rally_p50") is not None:
            lines.append(
                f"  Time-to-rally (paths that touched): median {path_metrics['time_to_rally_p50']:.0f}d, "
                f"p25/p75 {path_metrics['time_to_rally_p25']:.0f}d/{path_metrics['time_to_rally_p75']:.0f}d"
            )

    # AI SYNTHESIS
    lines.append(hr("AI TWO-PASS SYNTHESIS (Claude Opus 4.7)"))
    if pass1:
        lines.append(f"  PASS 1: drift={pass1.drift_estimate:+.1%}/yr  conf={pass1.confidence}  vol_regime={pass1.vol_regime}  narrative={pass1.narrative_score}  sources={pass1.raw_sources_cited}  cost=${pass1.cost_usd:.2f}")
        lines.append(f"    Catalysts identified: {len(pass1.catalysts)}")
        for c in pass1.catalysts[:5]:
            if isinstance(c, dict):
                lines.append(f"      • {c.get('name','?')} ({c.get('date_or_window','?')}, {c.get('direction_risk','?')}, magnitude {c.get('magnitude','?')})")
            else:
                lines.append(f"      • {c}")
        lines.append(f"    Bull factors HIGH-weight: {sum(1 for f in pass1.bull_factors if _factor_weight(f) == 'high')}")
        lines.append(f"    Bear factors HIGH-weight: {sum(1 for f in pass1.bear_factors if _factor_weight(f) == 'high')}")
    else:
        lines.append("  PASS 1: failed or skipped")
    if pass2:
        rev = pass2.revision_from_prior_pass
        rev_str = f"({rev:+.1%} from Pass 1)" if rev is not None else ""
        lines.append(f"  PASS 2: drift={pass2.drift_estimate:+.1%}/yr  conf={pass2.confidence}  {rev_str}  cost=${pass2.cost_usd:.2f}")
        if pass2.key_risks:
            for risk in pass2.key_risks[:3]:
                lines.append(f"    → {risk}")
    else:
        lines.append("  PASS 2: failed or skipped")

    # CATALYST STRESS TEST
    if catalyst_stress:
        lines.append(hr("CATALYST IMPACT STRESS TEST (top 3, on 20% disappointment)"))
        for c in catalyst_stress[:3]:
            lines.append(f"  {c.get('catalyst_name','?'):<40} drift shock: {c.get('drift_shock_pp_on_disappointment', 0):+.1f}pp")

    # BACKTEST LAYER
    lines.append(hr("BACKTESTING — model performance to date"))
    lines.append(f"  N days tracked: {backtest['n_samples']} (need ≥{BACKTEST_MIN_SAMPLES} for statistical validity)")
    if not backtest["sufficient_data"]:
        lines.append(f"  Status: {backtest.get('message', 'insufficient_data')}")
        lines.append("  Calibration metrics: insufficient data")
    else:
        lines.append(f"  Dip predictions resolved: {backtest.get('dip_predictions_resolved', 0)}")
        lines.append(f"  Rally predictions resolved: {backtest.get('rally_predictions_resolved', 0)}")

    if backtest.get("per_day_status"):
        lines.append(f"\n  Recent prior predictions:")
        for s in backtest["per_day_status"][-7:]:
            lines.append(f"    {s['date']}  dip ${s['dip_target']:,.0f} / rally ${s['rally_target']:,.0f}  "
                        f"P(RT)={s['p_round_trip']:.0%}  elapsed {s['days_elapsed']}d, remaining {s['remaining']}d  [{s['status']}]")

    # RELIABILITY COMPONENTS
    lines.append(hr("RELIABILITY COMPONENTS (assess each independently)"))
    lines.append(f"  Math methods agreement: {method_check['agreement_status']}")
    lines.append(f"  σ anchors: {vol_profile.anchors_count}/5 (divergence {vol_profile.divergence_pp:.1f}pp)")
    if vol_profile.garch_alpha_plus_beta > 0:
        ab_label = (
            "(NEAR UNIT-ROOT)" if vol_profile.near_unit_root
            else "(high persistence)" if vol_profile.garch_alpha_plus_beta > 0.95
            else "(stable)"
        )
        lines.append(f"  GARCH α+β: {vol_profile.garch_alpha_plus_beta:.3f} {ab_label}")
    else:
        lines.append(f"  GARCH α+β: fit failed")
    lines.append(f"  Drift signals active: {sum(1 for s in base_signals if s.confidence != 'LOW')}/{len(base_signals)} non-LOW")
    if pass1 and pass2:
        lines.append(f"  AI Pass1→Pass2 revision: {pass2.revision_from_prior_pass:+.1%} drift" if pass2.revision_from_prior_pass is not None else "  AI Pass1→Pass2 revision: n/a")

    # FOOTER
    lines.append(hr())
    lines.append(f"  Runtime: {runtime_seconds:.1f}s  |  AI cost this run: ${total_ai_cost:.2f}")
    lines.append(f"  History: output/round_trip_history_{snapshot.ticker}.csv")
    lines.append(f"  Dashboard: output/{snapshot.ticker.lower()}_dipnrally_dashboard.html")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# HTML dashboard
# =============================================================================

def generate_html_dashboard(
    output_path,
    snapshot,
    best,
    vol_profile,
    base_signals,
    pass1,
    pass2,
    method_check,
    backtest,
    history_rows,
    conviction_dip,
    conviction_rally_cond,
    horizon_days,
):
    """Single-file HTML dashboard. Clean CSS, no JS, embedded matplotlib PNGs."""
    chart_signals_png = _make_signal_contribution_chart(base_signals)
    chart_history_png = _make_history_trajectory_chart(history_rows, snapshot.spot, best)
    chart_method_png = _make_method_agreement_chart(method_check)

    best_block = ""
    if best:
        best_block = f"""
    <div class="headline">
      <div class="big">Round-trip recommendation</div>
      <div class="pair">
        <div class="leg"><div class="lbl">Dip buy-limit</div><div class="val">${best.dip_price:,.0f}</div><div class="sub">{best.p_dip_touched:.0%} touch / ~day {best.expected_days_to_dip:.0f}</div></div>
        <div class="arrow">→</div>
        <div class="leg"><div class="lbl">Rally sell-limit</div><div class="val">${best.rally_price:,.0f}</div><div class="sub">{best.p_rally_given_dip:.0%} cond / +{best.expected_days_dip_to_rally:.0f}d</div></div>
      </div>
      <div class="metrics">
        <div><span class="m-lbl">Joint P</span><span class="m-val">{best.p_round_trip:.0%}</span></div>
        <div><span class="m-lbl">Gain/sh</span><span class="m-val">+${best.expected_gain_per_share:,.0f}</span></div>
        <div><span class="m-lbl">Bag-hold P</span><span class="m-val">{best.p_bag_hold:.0%}</span></div>
        <div><span class="m-lbl">Net EV</span><span class="m-val">${best.net_expected_value:,.0f}</span></div>
      </div>
    </div>
"""
    else:
        best_block = """
    <div class="headline none">
      <div class="big">No pair meets conviction thresholds</div>
      <div class="sub">Re-run after next close.</div>
    </div>
"""

    signal_rows = "\n".join(
        f"      <tr><td>{s.name}</td><td>{s.mu_annual:+.1%}</td><td>{s.confidence}</td><td>{s.weight:.0%}</td></tr>"
        for s in base_signals
    )
    method_rows = "\n".join(
        f"      <tr><td>{q}</td><td>{mc:.1f}%</td><td>{pde:.1f}%</td><td>{delta:.2f}pp</td></tr>"
        for q, mc, pde, delta in method_check["table"]
    )
    ai_block = ""
    if pass1 and pass2:
        ai_block = f"""
    <div class="ai-block">
      <div class="ai-pass"><strong>Pass 1:</strong> drift {pass1.drift_estimate:+.1%}/yr, conf {pass1.confidence}, narrative {pass1.narrative_score}, {len(pass1.catalysts)} catalysts, ${pass1.cost_usd:.2f}</div>
      <div class="ai-pass"><strong>Pass 2:</strong> revised drift {pass2.drift_estimate:+.1%}/yr ({(pass2.revision_from_prior_pass or 0):+.1%} from Pass 1), conf {pass2.confidence}, ${pass2.cost_usd:.2f}</div>
    </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DipRally Engine — {snapshot.ticker} — {snapshot.timestamp:%Y-%m-%d}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; background: #0f1115; color: #e5e7eb; line-height: 1.5; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  h1 {{ color: #fff; font-size: 22px; margin: 0 0 4px 0; }}
  h2 {{ color: #d1d5db; font-size: 15px; text-transform: uppercase; letter-spacing: 1px;
        margin: 32px 0 12px 0; border-bottom: 1px solid #2d3138; padding-bottom: 8px; }}
  .meta {{ color: #9ca3af; font-size: 13px; margin-bottom: 16px; }}
  .headline {{ background: #1a1d24; border: 1px solid #2d3138; border-radius: 8px;
              padding: 24px; margin: 20px 0; }}
  .headline.none {{ background: #1f1a1a; border-color: #4b3030; }}
  .big {{ font-size: 14px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; }}
  .pair {{ display: flex; align-items: center; justify-content: space-around; margin: 16px 0; }}
  .leg {{ text-align: center; }}
  .leg .lbl {{ font-size: 12px; color: #9ca3af; }}
  .leg .val {{ font-size: 36px; font-weight: 600; color: #fff; margin: 4px 0; }}
  .leg .sub {{ font-size: 12px; color: #6b7280; }}
  .arrow {{ font-size: 28px; color: #6b7280; }}
  .metrics {{ display: flex; justify-content: space-around; margin-top: 20px; padding-top: 16px; border-top: 1px solid #2d3138; }}
  .metrics div {{ text-align: center; }}
  .metrics .m-lbl {{ display: block; font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metrics .m-val {{ display: block; font-size: 18px; font-weight: 500; color: #e5e7eb; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #2d3138; }}
  th {{ color: #9ca3af; font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
  td {{ color: #e5e7eb; }}
  .chart {{ margin: 16px 0; background: #fff; border-radius: 6px; padding: 8px; }}
  .chart img {{ width: 100%; display: block; }}
  .ai-block {{ background: #1a1d24; border-left: 3px solid #3b82f6; padding: 12px 16px; margin: 12px 0; }}
  .ai-pass {{ font-size: 13px; margin: 4px 0; }}
  .footer {{ color: #6b7280; font-size: 11px; margin-top: 32px; text-align: center; }}
  .flag {{ color: #f59e0b; }}
</style>
</head>
<body>
<div class="container">
  <h1>DipRally Engine — {snapshot.ticker}</h1>
  <div class="meta">{snapshot.ticker} @ ${snapshot.spot:.2f} · {snapshot.timestamp:%Y-%m-%d %H:%M}
    · σ {vol_profile.blended_sigma:.0%} · YTD {snapshot.ytd_return:+.0%}
    · thresholds {conviction_dip:.0%}/{conviction_rally_cond:.0%}
    · horizon {horizon_days}d</div>
  {best_block}

  <h2>11-Day Trajectory</h2>
  <div class="chart"><img src="data:image/png;base64,{chart_history_png}"></div>

  <h2>Drift Signal Contributions</h2>
  <div class="chart"><img src="data:image/png;base64,{chart_signals_png}"></div>
  <table>
    <thead><tr><th>Signal</th><th>μ (ann)</th><th>Confidence</th><th>Weight</th></tr></thead>
    <tbody>
{signal_rows}
    </tbody>
  </table>

  <h2>Three-Method Math Cross-Check</h2>
  <div class="chart"><img src="data:image/png;base64,{chart_method_png}"></div>
  <table>
    <thead><tr><th>Quantity</th><th>MC</th><th>PDE</th><th>Δ</th></tr></thead>
    <tbody>
{method_rows}
    </tbody>
  </table>
  {"".join(f'<div class="flag">⚠ {f}</div>' for f in method_check["flags"])}

  <h2>AI Two-Pass Synthesis</h2>
  {ai_block}

  <div class="footer">
    DipRally Engine ({V2_VERSION}) · not for production trading without risk management
  </div>
</div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def _matplotlib_to_b64(fig) -> str:
    """Render figure to base64 PNG string."""
    try:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def _make_signal_contribution_chart(base_signals) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [s.name for s in base_signals]
        contributions = [s.mu_annual * s.weight for s in base_signals]
        colors = ["#10b981" if c >= 0 else "#ef4444" for c in contributions]
        fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.4)))
        y = np.arange(len(names))
        ax.barh(y, contributions, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Contribution to drift (weighted μ, annualised)")
        ax.axvline(0, color="#333", linewidth=0.5)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.set_title("Drift signal weighted contributions", fontsize=11)
        return _matplotlib_to_b64(fig)
    except Exception:
        return ""


def _make_history_trajectory_chart(history_rows, spot_now, best) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not history_rows:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No history yet — runs accumulate here",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12, color="#666")
            ax.set_xticks([])
            ax.set_yticks([])
            return _matplotlib_to_b64(fig)

        def _safe_float(v):
            try:
                return float(v) if v not in (None, "") else 0.0
            except (TypeError, ValueError):
                return 0.0

        dates = [r.get("date", "") for r in history_rows]
        spots = [_safe_float(r.get("spot", 0)) for r in history_rows]
        dips = [_safe_float(r.get("recommended_dip", 0)) for r in history_rows]
        rallies = [_safe_float(r.get("recommended_rally", 0)) for r in history_rows]
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(dates))
        ax.plot(x, spots, color="#3b82f6", label="Spot", linewidth=2)
        ax.plot(x, dips, color="#ef4444", label="Dip target", linewidth=1.5, linestyle="--")
        ax.plot(x, rallies, color="#10b981", label="Rally target", linewidth=1.5, linestyle="--")
        ax.fill_between(x, dips, rallies, alpha=0.08, color="#9ca3af")
        ax.set_xticks(x[::max(1, len(x)//10)])
        ax.set_xticklabels([d[5:] for d in dates[::max(1, len(x)//10)]], rotation=45, fontsize=8)
        ax.set_ylabel("Price ($)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(linestyle=":", alpha=0.4)
        ax.set_title("Spot, dip, rally trajectory", fontsize=11)
        return _matplotlib_to_b64(fig)
    except Exception:
        return ""


def _make_method_agreement_chart(method_check) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        table = method_check["table"]
        labels = [t[0] for t in table]
        mc_vals = [t[1] for t in table]
        pde_vals = [t[2] for t in table]
        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - w/2, mc_vals, w, label="MC", color="#3b82f6")
        ax.bar(x + w/2, pde_vals, w, label="PDE", color="#8b5cf6")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Probability (%)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_title("MC vs PDE first-passage agreement", fontsize=11)
        return _matplotlib_to_b64(fig)
    except Exception:
        return ""

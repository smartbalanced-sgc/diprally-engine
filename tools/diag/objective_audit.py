"""Objective-function audit — does EV-rank pick the same (dip, rally) pair
that P(round-trip) rank picks, and how much do they diverge?

Mission per CLAUDE.md: "lock in gains from expected short-horizon RALLIES
via dip-and-rally round-trips inside 20 trading days." That language is
hit-rate / round-trip-completion language. The engine currently ranks the
grid by EV (expected payoff per share, averaged over all MC paths). EV is
the mission-stated FLOOR ("refuses negative-EV setups") but not its stated
selection metric.

This harness instantiates 3 synth setups across the σ-class spectrum,
scans the full (dip, rally) grid each engine uses, and tabulates each pair's:

  EV_bps        — current selection metric (max over wait/direct)
  P_round_trip  — strategy-specific: P(dip filled AND rally before stop)
  hit_rate      — strategy-specific: P(profitable round-trip exit)
  payoff_std    — path-level σ of payoff (risk lens)
  sharpe        — EV / payoff_std (risk-adjusted)
  pick_strategy — WAIT-FOR-DIP or DIRECT

Then ranks each pair by EV vs by P_round_trip vs by Sharpe and reports
the divergence: do these three metrics agree on the BEST pair, or pick
materially different setups?

If divergence is large → the engine has been optimizing the wrong
quantity, and Milestone B should be a metric swap, not a momentum addon.
If divergence is small → EV is a good proxy for round-trip probability
on these regimes; momentum addon is the right direction.

No engine changes — pure diagnostic.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.config import (
    SIGMA_CLASSES, PATIENCE_WINDOW_TD, MIN_DIP_PROBABILITY,
)
from src.math_utils import (
    run_mc_joint_conditional, precompute_first_touch_days,
)

N_PATHS = 60_000
HORIZON = 20


def per_path_payoffs(
    paths, S0, dip_price, rally_price, fric_wait, fric_direct,
    dip_first, rally_first, patience_window, swing_stop_pct,
):
    """Mirror of compute_dual_ev but returns per-path payoff arrays for
    BOTH branches so the harness can compute distribution statistics
    (std, percentiles, hit-rate decomposition)."""
    n_paths, n_days = paths.shape
    days_idx = np.arange(n_days)[None, :]
    use_stop = swing_stop_pct is not None and swing_stop_pct > 0.0
    eff_patience = patience_window if patience_window is not None else n_days

    # DIRECT branch — entry at S0 day 0, exit at min(rally_first, stop_first, window_end)
    direct_window = days_idx <= eff_patience
    direct_exit_idx = min(eff_patience, n_days - 1)
    direct_exit_price = paths[:, direct_exit_idx]
    direct_rally_mask = (paths >= rally_price) & direct_window
    direct_rally_hit = direct_rally_mask.any(axis=1)
    direct_rally_first = np.where(
        direct_rally_hit, direct_rally_mask.argmax(axis=1), n_days,
    )
    if use_stop:
        stop_lvl_direct = S0 * (1.0 - swing_stop_pct)
        direct_stop_mask = (paths <= stop_lvl_direct) & direct_window
        direct_stop_hit = direct_stop_mask.any(axis=1)
        direct_stop_first = np.where(
            direct_stop_hit, direct_stop_mask.argmax(axis=1), n_days,
        )
        stopped_first_direct = direct_stop_hit & (
            direct_stop_first < direct_rally_first
        )
    else:
        stop_lvl_direct = 0.0
        stopped_first_direct = np.zeros(n_paths, dtype=bool)

    payoff_direct = np.where(
        stopped_first_direct,
        stop_lvl_direct - S0 - fric_direct,
        np.where(
            direct_rally_hit,
            rally_price - S0 - fric_direct,
            direct_exit_price - S0 - fric_direct,
        ),
    )

    # WAIT branch — entry at dip price ONLY if dip touched, then same logic
    # from dip-touch day forward
    dip_any = dip_first < n_days
    after_dip = days_idx >= dip_first[:, None]
    within_window = after_dip & (
        days_idx <= (dip_first[:, None] + eff_patience)
    )
    wait_exit_idx = np.minimum(
        dip_first + eff_patience, n_days - 1
    ).astype(int)
    wait_exit_price = paths[np.arange(n_paths), wait_exit_idx]
    wait_rally_mask = (paths >= rally_price) & within_window
    wait_rally_hit = wait_rally_mask.any(axis=1)
    wait_rally_first = np.where(
        wait_rally_hit, wait_rally_mask.argmax(axis=1), n_days,
    )
    if use_stop:
        stop_lvl_wait = dip_price * (1.0 - swing_stop_pct)
        wait_stop_mask = (paths <= stop_lvl_wait) & within_window
        wait_stop_hit = wait_stop_mask.any(axis=1)
        wait_stop_first = np.where(
            wait_stop_hit, wait_stop_mask.argmax(axis=1), n_days,
        )
        stopped_first_wait = wait_stop_hit & (wait_stop_first < wait_rally_first)
        stop_payoff_wait = stop_lvl_wait - dip_price - fric_wait
    else:
        stopped_first_wait = np.zeros(n_paths, dtype=bool)
        stop_payoff_wait = 0.0

    rt_payoff = rally_price - dip_price - fric_wait
    bag_payoff = wait_exit_price - dip_price - fric_wait

    payoff_wait = np.where(
        ~dip_any,
        0.0,
        np.where(
            stopped_first_wait,
            stop_payoff_wait,
            np.where(wait_rally_hit, rt_payoff, bag_payoff),
        ),
    )

    return {
        "payoff_direct": payoff_direct,
        "payoff_wait": payoff_wait,
        "direct_rally_hit": direct_rally_hit,
        "stopped_first_direct": stopped_first_direct,
        "dip_any": dip_any,
        "wait_rally_hit": wait_rally_hit & dip_any,
        "stopped_first_wait": stopped_first_wait,
    }


def scan_setup(name, sigma_class, S0, sigma, mu):
    cls = SIGMA_CLASSES[sigma_class]
    g = cls.grid
    fric_bps = cls.friction_bps_round_trip
    stop_pct = getattr(cls, "swing_stop_pct", None)

    dip_step = S0 * g.dip_step_pct
    rally_step = S0 * g.rally_step_pct
    dip_min = S0 * (1.0 - g.dip_max_depth_pct)
    dip_max = S0 * 0.99
    rally_min = S0 * 1.01
    rally_max = S0 * (1.0 + g.rally_max_reach_pct)
    dip_grid = np.arange(dip_min, dip_max, dip_step)
    rally_grid = np.arange(rally_min, rally_max, rally_step)

    paths = run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu, horizon_days=HORIZON,
        n_paths=N_PATHS, distribution="student_t", df=5.0, seed=42,
    )
    dip_first_all = precompute_first_touch_days(
        paths, S0, dip_grid, sigma, None, "down", seed=42,
    )
    rally_first_all = precompute_first_touch_days(
        paths, S0, rally_grid, sigma, None, "up", seed=43,
    )

    rows = []
    for i, dip in enumerate(dip_grid):
        for j, rally in enumerate(rally_grid):
            fric_wait = (dip + rally) / 2.0 * fric_bps / 10000.0
            fric_direct = (S0 + rally) / 2.0 * fric_bps / 10000.0
            r = per_path_payoffs(
                paths, S0, float(dip), float(rally),
                fric_wait, fric_direct,
                dip_first_all[:, i], rally_first_all[:, j],
                PATIENCE_WINDOW_TD, stop_pct,
            )
            ev_direct = float(r["payoff_direct"].mean())
            ev_wait = float(r["payoff_wait"].mean())
            std_direct = float(r["payoff_direct"].std())
            std_wait = float(r["payoff_wait"].std())
            p_dip = float(r["dip_any"].mean())

            wait_eligible = p_dip >= MIN_DIP_PROBABILITY
            if wait_eligible and ev_wait >= ev_direct:
                strategy = "WAIT"
                ev = ev_wait
                std = std_wait
                ev_pct = ev_wait / dip if dip > 0 else 0.0
                p_rt = float((r["dip_any"] & r["wait_rally_hit"]).mean())
                hit_rate = p_rt  # round-trip completion = profitable exit (rally > dip + friction)
            else:
                strategy = "DIRECT"
                ev = ev_direct
                std = std_direct
                ev_pct = ev_direct / S0
                p_rt = float(r["direct_rally_hit"].mean())
                hit_rate = p_rt

            sharpe = ev / std if std > 1e-9 else 0.0
            rows.append({
                "dip": float(dip), "rally": float(rally),
                "dip_pct": (S0 - dip) / S0,
                "rally_pct": (rally - S0) / S0,
                "strategy": strategy,
                "ev_bps": ev_pct * 10000,
                "p_round_trip": p_rt,
                "hit_rate": hit_rate,
                "payoff_std_pct": std / S0 * 100,
                "sharpe": sharpe,
                "p_dip": p_dip,
            })
    return name, sigma_class, S0, sigma, mu, rows


def rank_and_compare(name, sigma_class, S0, sigma, mu, rows):
    ev_hurdle_bps = SIGMA_CLASSES[sigma_class].ev_hurdle_bps
    by_ev = sorted(rows, key=lambda r: r["ev_bps"], reverse=True)
    by_prt = sorted(rows, key=lambda r: r["p_round_trip"], reverse=True)
    by_sharpe = sorted(rows, key=lambda r: r["sharpe"], reverse=True)
    # Two-stage mission-aligned: filter to EV >= hurdle (refusal floor),
    # then rank by P(round-trip) (mission's "lock-in" hit-rate language).
    survivors = [r for r in rows if r["ev_bps"] >= ev_hurdle_bps]
    by_two_stage = sorted(survivors, key=lambda r: r["p_round_trip"], reverse=True)
    # Composite: P(RT) × |EV - hurdle| — bonus for clearing hurdle by a margin
    by_composite = sorted(
        survivors,
        key=lambda r: r["p_round_trip"] * (r["ev_bps"] - ev_hurdle_bps),
        reverse=True,
    )

    print()
    print("=" * 90)
    print(
        f"{name}  |  class={sigma_class}  S0={S0}  σ={sigma:.2f}  μ={mu:+.2f}"
        f"  |  {len(rows)} (dip, rally) pairs"
    )
    print("=" * 90)

    def fmt(r):
        return (
            f"dip={r['dip_pct']*100:5.1f}%  rally=+{r['rally_pct']*100:4.1f}%  "
            f"{r['strategy']:>6}  "
            f"EV={r['ev_bps']:+7.1f}bps  P_RT={r['p_round_trip']:.3f}  "
            f"σ_payoff={r['payoff_std_pct']:5.1f}%  Sharpe={r['sharpe']:+.3f}"
        )

    print("\nTop 5 by EV (current engine metric):")
    for r in by_ev[:5]:
        print(f"  {fmt(r)}")

    print("\nTop 5 by P(round-trip) (mission-aligned metric):")
    for r in by_prt[:5]:
        print(f"  {fmt(r)}")

    print("\nTop 5 by Sharpe (risk-adjusted):")
    for r in by_sharpe[:5]:
        print(f"  {fmt(r)}")

    print(f"\nTop 5 by TWO-STAGE [EV>={ev_hurdle_bps}bps then max P(RT)] "
          f"({len(survivors)}/{len(rows)} pairs clear hurdle):")
    if by_two_stage:
        for r in by_two_stage[:5]:
            print(f"  {fmt(r)}")
    else:
        print("  (no setup clears the EV hurdle — engine would refuse)")

    print("\nTop 5 by COMPOSITE [P(RT) × (EV - hurdle), survivors only]:")
    if by_composite:
        for r in by_composite[:5]:
            print(f"  {fmt(r)}")

    # Divergence
    ev_top = by_ev[0]
    prt_top = by_prt[0]
    sharpe_top = by_sharpe[0]
    same_ev_prt = (ev_top["dip"], ev_top["rally"]) == (prt_top["dip"], prt_top["rally"])
    same_ev_sharpe = (ev_top["dip"], ev_top["rally"]) == (sharpe_top["dip"], sharpe_top["rally"])

    print("\nDivergence:")
    print(f"  EV-top == P(RT)-top:  {same_ev_prt}")
    print(f"  EV-top == Sharpe-top: {same_ev_sharpe}")
    if not same_ev_prt:
        print(f"  EV-top has P(RT)={ev_top['p_round_trip']:.3f}; "
              f"P(RT)-top has P(RT)={prt_top['p_round_trip']:.3f}; "
              f"EV-top has EV={ev_top['ev_bps']:+.1f}bps, P(RT)-top has EV={prt_top['ev_bps']:+.1f}bps")
    if not same_ev_sharpe:
        print(f"  EV-top has Sharpe={ev_top['sharpe']:+.3f}; "
              f"Sharpe-top has Sharpe={sharpe_top['sharpe']:+.3f}; "
              f"Sharpe-top has EV={sharpe_top['ev_bps']:+.1f}bps")

    # Overlap stats — how many of EV's top-10 are in P(RT)'s top-10?
    ev_top10 = {(round(r["dip"], 4), round(r["rally"], 4)) for r in by_ev[:10]}
    prt_top10 = {(round(r["dip"], 4), round(r["rally"], 4)) for r in by_prt[:10]}
    sharpe_top10 = {(round(r["dip"], 4), round(r["rally"], 4)) for r in by_sharpe[:10]}
    print(f"  |EV-top10 ∩ P(RT)-top10|  = {len(ev_top10 & prt_top10)}/10")
    print(f"  |EV-top10 ∩ Sharpe-top10| = {len(ev_top10 & sharpe_top10)}/10")
    print(f"  |P(RT)-top10 ∩ Sharpe-top10| = {len(prt_top10 & sharpe_top10)}/10")


def main():
    setups = [
        # name, sigma_class, S0, sigma_annual, mu_annual
        ("MU-shape (HIGH, bullish μ)", "HIGH", 230.0, 0.90, 0.25),
        ("MRAM-shape (EXTREME, neutral μ)", "EXTREME", 12.0, 1.30, 0.00),
        ("MRAM-shape (EXTREME, bullish μ)", "EXTREME", 12.0, 1.30, 0.30),
        ("MRAM-shape (EXTREME, bearish μ)", "EXTREME", 12.0, 1.30, -0.20),
        ("LRCX-shape (MID, mildly bullish μ)", "MID", 1100.0, 0.45, 0.12),
    ]
    for name, cls, S0, sigma, mu in setups:
        rank_and_compare(*scan_setup(name, cls, S0, sigma, mu))


if __name__ == "__main__":
    main()

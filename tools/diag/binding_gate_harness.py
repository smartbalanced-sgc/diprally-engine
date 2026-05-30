"""Numerical harness: synthesize realistic inputs per σ-class, run the
ACTUAL engine math (run_mc_joint_conditional, analyze_joint_conditional,
compute_dual_ev), then evaluate each gate in the verdict waterfall.

Goal: identify the binding gate per σ-class under a range of (μ, σ, mom_30d)
inputs. Falsify or confirm the null 'engine CANNOT BUY'.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from src.config import (
    SIGMA_CLASSES, EV_HURDLE_BPS_OF_DIP,
    PARABOLA_FILTER_MOM_30D_THRESHOLD, TREND_FILTER_MOM_30D_THRESHOLD,
    DEFAULT_HORIZON_DAYS, DEFAULT_MC_PATHS, MC_DISTRIBUTION,
    GRID_PREFILTER_LOOSENESS, PATIENCE_WINDOW_TD, MIN_DIP_PROBABILITY,
)
from src.math_utils import (
    run_mc_joint_conditional, precompute_first_touch_days,
    analyze_joint_conditional, compute_dual_ev,
)

# Realistic σ-class inputs based on PR-39 calibration notes:
# EXTREME boundary ≥1.20; HIGH 0.65-1.20; MID <0.65.
# Realistic μ post-Bayesian (from CLAUDE.md / smoke history): 0% to +25%/yr.
# mom_30d: typical NON-extreme dip-and-rally range for a working setup.
SCENARIOS = [
    # (label, sigma_class, sigma, mu, mom_30d)
    ("EXTREME / σ=1.30 / μ=0%  / mom=0",       "EXTREME", 1.30, 0.00,  0.00),
    ("EXTREME / σ=1.30 / μ=+15% / mom=-10%",   "EXTREME", 1.30, 0.15, -0.10),
    ("EXTREME / σ=1.30 / μ=+25% / mom=+25%",   "EXTREME", 1.30, 0.25,  0.25),
    ("EXTREME / σ=1.50 / μ=+10% / mom=-15%",   "EXTREME", 1.50, 0.10, -0.15),
    ("HIGH    / σ=0.80 / μ=0%   / mom=0",       "HIGH",    0.80, 0.00,  0.00),
    ("HIGH    / σ=0.80 / μ=+15% / mom=-10%",    "HIGH",    0.80, 0.15, -0.10),
    ("HIGH    / σ=1.00 / μ=+20% / mom=+30%",    "HIGH",    1.00, 0.20,  0.30),
    ("MID     / σ=0.40 / μ=+10% / mom=0",       "MID",     0.40, 0.10,  0.00),
    ("MID     / σ=0.50 / μ=+15% / mom=-5%",     "MID",     0.50, 0.15, -0.05),
    ("MID     / σ=0.40 / μ=+20% / mom=+30%",    "MID",     0.40, 0.20,  0.30),
]

S0 = 100.0
H = DEFAULT_HORIZON_DAYS   # 20

def eval_scenario(label, cls, sigma, mu, mom_30d):
    spec = SIGMA_CLASSES[cls]
    grid = spec.grid
    fric_bps = spec.friction_bps_round_trip
    ev_hurdle_bps = getattr(spec, "ev_hurdle_bps", None) or EV_HURDLE_BPS_OF_DIP
    para_thr = getattr(spec, "parabola_mom_30d_threshold", None) or PARABOLA_FILTER_MOM_30D_THRESHOLD
    conv_dip = spec.conviction.dip
    conv_rally_cond = spec.conviction.rally_conditional

    df = MC_DISTRIBUTION.per_class.get(cls, MC_DISTRIBUTION.default_df)
    paths = run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu, horizon_days=H, n_paths=DEFAULT_MC_PATHS,
        vol_schedule=None, distribution=MC_DISTRIBUTION.default, df=df, seed=42,
    )

    dip_min = S0 * (1.0 - grid.dip_max_depth_pct)
    dip_max = S0 * 0.99
    rally_min = S0 * 1.01
    rally_max = S0 * (1.0 + grid.rally_max_reach_pct)
    dip_grid = np.arange(dip_min, dip_max, S0 * grid.dip_step_pct)
    rally_grid = np.arange(rally_min, rally_max, S0 * grid.rally_step_pct)

    dip_first_all = precompute_first_touch_days(paths, S0, dip_grid, sigma, None, "down", seed=42)
    rally_first_all = precompute_first_touch_days(paths, S0, rally_grid, sigma, None, "up", seed=43)

    best = None  # tuple (ev_for_hurdle, p_dip, p_rally_cond, dip, rally, subtype, ev_direct_pct_spot, ev_wait_pct_dip)
    best_qualified = None
    best_unqualified = None
    n_pre = 0  # passed pre-filter
    n_qualified = 0  # passed strict thresholds
    p_dip_max_seen = 0.0
    p_rcond_max_seen = 0.0

    for i, dip in enumerate(dip_grid):
        for j, rally in enumerate(rally_grid):
            res = analyze_joint_conditional(
                paths, S0, float(dip), float(rally), H,
                dip_first_days=dip_first_all[:, i],
                rally_first_days=rally_first_all[:, j],
            )
            p_dip = res["p_dip_touched_marginal"]
            p_rcond = res["p_rally_given_dip_conditional"]
            p_dip_max_seen = max(p_dip_max_seen, p_dip)
            p_rcond_max_seen = max(p_rcond_max_seen, p_rcond)

            if p_dip < conv_dip - GRID_PREFILTER_LOOSENESS:
                continue
            if p_rcond < conv_rally_cond - GRID_PREFILTER_LOOSENESS:
                continue
            n_pre += 1
            fric_per = (float(dip) + float(rally)) / 2.0 * fric_bps / 10000.0

            dual = compute_dual_ev(
                paths, S0, float(dip), float(rally), fric_per,
                dip_first_days=dip_first_all[:, i],
                rally_first_days=rally_first_all[:, j],
                patience_window_td=PATIENCE_WINDOW_TD,
            )
            wait_ok = dual["p_dip_filled"] >= MIN_DIP_PROBABILITY
            if wait_ok and dual["ev_wait_per_share"] >= dual["ev_direct_per_share"]:
                ev_net = dual["ev_wait_per_share"]
                ev_pct = dual["ev_wait_pct_of_dip"]
                sub = "WAIT"
            else:
                ev_net = dual["ev_direct_per_share"]
                ev_pct = dual["ev_direct_pct_of_spot"]
                sub = "DIRECT"

            entry = (ev_pct, p_dip, p_rcond, float(dip), float(rally), sub,
                     dual["ev_direct_pct_of_spot"], dual["ev_wait_pct_of_dip"], ev_net)
            if best_unqualified is None or entry[0] > best_unqualified[0]:
                best_unqualified = entry
            if p_dip >= conv_dip and p_rcond >= conv_rally_cond:
                n_qualified += 1
                if best_qualified is None or entry[0] > best_qualified[0]:
                    best_qualified = entry

    best = best_qualified or best_unqualified
    ev_hurdle_threshold = ev_hurdle_bps / 10000.0

    # Apply waterfall (as in engine.py)
    trend_refused = (mom_30d < TREND_FILTER_MOM_30D_THRESHOLD)  # no AI catalysts in synth
    para_refused = (mom_30d >= para_thr)  # no AI bearish catalyst in synth
    if best is None:
        verdict = "WAIT (no pair passed prefilter)"
    else:
        ev_pct = best[0]
        if trend_refused:
            verdict = f"REFUSED-TREND (mom_30d {mom_30d*100:+.0f}% < {TREND_FILTER_MOM_30D_THRESHOLD*100:.0f}%)"
        elif para_refused:
            verdict = f"REFUSED-PARABOLA (mom_30d {mom_30d*100:+.0f}% >= {para_thr*100:.0f}%)"
        elif ev_pct < ev_hurdle_threshold:
            verdict = f"REFUSED-EV (ev/dip {ev_pct*1e4:+.1f}bps < {ev_hurdle_bps}bps)"
        elif best_qualified is None:
            verdict = f"BELOW-THRESHOLD (best p_dip={best[1]:.2f}<{conv_dip:.2f} or p_rcond={best[2]:.2f}<{conv_rally_cond:.2f})"
        elif best[8] < 0:
            verdict = f"NEGATIVE-EV (ev_per_share={best[8]:.2f})"
        else:
            verdict = "BUY"

    print(f"\n=== {label} ===")
    print(f"   sigma_class={cls} σ={sigma:.2f} μ={mu*100:+.1f}% mom_30d={mom_30d*100:+.1f}%")
    print(f"   thresholds: conv_dip={conv_dip:.2f} conv_rcond={conv_rally_cond:.2f}  "
          f"EV hurdle={ev_hurdle_bps}bps ({ev_hurdle_bps/100:.2f}%)  "
          f"parabola≥{para_thr*100:.0f}% friction={fric_bps}bps ({fric_bps/100:.2f}%)")
    print(f"   grid: dip {dip_min:.1f}..{dip_max:.1f} step={S0*grid.dip_step_pct:.2f}  "
          f"rally {rally_min:.1f}..{rally_max:.1f} step={S0*grid.rally_step_pct:.2f}")
    print(f"   grid scan: max p_dip seen={p_dip_max_seen:.2f}  max p_rally|dip seen={p_rcond_max_seen:.2f}  "
          f"n_prefilter_passed={n_pre}  n_qualified_strict={n_qualified}")
    if best:
        ev_pct = best[0]
        print(f"   best pair: dip={best[3]:.2f} rally={best[4]:.2f} strategy={best[5]}  "
              f"p_dip={best[1]:.2f} p_rcond={best[2]:.2f}")
        print(f"   ev_direct_pct_spot={best[6]*1e4:+.1f}bps  ev_wait_pct_dip={best[7]*1e4:+.1f}bps  "
              f"chosen_ev_pct={ev_pct*1e4:+.1f}bps ({ev_pct*100:+.2f}%)")
    print(f"   VERDICT → {verdict}")


def main():
    print(f"horizon_days={H}  paths={DEFAULT_MC_PATHS}  trend_filter_thr={TREND_FILTER_MOM_30D_THRESHOLD}")
    print(f"MC distribution: {MC_DISTRIBUTION.default} default_df={MC_DISTRIBUTION.default_df}")
    print(f"patience_window_td={PATIENCE_WINDOW_TD}  min_dip_probability={MIN_DIP_PROBABILITY}")
    for sc in SCENARIOS:
        eval_scenario(*sc)


if __name__ == "__main__":
    main()

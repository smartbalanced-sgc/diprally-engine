"""Mean-reversion-anchor probe (2026-05-30 audit step 3).

Hypothesis to test: re-aiming the mean_reversion anchor from "5% BELOW spot"
(current YAML, wrong direction) to AT or ABOVE spot supplies enough rally
bias to push the 4 near-miss tickers (RKLB, MU, AMAT, ARM at +9 to +24 bps)
over their EV hurdles, without producing spurious BUYs on falling-knife
setups (ENGN-style mom_30d < -25%).

Sweeps (anchor_pct, strength k) for each σ-class, plus the strict
'don't BUY on falling knife' test using a mom_30d-shifted path series.

Current mean_reversion math (src/math_utils.py run_mc_joint_conditional):
    deviation = (path[t-1] - anchor) / anchor
    mr_drift  = -k * deviation
So anchor at-spot → mr_drift = 0 at start, restoring toward spot as path drifts.
anchor above spot → mr_drift > 0 when at spot (rally bias).
anchor below spot → mr_drift < 0 when at spot (the current YAML's wrong-direction default).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from src.config import (
    SIGMA_CLASSES, PATIENCE_WINDOW_TD,
    MC_DISTRIBUTION, DEFAULT_HORIZON_DAYS, DEFAULT_MC_PATHS,
)
from src.math_utils import (
    run_mc_joint_conditional,
    precompute_first_touch_days,
    compute_dual_ev,
)

S0 = 100.0
H = DEFAULT_HORIZON_DAYS  # 20


def probe(cls: str, sigma: float, mu: float, dip: float, rally: float,
          anchor_pct_above_spot: float, k: float, swing_stop_pct: float):
    """One MC run + EV measurement at a given (anchor, k). Uses
    precompute_first_touch_days for bridge-corrected dip/rally barrier
    detection so the probe matches engine.scan_dip_rally_grid behavior.
    Stops still use daily-close detection (matches engine implementation
    in compute_dual_ev).

    Returns (chosen, ev_direct, ev_wait, p_round_trip_strict,
    p_stopped_wait, p_dip_filled).
    """
    anchor = S0 * (1.0 + anchor_pct_above_spot) if k > 0 else None
    df = MC_DISTRIBUTION.per_class.get(cls, MC_DISTRIBUTION.default_df)
    paths = run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu, horizon_days=H,
        n_paths=DEFAULT_MC_PATHS,
        distribution=MC_DISTRIBUTION.default, df=df, seed=42,
        mean_reversion_strength=k,
        mean_reversion_anchor=anchor,
    )
    dip_first = precompute_first_touch_days(
        paths, S0, np.array([dip]), sigma, None, "down", seed=42,
    )[:, 0]
    rally_first = precompute_first_touch_days(
        paths, S0, np.array([rally]), sigma, None, "up", seed=43,
    )[:, 0]
    fric_bps = SIGMA_CLASSES[cls].friction_bps_round_trip
    fric_w = (dip + rally) / 2 * fric_bps / 10000
    fric_d = (S0 + rally) / 2 * fric_bps / 10000
    out = compute_dual_ev(
        paths, S0, dip, rally, fric_w,
        dip_first_days=dip_first,
        rally_first_days=rally_first,
        patience_window_td=PATIENCE_WINDOW_TD,
        swing_stop_pct=swing_stop_pct,
        friction_per_share_direct=fric_d,
    )
    chosen = max(out["ev_wait_pct_of_dip"], out["ev_direct_pct_of_spot"])
    return (chosen, out["ev_direct_pct_of_spot"], out["ev_wait_pct_of_dip"],
            out["p_round_trip_strict"], out["p_stopped_wait"],
            out["p_dip_filled"])


def header():
    return (f"  {'anchor%':>8} {'k':>5} {'p_dip':>6} {'p_rt_strict':>11} "
            f"{'p_stp_W':>8} {'ev_wait':>10} {'ev_direct':>11} {'CHOSEN':>10}")


def row(anchor_pct, k, dip_filled, p_rt, p_stp, ev_w, ev_d, chosen, hurdle):
    flag = '✓BUY' if chosen >= hurdle/10000 else '   ✗'
    return (f"  {anchor_pct*100:+7.1f}% {k:>5.2f} {dip_filled:>6.2f} "
            f"{p_rt:>11.3f} {p_stp:>8.2f}  {ev_w*100:+8.2f}%  {ev_d*100:+9.2f}%  "
            f"{chosen*100:+8.2f}% {flag}")


# Scenarios pulled from the LIVE engine output (post-stop-layer fix).
# (cls, sigma, mu, dip, rally, swing_stop, hurdle_bps, label_ticker)
LIVE_SCENARIOS = [
    ("EXTREME", 1.486, 0.119, 10.86*0.92, 10.86*1.10, 0.10, 25, "LWLG"),
    ("HIGH",    0.95,  0.12,  100*0.96,   100*1.07,   0.07, 25, "MU/RKLB-shaped"),
    ("MID",     0.45,  0.10,  100*0.98,   100*1.04,   0.05, 50, "AMAT-shaped"),
]


def main():
    print("=" * 88)
    print(" MEAN-REVERSION-ANCHOR PROBE — fixing the 2nd binding gate (μ vs vol drag)")
    print("=" * 88)
    print(f" horizon={H}td  paths={DEFAULT_MC_PATHS}  stop layer ACTIVE")
    print(f" post-2026-05-30 YAML: mean_reversion.anchor_pct_above_spot = 0.0 (at-spot), default_strength = 2.0 (ON)")
    print(f" probe sweep: anchor_pct_above_spot ∈ [-5%, 0%, +5%, +10%]")
    print(f"              strength k ∈ [0, 1, 2, 5, 10]")
    print()

    for cls, sigma, mu, dip_abs, rally_abs, stop, hurdle_bps, label in LIVE_SCENARIOS:
        # Renormalize dip/rally to S0=100 reference
        dip = 100 * dip_abs / dip_abs * (dip_abs / (dip_abs if cls!='EXTREME' else 10.86))
        # Easier: just convert to pct
        if cls == "EXTREME":
            dip = 92.0; rally = 110.0
        else:
            dip = dip_abs
            rally = rally_abs
        hurdle = hurdle_bps / 10000
        print(f"--- {cls} σ={sigma:.2f} μ={mu*100:+.0f}% dip={dip:.1f} rally={rally:.1f} stop={stop*100:.0f}% hurdle={hurdle_bps}bps ({label}) ---")
        print(header())
        for anchor_pct in [-0.05, 0.0, 0.05, 0.10]:
            for k in [0.0, 1.0, 2.0, 5.0, 10.0]:
                chosen, ev_d, ev_w, p_rt, p_stp, p_filled = probe(
                    cls, sigma, mu, dip, rally,
                    anchor_pct_above_spot=anchor_pct, k=k,
                    swing_stop_pct=stop,
                )
                print(row(anchor_pct, k, p_filled, p_rt, p_stp, ev_w, ev_d, chosen, hurdle_bps))
        print()

    print("=" * 88)
    print(" FALSE-POSITIVE GUARD: ENGN-shape falling knife (μ very negative, σ huge)")
    print(" If mean-rev re-aim turns this into a BUY, the math is over-corrected.")
    print("=" * 88)
    print(f" Setup: EXTREME σ=1.50 μ=-0.30 (deeply impaired) — sacred #14 trend filter ")
    print(f" applies separately at mom_30d < -25%, but here we test the MC-only EV.")
    print(header())
    for anchor_pct in [-0.05, 0.0, 0.05, 0.10]:
        for k in [0.0, 2.0, 5.0]:
            chosen, ev_d, ev_w, p_rt, p_stp, p_filled = probe(
                "EXTREME", 1.50, -0.30, 92.0, 110.0,
                anchor_pct_above_spot=anchor_pct, k=k,
                swing_stop_pct=0.10,
            )
            print(row(anchor_pct, k, p_filled, p_rt, p_stp, ev_w, ev_d, chosen, 25))
    print()
    print("Interpretation: any 'BUY' here means the mean-reversion calibration")
    print("rescues impaired-thesis paths. Sacred #14 catches this at the trend")
    print("filter (mom_30d < -25% gate) BEFORE this EV is computed, so MC-only")
    print("BUYs on -μ paths are fine SO LONG AS sacred #14 is firing in production.")


if __name__ == "__main__":
    main()

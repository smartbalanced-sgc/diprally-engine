"""Decompose EV negativity. Run the same setup under multiple
permutations to attribute the ~3% negative EV between distribution
(student_t vs normal), exit stops, and μ. EXTREME class, σ=1.30."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from src.config import SIGMA_CLASSES, PATIENCE_WINDOW_TD, MIN_DIP_PROBABILITY, DEFAULT_MC_PATHS, DEFAULT_HORIZON_DAYS
from src.math_utils import run_mc_joint_conditional, precompute_first_touch_days, analyze_joint_conditional, compute_dual_ev

S0 = 100.0
H = 20
N = 100_000
sigma = 1.30
dip, rally = 92.0, 112.0
fric_bps = 70.0
fric = (dip+rally)/2 * fric_bps/10000

def run(dist, df, mu, stop_pct=None, patience=PATIENCE_WINDOW_TD, label=""):
    paths = run_mc_joint_conditional(
        S0=S0, sigma=sigma, mu=mu, horizon_days=H, n_paths=N,
        distribution=dist, df=df, seed=42,
    )
    # WAIT EV
    dip_first = precompute_first_touch_days(paths, S0, np.array([dip]), sigma, None, "down", seed=42)[:, 0]
    rally_first = precompute_first_touch_days(paths, S0, np.array([rally]), sigma, None, "up", seed=43)[:, 0]
    days_idx = np.arange(H)[None, :]
    n_paths_local = paths.shape[0]

    dip_any = dip_first < H
    after = days_idx >= dip_first[:, None]
    eff_patience = patience if patience is not None else H
    within = after & (days_idx <= (dip_first[:, None] + eff_patience))
    rally_after_dip = ((paths >= rally) & within).any(axis=1)
    wait_exit_idx = np.minimum(dip_first + eff_patience, H-1).astype(int)
    wait_exit_price = paths[np.arange(n_paths_local), wait_exit_idx]
    # apply stop on bag-hold
    if stop_pct is not None:
        stop_level = dip * (1 - stop_pct)
        # if any path touched stop after dip, exit at stop
        stopped = ((paths <= stop_level) & after).any(axis=1)
        wait_exit_price = np.where(stopped, stop_level, wait_exit_price)
    rt_payoff = rally - dip - fric
    bag_payoff = wait_exit_price - dip - fric
    wait_payoff = np.where(~dip_any, 0.0, np.where(rally_after_dip, rt_payoff, bag_payoff))
    ev_wait = float(wait_payoff.mean())
    ev_wait_pct_dip = ev_wait / dip * 1e4

    # DIRECT EV
    eff_patience_d = patience if patience is not None else H
    direct_within = days_idx <= eff_patience_d
    direct_rally = ((paths >= rally) & direct_within).any(axis=1)
    direct_exit_idx = min(eff_patience_d, H-1)
    direct_exit_price = paths[:, direct_exit_idx]
    if stop_pct is not None:
        stop_level_d = S0 * (1 - stop_pct)
        stopped_d = (paths <= stop_level_d).any(axis=1)
        direct_exit_price = np.where(stopped_d, stop_level_d, direct_exit_price)
    direct_payoff = np.where(direct_rally, rally - S0 - fric, direct_exit_price - S0 - fric)
    ev_direct = float(direct_payoff.mean())
    ev_direct_pct_spot = ev_direct / S0 * 1e4

    # Diagnostics
    p_dip = float(dip_any.mean())
    p_rally_after = float(rally_after_dip.mean())
    p_rally_any = float((paths >= rally).any(axis=1).mean())
    bag_paths = dip_any & ~rally_after_dip
    bag_term_median = float(np.median(paths[bag_paths, -1])) if bag_paths.any() else float('nan')
    bag_payoff_mean = float(np.mean(bag_payoff[bag_paths])) if bag_paths.any() else float('nan')

    print(f"  {label:<55} dist={dist:<9} df={df}  μ={mu:+.2f}  stop={str(stop_pct):<6} pat={patience}")
    print(f"    p_dip={p_dip:.2f}  p_rally_any={p_rally_any:.2f}  p_rally_after_dip={p_rally_after:.2f}  "
          f"bag_term_median={bag_term_median:.1f}  bag_mean_loss={bag_payoff_mean:+.1f}")
    print(f"    ev_wait_pct_dip={ev_wait_pct_dip:+.1f}bps   ev_direct_pct_spot={ev_direct_pct_spot:+.1f}bps")
    print()


print(f"\nEXTREME class probe: spot={S0}, dip={dip}, rally={rally}, σ={sigma}, H={H}td, friction={fric_bps}bps RT")
print(f"   default patience_window_td={PATIENCE_WINDOW_TD}, default horizon={DEFAULT_HORIZON_DAYS}\n")

print("AS-IS (engine default: student_t df=4, no stop, patience=40):")
run("student_t", 4.0, 0.00, None, 40, "engine default μ=0")
run("student_t", 4.0, 0.15, None, 40, "engine default μ=+15%")
run("student_t", 4.0, 0.50, None, 40, "engine default μ=+50%")

print("FLIP distribution to normal (everything else default):")
run("normal", 5.0, 0.00, None, 40, "normal μ=0")
run("normal", 5.0, 0.15, None, 40, "normal μ=+15%")

print("ADD a -10% exit stop (student_t df=4, μ=0%, +15%):")
run("student_t", 4.0, 0.00, 0.10, 40, "stop=10% μ=0")
run("student_t", 4.0, 0.15, 0.10, 40, "stop=10% μ=+15%")

print("ADD a -5% exit stop (student_t df=4, μ=0%, +15%):")
run("student_t", 4.0, 0.00, 0.05, 40, "stop=5% μ=0")
run("student_t", 4.0, 0.15, 0.05, 40, "stop=5% μ=+15%")

print("PATIENCE shorter than horizon (10 td, no stop):")
run("student_t", 4.0, 0.00, None, 10, "patience=10 μ=0")

print("Vol-drag-neutral μ=σ²/2 ≈ 0.845 (student_t default):")
run("student_t", 4.0, 0.845, None, 40, "μ=σ²/2 (drift-corrected)")

# MID with stop probe
sigma = 0.40
dip, rally = 95.0, 108.0
fric_bps = 18.0
fric = (dip+rally)/2 * fric_bps/10000

print("\n--- MID class probe (σ=0.40, dip=95, rally=108, fric=18bps) ---")
run("student_t", 7.0, 0.15, None, 40, "MID AS-IS μ=+15%")
run("student_t", 7.0, 0.15, 0.05, 40, "MID +5% stop μ=+15%")
run("student_t", 7.0, 0.15, 0.10, 40, "MID +10% stop μ=+15%")

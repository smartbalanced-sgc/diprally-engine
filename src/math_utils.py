"""Math layer: GARCH fits, MC paths, PDE first-passage, closed-form barriers,
vol triangulation, vol schedule, path metrics.

Three-method cross-check (MC + PDE + closed-form) per sacred decision #8.
Brownian bridge correction on MC barrier (#9).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.stats import norm

from src.config import (
    DEFAULT_MC_PATHS,
    GARCH_FALLBACK_BARS,
    GARCH_INITIAL_ALPHA,
    GARCH_INITIAL_BETA,
    GARCH_INITIAL_OMEGA,
    GARCH_INITIAL_OMEGA_FULL,
    GARCH_MIN_DATA_BARS,
    PDE_N_SPACE,
    PDE_N_TIME,
    REALIZED_VOL_WINDOWS,
    VOL_SCHEDULE_MULTIPLIERS,
    method_refusal_pp,
    method_tolerance_pp,
)


# =============================================================================
# GARCH / RSI / drift enrichment
# =============================================================================

def fit_garch_11(returns):
    """GARCH(1,1) one-step-ahead variance forecast. Scalar return.
    Parameters in config/diprally.yaml under garch (sacred #17, D-W2-9)."""
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < GARCH_MIN_DATA_BARS:
        return r.var()

    def neg_ll(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        T = len(r)
        s2 = np.zeros(T)
        s2[0] = r.var()
        for t in range(1, T):
            s2[t] = omega + alpha * r.iloc[t-1]**2 + beta * s2[t-1]
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + r.values**2 / s2)

    try:
        res = minimize(neg_ll,
                       [GARCH_INITIAL_OMEGA, GARCH_INITIAL_ALPHA, GARCH_INITIAL_BETA],
                       method="L-BFGS-B",
                       bounds=[(1e-6, 1), (0, 1), (0, 1)])
        omega, alpha, beta = res.x
        last_var = r.tail(20).var()
        return omega + alpha * r.iloc[-1]**2 + beta * last_var
    except Exception:
        return r.tail(GARCH_FALLBACK_BARS).var()


def fit_garch_11_full(returns: pd.Series) -> dict:
    """GARCH(1,1) full fit returning {omega, alpha, beta, forecast_variance, fit_ok}.

    σ²(t) = ω + α r²(t-1) + β σ²(t-1)
    Stationarity: α + β < 1 (else non-stationary / IGARCH).
    """
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < GARCH_MIN_DATA_BARS:
        return {
            "omega": 0.0, "alpha": 0.0, "beta": 0.0,
            "forecast_variance": float(r.var()) if len(r) > 0 else 1e-6,
            "fit_ok": False,
        }

    def neg_ll(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e10
        T = len(r)
        s2 = np.zeros(T)
        s2[0] = r.var()
        for t in range(1, T):
            s2[t] = omega + alpha * r.iloc[t - 1] ** 2 + beta * s2[t - 1]
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + r.values ** 2 / s2)

    try:
        res = minimize(
            neg_ll,
            [GARCH_INITIAL_OMEGA_FULL, GARCH_INITIAL_ALPHA, GARCH_INITIAL_BETA],
            method="L-BFGS-B",
            bounds=[(1e-8, 1.0), (0.0, 1.0), (0.0, 0.9999)],
        )
        omega, alpha, beta = res.x
        last_var = float(r.tail(20).var())
        forecast_var = float(omega + alpha * r.iloc[-1] ** 2 + beta * last_var)
        return {
            "omega": float(omega),
            "alpha": float(alpha),
            "beta": float(beta),
            "forecast_variance": forecast_var,
            "fit_ok": bool(res.success and forecast_var > 0 and not np.isnan(forecast_var)),
        }
    except Exception:
        fallback_var = float(r.tail(GARCH_FALLBACK_BARS).var())
        return {
            "omega": 0.0, "alpha": 0.0, "beta": 0.0,
            "forecast_variance": fallback_var, "fit_ok": False,
        }


def compute_rsi_14(closes):
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0


def enrichment_drift(rsi, mom_5d):
    rsi_drift = (50.0 - rsi) / 500.0
    mom_drift = -mom_5d / 1000.0
    return max(-0.10, min(0.10, rsi_drift + mom_drift))


def compute_realized_vol(returns, windows=None):
    """Realized vol over multiple rolling windows. Returns {window: annualised sigma}.

    `windows` defaults to config realized_vol_windows (sacred #17, D-W2-9).
    Default (30, 60, 90) anchors σ triangulation.
    """
    if windows is None:
        windows = REALIZED_VOL_WINDOWS
    out = {}
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    for w in windows:
        if len(r) < w + 1:
            out[w] = None
            continue
        recent = r.tail(w)
        out[w] = float(recent.std() * np.sqrt(252))
    return out


def triangulate_sigma(garch_sigma, realized_vol_dict, options_iv_data):
    """Triangulate sigma across GARCH + realized + options IV."""
    anchors = {}
    if garch_sigma is not None:
        anchors["garch"] = float(garch_sigma)
    for w, v in (realized_vol_dict or {}).items():
        if v is not None:
            anchors[f"realized_{w}d"] = float(v)
    if options_iv_data and options_iv_data.get("is_liquid"):
        anchors["options_iv"] = float(options_iv_data["iv"])

    if not anchors:
        return None
    values = list(anchors.values())
    blended = float(np.mean(values))
    divergence = (max(values) - min(values)) if len(values) > 1 else 0.0

    return {
        "blended": blended,
        "anchors": anchors,
        "n_anchors": len(anchors),
        "divergence_pp": divergence * 100,
    }


# =============================================================================
# Closed-form barrier probabilities (reflection principle)
# =============================================================================

def closed_touch_up(S0, U, T, mu, sigma):
    """P(max S_t >= U over [0,T]) for GBM — reflection + Girsanov."""
    nu = mu - sigma**2 / 2
    s = sigma * np.sqrt(T)
    u = np.log(U / S0)
    return ((1 - norm.cdf((u - nu * T) / s))
            + np.exp(2 * nu * u / sigma**2) * (1 - norm.cdf((u + nu * T) / s)))


def closed_touch_down(S0, L, T, mu, sigma):
    """P(min S_t <= L over [0,T]) for GBM."""
    nu = mu - sigma**2 / 2
    s = sigma * np.sqrt(T)
    l = np.log(L / S0)
    return (norm.cdf((l - nu * T) / s)
            + np.exp(2 * nu * l / sigma**2) * norm.cdf((l + nu * T) / s))


# =============================================================================
# PDE two-barrier first-passage (Fokker-Planck, Crank-Nicolson)
# =============================================================================

def pde_two_barrier(S0, U, L, T, mu, sigma, n_space=None, n_time=None):
    """Solve Fokker-Planck PDE with absorbing barriers at L and U.

    PDE in log-space: dp/dt = -nu * dp/dx + (sigma^2/2) * d^2p/dx^2
    where nu = mu - sigma^2/2 (Ito correction).

    Grid resolution (n_space, n_time) defaults from config pde_grid
    (sacred #17, D-W2-9). Researchers tuning accuracy/runtime tradeoff
    edit YAML rather than function signatures.
    """
    if n_space is None:
        n_space = PDE_N_SPACE
    if n_time is None:
        n_time = PDE_N_TIME
    x_L, x_U = np.log(L), np.log(U)
    x = np.linspace(x_L, x_U, n_space)
    dx = x[1] - x[0]
    dt = T / n_time
    nu = mu - 0.5 * sigma**2

    n_int = n_space - 2
    p = np.zeros(n_int)
    i0 = int(np.argmin(np.abs(x - np.log(S0))))
    if 1 <= i0 <= n_space - 2:
        p[i0 - 1] = 1.0 / dx

    a = nu / (2 * dx) + 0.5 * sigma**2 / dx**2
    b_coef = -sigma**2 / dx**2
    c = -nu / (2 * dx) + 0.5 * sigma**2 / dx**2

    M_main = np.full(n_int, 1 - 0.5 * dt * b_coef)
    M_low = np.full(n_int - 1, -0.5 * dt * a)
    M_up = np.full(n_int - 1, -0.5 * dt * c)
    N_main = np.full(n_int, 1 + 0.5 * dt * b_coef)
    N_low = np.full(n_int - 1, 0.5 * dt * a)
    N_up = np.full(n_int - 1, 0.5 * dt * c)

    M_mat = diags([M_low, M_main, M_up], [-1, 0, 1], format="csc")
    N_mat = diags([N_low, N_main, N_up], [-1, 0, 1], format="csc")
    solver = splu(M_mat)

    cum_U = cum_L = 0.0
    for _ in range(n_time):
        p_new = solver.solve(N_mat @ p)
        avg_top = 0.5 * (p[-1] + p_new[-1])
        avg_bot = 0.5 * (p[0] + p_new[0])
        cum_U += (0.5 * sigma**2 * avg_top / dx) * dt
        cum_L += (0.5 * sigma**2 * avg_bot / dx) * dt
        p = p_new

    p_neither = float(np.sum(p) * dx)
    x_int = x[1:-1]
    if p_neither > 1e-9:
        E_term = float(np.sum(np.exp(x_int) * p) * dx / p_neither)
    else:
        E_term = 0.5 * (U + L)

    return {
        "p_U_first": float(cum_U),
        "p_L_first": float(cum_L),
        "p_neither": p_neither,
        "E_term_neither": E_term,
        "total": float(cum_U + cum_L + p_neither),
    }


# =============================================================================
# Monte Carlo — joint conditional with bridge correction
# =============================================================================

def _draw_innovations(rng, n_paths: int, horizon_days: int,
                       distribution: str = "normal",
                       df: float = 5.0) -> np.ndarray:
    """W9 PR #48: draw daily innovation matrix shape (n_paths, horizon_days)
    with unit variance, suitable for GBM step σ * sqrt(dt) * z.

    distribution="normal"     → standard normal draws (classic GBM).
    distribution="student_t"  → Student-t(df) draws rescaled to unit
                                variance (raw T(df) variance is df/(df-2),
                                so we divide by sqrt(df/(df-2)) to make
                                σ-scaling work identically to the normal
                                case). df must be > 2.

    Student-t innovations preserve the σ-input interpretation (annual
    vol) while making daily moves fat-tailed. Kurtosis at df=5 is 6
    vs normal's 3 — captures the parabolic / panic moves the
    diprally universe routinely produces.
    """
    if distribution == "normal":
        return rng.standard_normal((n_paths, horizon_days))
    if distribution == "student_t":
        if df <= 2.0:
            raise ValueError(
                f"Student-t df must be > 2 for finite variance (got {df})"
            )
        # Raw Student-t has variance df/(df-2); rescale to unit variance
        # so σ * sqrt(dt) * z keeps the same σ interpretation as normal.
        raw = rng.standard_t(df, size=(n_paths, horizon_days))
        scale = np.sqrt(df / (df - 2.0))
        return raw / scale
    raise ValueError(
        f"Unknown MC distribution {distribution!r} "
        f"(expected 'normal' or 'student_t')"
    )


def run_mc_joint_conditional(
    S0: float,
    sigma: float,
    mu: float,
    horizon_days: int,
    n_paths: int = DEFAULT_MC_PATHS,
    vol_schedule: Optional[np.ndarray] = None,
    mean_reversion_strength: float = 0.0,
    mean_reversion_anchor: Optional[float] = None,
    seed: int = 42,
    distribution: str = "normal",
    df: float = 5.0,
) -> np.ndarray:
    """Generate Monte Carlo paths with optional time-varying vol and mean reversion.

    Returns paths shape (n_paths, horizon_days) — daily prices, not including
    initial spot. Joint-conditional analysis in scan_dip_rally_grid uses these
    paths to compute P(dip touched then rally before horizon end).

    Mean reversion OFF by default.
    """
    # W9 PR #48: use a local Generator so distribution-switching is
    # deterministic and doesn't poison the global RNG. Student-t draws
    # go through _draw_innovations which rescales to unit variance.
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    sd = sigma * np.sqrt(dt)

    z = _draw_innovations(rng, n_paths, horizon_days,
                            distribution=distribution, df=df)
    paths = np.zeros((n_paths, horizon_days + 1))
    paths[:, 0] = S0

    for t in range(1, horizon_days + 1):
        if vol_schedule is not None:
            sd_t = vol_schedule[t - 1] * np.sqrt(dt)
        else:
            sd_t = sd

        if mean_reversion_strength > 0 and mean_reversion_anchor is not None:
            deviation = (paths[:, t - 1] - mean_reversion_anchor) / mean_reversion_anchor
            mr_drift = -mean_reversion_strength * deviation
        else:
            mr_drift = 0.0

        gbm_drift = (mu - 0.5 * sigma**2) * dt + mr_drift * dt
        log_step = gbm_drift + sd_t * z[:, t - 1]
        paths[:, t] = paths[:, t - 1] * np.exp(log_step)

    return paths[:, 1:]


def precompute_first_touch_days(
    paths: np.ndarray,
    S0: float,
    barriers: np.ndarray,
    sigma: float,
    vol_schedule: Optional[np.ndarray],
    direction: str,
    seed: int = 42,
) -> np.ndarray:
    """For each barrier, first-touch day per path with Brownian bridge correction.

    Returns (n_paths, n_barriers) with sentinel n_days when never touched.
    """
    n_paths, n_days = paths.shape
    n_barriers = len(barriers)
    result = np.full((n_paths, n_barriers), n_days, dtype=np.int32)

    prev = np.concatenate([np.full((n_paths, 1), S0), paths[:, :-1]], axis=1)
    log_prev = np.log(prev)
    log_curr = np.log(paths)

    if vol_schedule is not None:
        sigma_d_sq = (vol_schedule.astype(float) ** 2) / 252.0
    else:
        sigma_d_sq = np.full(n_days, (sigma ** 2) / 252.0)
    sigma_d_sq = np.maximum(sigma_d_sq, 1e-12)

    rng = np.random.default_rng(seed=seed)

    for i, B in enumerate(barriers):
        log_B = float(np.log(B))
        if direction == "down":
            close_touch = paths <= B
            both_safe = (log_prev > log_B) & (log_curr > log_B)
            dx = log_prev - log_B
            dy = log_curr - log_B
        else:
            close_touch = paths >= B
            both_safe = (log_prev < log_B) & (log_curr < log_B)
            dx = log_B - log_prev
            dy = log_B - log_curr

        with np.errstate(divide="ignore", invalid="ignore"):
            exponent = -2.0 * dx * dy / sigma_d_sq[np.newaxis, :]
        p_touch_bridge = np.where(both_safe, np.exp(exponent), 0.0)
        u = rng.random(p_touch_bridge.shape)
        bridge_touch = (u < p_touch_bridge) & both_safe
        touch_mask = close_touch | bridge_touch
        touch_any = touch_mask.any(axis=1)
        first_day = np.where(touch_any, touch_mask.argmax(axis=1), n_days)
        result[:, i] = first_day

    return result


def analyze_joint_conditional(
    paths: np.ndarray,
    S0: float,
    dip_price: float,
    rally_price: float,
    horizon_days: int,
    sigma: Optional[float] = None,
    vol_schedule: Optional[np.ndarray] = None,
    dip_first_days: Optional[np.ndarray] = None,
    rally_first_days: Optional[np.ndarray] = None,
) -> dict:
    """For each MC path, track whether dip and rally touched in correct order.
    Returns four-scenario breakdown summing to 1.0.

    Scenarios:
      A. round_trip:           dip then rally before horizon
      B. bag_hold:             dip touched, rally never
      C. no_trade_rally_first: rally touched before any dip
      D. neither:              never touched either barrier
    """
    n_paths, n_days = paths.shape

    if dip_first_days is not None and rally_first_days is not None:
        dip_first_day = dip_first_days
        rally_first_day = rally_first_days
    elif sigma is not None:
        dip_arr = precompute_first_touch_days(
            paths, S0, np.array([dip_price]), sigma, vol_schedule, "down"
        )
        rally_arr = precompute_first_touch_days(
            paths, S0, np.array([rally_price]), sigma, vol_schedule, "up", seed=43
        )
        dip_first_day = dip_arr[:, 0]
        rally_first_day = rally_arr[:, 0]
    else:
        dip_mask = paths <= dip_price
        rally_mask = paths >= rally_price
        dip_any_local = dip_mask.any(axis=1)
        rally_any_local = rally_mask.any(axis=1)
        dip_first_day = np.where(dip_any_local, dip_mask.argmax(axis=1), n_days)
        rally_first_day = np.where(rally_any_local, rally_mask.argmax(axis=1), n_days)

    dip_any = dip_first_day < n_days
    rally_any = rally_first_day < n_days

    both_touched = dip_any & rally_any
    round_trip = both_touched & (dip_first_day < rally_first_day)
    rally_first = (rally_any & ~dip_any) | (both_touched & (rally_first_day <= dip_first_day))
    bag_hold = dip_any & ~rally_any
    neither = ~dip_any & ~rally_any

    total = round_trip.sum() + rally_first.sum() + bag_hold.sum() + neither.sum()
    assert total == n_paths, f"scenario partition error: {total} != {n_paths}"

    n_round_trip = int(round_trip.sum())
    n_rally_first = int(rally_first.sum())
    n_bag_hold = int(bag_hold.sum())
    n_neither = int(neither.sum())

    p_dip_touched_first = float((n_round_trip + n_bag_hold) / n_paths)
    p_dip_touched_any = float(dip_any.sum() / n_paths)
    p_rally_touched_any = float(rally_any.sum() / n_paths)
    p_rally_given_dip = (
        float(n_round_trip / (n_round_trip + n_bag_hold))
        if (n_round_trip + n_bag_hold) > 0 else 0.0
    )

    if n_round_trip > 0:
        rt_dip_days = dip_first_day[round_trip]
        rt_rally_days = rally_first_day[round_trip]
        exp_days_to_dip = float(np.mean(rt_dip_days))
        exp_days_dip_to_rally = float(np.mean(rt_rally_days - rt_dip_days))
    else:
        exp_days_to_dip = 0.0
        exp_days_dip_to_rally = 0.0

    if n_bag_hold > 0:
        bag_hold_terminals = paths[bag_hold, -1]
        bag_hold_terminal_median = float(np.median(bag_hold_terminals))
    else:
        bag_hold_terminal_median = dip_price

    return {
        "n_paths": n_paths,
        "p_round_trip": n_round_trip / n_paths,
        "p_bag_hold": n_bag_hold / n_paths,
        "p_no_trade_rally_first": n_rally_first / n_paths,
        "p_neither": n_neither / n_paths,
        "p_dip_touched_marginal": p_dip_touched_first,
        "p_dip_touched_any": p_dip_touched_any,
        "p_rally_touched_any": p_rally_touched_any,
        "p_rally_given_dip_conditional": p_rally_given_dip,
        "expected_days_to_dip": exp_days_to_dip,
        "expected_days_dip_to_rally": exp_days_dip_to_rally,
        "bag_hold_terminal_median": bag_hold_terminal_median,
    }


def three_method_cross_check(
    S0: float,
    sigma: float,
    mu: float,
    horizon_days: int,
    dip_price: float,
    rally_price: float,
    mc_result: dict,
) -> dict:
    """Cross-check MC first-passage against PDE + closed-form. Returns
    agreement table and disagreement flags.
    """
    T_years = horizon_days / 252.0

    pde = pde_two_barrier(S0, rally_price, dip_price, T_years, mu, sigma)
    p_rally_first_pde = pde["p_U_first"]
    p_dip_first_pde = pde["p_L_first"]

    p_touch_dip_closed = closed_touch_down(S0, dip_price, T_years, mu, sigma)
    p_touch_rally_closed = closed_touch_up(S0, rally_price, T_years, mu, sigma)

    p_dip_first_mc = mc_result["p_bag_hold"] + mc_result["p_round_trip"]
    p_rally_first_mc = mc_result["p_no_trade_rally_first"]
    p_touch_dip_marginal_mc = mc_result.get("p_dip_touched_any", p_dip_first_mc)
    p_touch_rally_marginal_mc = mc_result.get("p_rally_touched_any",
                                               mc_result["p_no_trade_rally_first"] + mc_result["p_round_trip"])

    flags = []
    pp = lambda x: x * 100.0

    diff_dip_first = abs(pp(p_dip_first_mc) - pp(p_dip_first_pde))
    diff_rally_first = abs(pp(p_rally_first_mc) - pp(p_rally_first_pde))
    diff_touch_dip = abs(pp(p_touch_dip_marginal_mc) - pp(p_touch_dip_closed))
    diff_touch_rally = abs(pp(p_touch_rally_marginal_mc) - pp(p_touch_rally_closed))

    # σ-scaled tolerances (irreducible bridge residual grows with σ)
    tol_fp = method_tolerance_pp(sigma, "first_passage")
    tol_marg = method_tolerance_pp(sigma, "marginal")
    refuse_fp = method_refusal_pp(sigma, "first_passage")
    refuse_marg = method_refusal_pp(sigma, "marginal")

    if diff_dip_first > tol_fp:
        flags.append(f"MC vs PDE disagree on P(dip first) by {diff_dip_first:.1f}pp (tol {tol_fp:.1f})")
    if diff_rally_first > tol_fp:
        flags.append(f"MC vs PDE disagree on P(rally first) by {diff_rally_first:.1f}pp (tol {tol_fp:.1f})")
    if diff_touch_dip > tol_marg:
        flags.append(f"MC vs closed-form disagree on marginal P(touch dip) by {diff_touch_dip:.1f}pp (tol {tol_marg:.1f})")
    if diff_touch_rally > tol_marg:
        flags.append(f"MC vs closed-form disagree on marginal P(touch rally) by {diff_touch_rally:.1f}pp (tol {tol_marg:.1f})")

    # Hard-refusal gate (sacred decision #16). Triggered when any disagreement
    # exceeds the refusal threshold (1.8× the flag tolerance). Refusal means
    # the math layer can't agree on the probabilities the recommendation
    # depends on — publishing a recommendation would be irresponsible.
    refusals = []
    if diff_dip_first > refuse_fp:
        refusals.append(f"P(dip first): {diff_dip_first:.1f}pp > refuse {refuse_fp:.1f}pp")
    if diff_rally_first > refuse_fp:
        refusals.append(f"P(rally first): {diff_rally_first:.1f}pp > refuse {refuse_fp:.1f}pp")
    if diff_touch_dip > refuse_marg:
        refusals.append(f"P(touch dip): {diff_touch_dip:.1f}pp > refuse {refuse_marg:.1f}pp")
    if diff_touch_rally > refuse_marg:
        refusals.append(f"P(touch rally): {diff_touch_rally:.1f}pp > refuse {refuse_marg:.1f}pp")

    if refusals:
        status = "⛔ REFUSED — method disagreement exceeds refusal threshold"
    elif flags:
        status = "⚠ disagreement flagged (within refusal tolerance)"
    else:
        status = "✓ all methods agree within tolerance"

    return {
        "table": [
            ("P(dip first)",      pp(p_dip_first_mc),      pp(p_dip_first_pde),       diff_dip_first),
            ("P(rally first)",    pp(p_rally_first_mc),    pp(p_rally_first_pde),     diff_rally_first),
            ("P(touch dip ever)", pp(p_touch_dip_marginal_mc), pp(p_touch_dip_closed),   diff_touch_dip),
            ("P(touch rally ever)", pp(p_touch_rally_marginal_mc), pp(p_touch_rally_closed), diff_touch_rally),
        ],
        "flags": flags,
        "refusals": refusals,
        "refused": bool(refusals),
        "tolerances": {"first_passage_pp": tol_fp, "marginal_pp": tol_marg,
                       "refuse_first_passage_pp": refuse_fp, "refuse_marginal_pp": refuse_marg,
                       "sigma_used": sigma},
        "pde_p_neither": pde["p_neither"],
        "pde_mass_conservation": pde["total"],
        "agreement_status": status,
    }


# =============================================================================
# Catalyst-aware time-varying vol schedule
# =============================================================================

def build_catalyst_vol_schedule(
    base_vol: float,
    horizon_days: int,
    self_earnings_date: Optional[datetime],
    peer_earnings_dates: list,
    macro_event_dates: list,
) -> np.ndarray:
    """Return per-day vol array of length horizon_days."""
    today = datetime.now().date()
    schedule = np.ones(horizon_days)

    # PR #76: index into `schedule` is a TRADING-day offset (MC dt=1/252).
    # Was using calendar-day delta — earnings 21 trading days out (~30
    # calendar) was written to schedule[30] (= 9 trading days late),
    # corrupting Brownian-bridge fidelity (sacred #9).
    from src.market_calendar import trading_days_after as _tda

    def _trading_offset(event_date):
        """Trading-day offset from today to event_date for the MC grid.
        schedule[0] = today's vol; schedule[k] = vol on the k-th trading
        day after today. Returns None if event is before today or beyond
        horizon."""
        try:
            n = _tda(today, event_date)
        except Exception:
            return None
        if n < 0 or n >= horizon_days:
            return None
        return n

    if self_earnings_date:
        try:
            ed = self_earnings_date.date() if hasattr(self_earnings_date, "date") else self_earnings_date
            d_idx = _trading_offset(ed)
            window = VOL_SCHEDULE_MULTIPLIERS["self_earnings_window_days"]
            if d_idx is not None:
                schedule[d_idx] = max(schedule[d_idx], VOL_SCHEDULE_MULTIPLIERS["self_earnings_day"])
                for off in range(1, window + 1):
                    if d_idx - off >= 0:
                        schedule[d_idx - off] = max(
                            schedule[d_idx - off],
                            VOL_SCHEDULE_MULTIPLIERS["self_earnings_pre_post"],
                        )
                    if d_idx + off < horizon_days:
                        schedule[d_idx + off] = max(
                            schedule[d_idx + off],
                            VOL_SCHEDULE_MULTIPLIERS["self_earnings_pre_post"],
                        )
        except Exception:
            pass

    for ped in peer_earnings_dates:
        try:
            d = ped.date() if hasattr(ped, "date") else ped
            d_idx = _trading_offset(d)
            window = VOL_SCHEDULE_MULTIPLIERS["peer_earnings_window_days"]
            if d_idx is not None:
                schedule[d_idx] = max(schedule[d_idx], VOL_SCHEDULE_MULTIPLIERS["peer_earnings_day"])
                for off in range(1, window + 1):
                    if d_idx - off >= 0:
                        schedule[d_idx - off] = max(
                            schedule[d_idx - off],
                            VOL_SCHEDULE_MULTIPLIERS["peer_earnings_pre_post"],
                        )
                    if d_idx + off < horizon_days:
                        schedule[d_idx + off] = max(
                            schedule[d_idx + off],
                            VOL_SCHEDULE_MULTIPLIERS["peer_earnings_pre_post"],
                        )
        except Exception:
            pass

    for mev in macro_event_dates:
        try:
            d = mev.date() if hasattr(mev, "date") else mev
            d_idx = _trading_offset(d)
            if d_idx is not None:
                schedule[d_idx] = max(schedule[d_idx], VOL_SCHEDULE_MULTIPLIERS["macro_event_day"])
        except Exception:
            pass

    return base_vol * schedule


# =============================================================================
# Path-dependent metrics (v2's version — drawdown distribution, panic floor,
# time-to-target). v1's simpler version was dropped per W0 plan.
# =============================================================================

def compute_path_metrics(paths: np.ndarray, S0: float, dip_price: float,
                          rally_price: float, panic_floor_pct: float) -> dict:
    """Extract path-dependent statistics from MC paths.

    Returns max-drawdown distribution, panic-floor touch probability, and
    time-to-target percentiles. W3 PR #24: panic_floor_pct is per-σ-class
    (passed by the engine from SIGMA_CLASSES[class].panic_floor_pct).
    """
    n_paths, n_days = paths.shape
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (running_max - paths) / running_max
    max_dd_per_path = drawdowns.max(axis=1)

    panic_floor = S0 * (1.0 - panic_floor_pct)
    p_panic_touched = float((paths.min(axis=1) <= panic_floor).mean())

    dip_touch_day = np.where(
        (paths <= dip_price).any(axis=1),
        (paths <= dip_price).argmax(axis=1),
        -1,
    )
    rally_touch_day = np.where(
        (paths >= rally_price).any(axis=1),
        (paths >= rally_price).argmax(axis=1),
        -1,
    )
    dip_days = dip_touch_day[dip_touch_day >= 0]
    rally_days = rally_touch_day[rally_touch_day >= 0]

    return {
        "max_dd_p50": float(np.percentile(max_dd_per_path, 50)),
        "max_dd_p75": float(np.percentile(max_dd_per_path, 75)),
        "max_dd_p90": float(np.percentile(max_dd_per_path, 90)),
        "max_dd_price_p50": float(S0 * (1 - np.percentile(max_dd_per_path, 50))),
        "max_dd_price_p75": float(S0 * (1 - np.percentile(max_dd_per_path, 75))),
        "max_dd_price_p90": float(S0 * (1 - np.percentile(max_dd_per_path, 90))),
        "panic_floor_price": float(panic_floor),
        "p_panic_touched": p_panic_touched,
        "time_to_dip_p50": float(np.percentile(dip_days, 50)) if len(dip_days) else None,
        "time_to_dip_p25": float(np.percentile(dip_days, 25)) if len(dip_days) else None,
        "time_to_dip_p75": float(np.percentile(dip_days, 75)) if len(dip_days) else None,
        "time_to_rally_p50": float(np.percentile(rally_days, 50)) if len(rally_days) else None,
        "time_to_rally_p25": float(np.percentile(rally_days, 25)) if len(rally_days) else None,
        "time_to_rally_p75": float(np.percentile(rally_days, 75)) if len(rally_days) else None,
    }

"""Tests for PR #80 — audit finding #11.

`engine.run_pipeline` computed:
    mu_effective_historical = mu_capped + enr * 252 / horizon_days

where `enr = enrichment_drift(rsi, mom_5d)` is a clamped ±0.10
mean-reversion bias (RSI overbought / recent +momentum → negative).
The `252 / horizon_days` multiplier made shorter horizons produce
LARGER drift adjustments — opposite of the intuitive scaling for a
short-term mean-reversion signal. At horizon=10, a clamped 0.06 bias
becomes 1.51 annualised drift — implausible magnitude that dominates
mu_hist itself.

Fix: `apply_enrichment_to_drift` treats `enr` as an annualised drift
adjustment, adds it directly to `mu_capped`. Max additive contribution
±0.10 annual across ALL horizons. Magnitudes sensible.

Verdict-level impact: at the default horizon (60 trading days), the
pre-fix contribution from a clamped enrichment was up to ±0.42
annualised; post-fix it's ±0.10. EV bps on every BUY will drift
correspondingly on the next cycle — most affected names are those
with extreme RSI (far from 50) or strong recent 5-day momentum.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.math_utils import apply_enrichment_to_drift, enrichment_drift


def test_combined_drift_bounded_by_pm_0_10_around_mu_capped():
    """The added contribution is at most ±0.10 in absolute terms,
    regardless of horizon (the old formula could push it to ±2.5)."""
    mu_capped = 0.0
    # Maximum-bearish enrichment: RSI=100, mom_5d=+0.30 (huge 5d move).
    out_bear = apply_enrichment_to_drift(mu_capped, rsi=100.0, mom_5d=0.30)
    assert -0.10 - 1e-9 <= out_bear <= 0.0
    # Maximum-bullish enrichment: RSI=0, mom_5d=-0.30.
    out_bull = apply_enrichment_to_drift(mu_capped, rsi=0.0, mom_5d=-0.30)
    assert 0.0 <= out_bull <= 0.10 + 1e-9


def test_neutral_rsi_zero_momentum_is_pass_through():
    """RSI=50 + flat momentum → enr=0 → output equals mu_capped exactly."""
    assert apply_enrichment_to_drift(0.05, rsi=50.0, mom_5d=0.0) == 0.05
    assert apply_enrichment_to_drift(-0.20, rsi=50.0, mom_5d=0.0) == -0.20


def test_does_not_scale_with_horizon():
    """Critical regression: result MUST be independent of any horizon
    parameter. Pre-fix the call site multiplied by 252/horizon_days,
    so the same RSI/mom_5d would yield different mu_effective values
    depending on the prediction horizon — making short-horizon backtests
    incomparable with long-horizon backtests. The new helper takes no
    horizon arg, structurally preventing that bug from returning."""
    import inspect
    sig = inspect.signature(apply_enrichment_to_drift)
    # No horizon-related parameter on the helper.
    for p_name in sig.parameters:
        assert "horizon" not in p_name.lower()
    # Functional check: the result for identical (mu_capped, rsi, mom_5d)
    # is exactly stable across multiple calls.
    a = apply_enrichment_to_drift(0.10, rsi=70.0, mom_5d=0.05)
    b = apply_enrichment_to_drift(0.10, rsi=70.0, mom_5d=0.05)
    assert a == b


def test_directional_sign_consistent_with_enrichment_drift():
    """Overbought (RSI=80) + strong +momentum → enr<0 → final < mu_capped."""
    mu_capped = 0.10
    enr = enrichment_drift(rsi=80.0, mom_5d=0.10)
    assert enr < 0
    out = apply_enrichment_to_drift(mu_capped, rsi=80.0, mom_5d=0.10)
    assert out < mu_capped
    assert out == pytest.approx(mu_capped + enr)


def test_engine_uses_helper_not_old_formula():
    """Catch any future revert by checking that engine.py imports and
    uses the helper, and does NOT contain the old `enr * 252 / ...`
    expression anymore."""
    engine_path = _REPO_ROOT / "src" / "engine.py"
    src = engine_path.read_text()
    assert "apply_enrichment_to_drift" in src, \
        "engine.py must call the new helper"
    # The exact pre-fix expression must not appear (allow for prefix
    # comments referencing the old line).
    code_lines = [
        ln for ln in src.splitlines()
        if "enr * 252 / horizon_days" in ln and not ln.lstrip().startswith("#")
    ]
    assert not code_lines, (
        f"engine.py still contains the buggy multiplier line: {code_lines}"
    )

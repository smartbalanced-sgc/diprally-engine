"""Sacred decision #17 acceptance tests — YAML config behavior verification.

The defining acceptance criterion for sacred #17 is:

    Editing config/diprally.yaml and re-running changes engine behavior
    with ZERO code edits.

These tests verify that by:
  1. Writing a perturbed YAML to a temp path
  2. Calling reload_config(tmp_path)
  3. Asserting the bound module constants now reflect the perturbed values
  4. Restoring the original by reloading the canonical path

No external dependencies. Runs as `python tests/test_config_yaml.py`.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

import src.config as cfg


def _write_perturbed(perturbations: dict) -> Path:
    """Load canonical YAML, apply nested perturbations, write to a temp file.
    Returns the temp path."""
    with open(cfg.CONFIG_PATH) as f:
        raw = yaml.safe_load(f)

    def _apply(d, perts):
        for k, v in perts.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                _apply(d[k], v)
            else:
                d[k] = v

    _apply(raw, perturbations)
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="diprally_test_")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(raw, f)
    return Path(tmp_path)


def _restore():
    """Reload canonical config so subsequent tests aren't polluted."""
    cfg.reload_config(cfg.CONFIG_PATH)


# ---------- Sacred #17 acceptance tests ----------

def test_canonical_config_loads_cleanly():
    """The shipped config/diprally.yaml must validate against the pydantic
    schema. If this fails, the YAML is broken — production-blocker."""
    config = cfg.reload_config(cfg.CONFIG_PATH)
    assert config.version == "DIPNRALLY-v1.0"
    assert config.conviction.dip_marginal == 0.65
    assert config.conviction.rally_conditional == 0.75
    assert config.conviction.ev_hurdle_bps_of_dip == 50


def test_perturbing_conviction_dip_threshold_changes_constant():
    """Edit YAML conviction.dip_marginal → 0.55. Reload. Verify
    DEFAULT_CONVICTION_DIP module constant now reflects 0.55. No code edits."""
    try:
        tmp = _write_perturbed({"conviction": {"dip_marginal": 0.55}})
        cfg.reload_config(tmp)
        assert cfg.DEFAULT_CONVICTION_DIP == 0.55, \
            f"perturbed value not bound: {cfg.DEFAULT_CONVICTION_DIP}"
        # Other values must stay default
        assert cfg.DEFAULT_CONVICTION_RALLY_COND == 0.75
        assert cfg.EV_HURDLE_BPS_OF_DIP == 50
    finally:
        _restore()
        os.unlink(tmp)


def test_perturbing_ev_hurdle_changes_threshold():
    """Sacred #13 EV-hurdle threshold tunable via YAML."""
    try:
        tmp = _write_perturbed({"conviction": {"ev_hurdle_bps_of_dip": 75}})
        cfg.reload_config(tmp)
        assert cfg.EV_HURDLE_BPS_OF_DIP == 75
    finally:
        _restore()
        os.unlink(tmp)


def test_perturbing_method_tolerance_changes_helper_output():
    """method_tolerance_pp and method_refusal_pp helpers must reflect
    perturbed YAML values. This is the σ-scaled tolerance from sacred #16."""
    try:
        tmp = _write_perturbed({
            "method_tolerance": {
                "marginal_floor_pp": 5.0,        # was 2.0
                "marginal_multiplier": 6.0,      # was 3.0
                "refusal_multiplier": 2.5,       # was 1.8
            }
        })
        cfg.reload_config(tmp)
        # At σ=0.30 (low), floor 5.0 binds since 6.0 × 0.30 = 1.8 < 5.0
        assert cfg.method_tolerance_pp(0.30, "marginal") == 5.0
        # At σ=1.00, multiplier wins: 6.0 × 1.0 = 6.0 > 5.0
        assert cfg.method_tolerance_pp(1.00, "marginal") == 6.0
        # Refusal at σ=1.0 = 6.0 × 2.5 = 15.0
        assert cfg.method_refusal_pp(1.00, "marginal") == 15.0
    finally:
        _restore()
        os.unlink(tmp)


def test_perturbing_pricing_changes_dispatch_output():
    """ai_pricing tunable via YAML. pricing_for_model helper must use the
    new rates after reload."""
    try:
        tmp = _write_perturbed({
            "ai_pricing": {
                "opus_input_per_token": 0.00002,    # was 0.000015 — bumped
                "opus_output_per_token": 0.0001,    # was 0.000075 — bumped
            }
        })
        cfg.reload_config(tmp)
        in_rate, out_rate = cfg.pricing_for_model(cfg.MODEL_OPUS)
        assert in_rate == 0.00002
        assert out_rate == 0.0001
    finally:
        _restore()
        os.unlink(tmp)


def test_perturbing_blend_weights_propagates():
    """Blend weights are config — researcher must be able to tune them
    without code edits."""
    try:
        tmp = _write_perturbed({
            "blend_weights_v2": {
                "ai": 0.10,           # was 0.26 — dramatic reduction
                "historical": 0.30,   # was 0.05 — dramatic increase
                "analyst": 0.16,
                "sector": 0.04,
                "macro": 0.07,
                "short_interest": 0.02,
                "peer_rs": 0.10,
                "sector_decoupling": 0.10,
                "catalyst_proximity": 0.10,
                "narrative": 0.10,
                # sacred #15: no insider key (dropped in D-W2-16)
            }
        })
        cfg.reload_config(tmp)
        assert cfg.BLEND_WEIGHTS_V2["ai"] == 0.10
        assert cfg.BLEND_WEIGHTS_V2["historical"] == 0.30
        assert "insider" not in cfg.BLEND_WEIGHTS_V2
    finally:
        _restore()
        os.unlink(tmp)


def test_pydantic_rejects_out_of_range_values():
    """Schema validation must catch bad values. conviction.dip_marginal
    in (0,1) — a value of 1.5 must fail the load."""
    from pydantic import ValidationError
    try:
        tmp = _write_perturbed({"conviction": {"dip_marginal": 1.5}})
        try:
            cfg.reload_config(tmp)
            assert False, "Should have raised ValidationError for dip_marginal=1.5"
        except ValidationError:
            pass  # expected
    finally:
        _restore()
        os.unlink(tmp)


def test_pydantic_rejects_extra_keys():
    """Strict schema — typos in YAML keys must fail loudly, not be silently
    ignored. Adding an unexpected key 'conviction.dip_margianl' (typo)
    must fail."""
    from pydantic import ValidationError
    try:
        with open(cfg.CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        raw["conviction"]["dip_margianl"] = 0.55  # typo: margianl
        fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="diprally_test_typo_")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(raw, f)
        try:
            cfg.reload_config(Path(tmp))
            assert False, "Should have raised on typo key 'dip_margianl'"
        except ValidationError:
            pass  # expected
    finally:
        _restore()
        os.unlink(tmp)


def test_existing_imports_unchanged_after_yaml_lift():
    """Sanity: every constant other modules import from src.config must
    still resolve. This is the backwards-compatibility guarantee."""
    names = [
        "FMP_BASE", "DEFAULT_LOOKBACK_DAYS",
        "OPUS_INPUT_PER_TOKEN", "OPUS_OUTPUT_PER_TOKEN",
        "SONNET_INPUT_PER_TOKEN", "SONNET_OUTPUT_PER_TOKEN",
        "HAIKU_INPUT_PER_TOKEN", "HAIKU_OUTPUT_PER_TOKEN",
        "WEB_SEARCH_PER_USE",
        "MODEL_OPUS", "MODEL_SONNET", "MODEL_HAIKU",
        "BLEND_WEIGHTS", "BLEND_WEIGHTS_V2", "CONFIDENCE_TO_SE",
        "DEFAULT_CONVICTION_DIP", "DEFAULT_CONVICTION_RALLY_COND",
        "EV_HURDLE_BPS_OF_DIP", "ANALYST_EXTREME_DRIFT_THRESHOLD",
        "DEFAULT_HORIZON_DAYS", "DEFAULT_MC_PATHS",
        "DEEP_DIP_AUTOSCALE_THRESHOLD", "DEEP_DIP_AUTOSCALE_PATHS",
        "SIGMA_CLASSES", "SIGMA_CLASS_BOUNDARIES",
        "NARRATIVE_DRIFT_ADJUSTMENT",
        "FACTOR_WEIGHTS", "FACTOR_NET_THRESHOLD", "FACTOR_TAIL_BIAS",
        "CATALYST_Z_THRESHOLD", "VOL_SCHEDULE_MULTIPLIERS",
        "METHOD_AGREEMENT_FLOOR_PP", "METHOD_AGREEMENT_MULTIPLIER",
        "METHOD_AGREEMENT_FIRST_PASSAGE_FLOOR_PP",
        "METHOD_AGREEMENT_FIRST_PASSAGE_MULTIPLIER",
        "METHOD_REFUSAL_MULTIPLIER",
        "METHOD_AGREEMENT_TOLERANCE_PP_MARGINAL",
        "METHOD_AGREEMENT_TOLERANCE_PP_FIRST_PASSAGE",
        "BACKTEST_MIN_SAMPLES",
        "BAG_HOLD_TERMINAL_ASSUMPTION",
        "V3_REVIEW_CRITERIA", "V2_VERSION",
    ]
    for name in names:
        assert hasattr(cfg, name), f"Missing module constant after YAML lift: {name}"
        value = getattr(cfg, name)
        assert value is not None, f"{name} is None"


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in sorted(inspect.getmembers(sys.modules[__name__], inspect.isfunction)):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except (AssertionError, Exception) as e:
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            fails += 1
    if fails:
        print(f"\n{fails} test(s) failed")
        sys.exit(1)
    print("\nALL TESTS PASSED")

"""Tests for PR #90 — `tickers_scratch:` ad-hoc cohort namespace.

Contract: scratch-cohort tickers must be registered enough that an
ad-hoc `--tickers ...` run gets full registry support (peer_rs signal,
σ-class hint, sector sanity check), but `list_universe()` must NOT
include them — so default orchestrator runs (no --tickers flag)
continue to iterate the institutional `tickers:` roster unchanged.

This is the "temp list, revert to default after" requirement: drop any
future cohort into `tickers_scratch:` indefinitely; the default cohort
stays exactly what's in `tickers:`. Code change zero per cohort swap.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# 1. list_universe() unchanged by scratch entries
# =============================================================================

def test_list_universe_excludes_scratch_tickers():
    """Default orchestrator run uses list_universe(). Scratch entries must
    NOT leak in — otherwise every default cycle would AI-spend on the
    temporary cohort."""
    from src.registry import list_universe
    universe = list_universe()
    for scratch in ("VRT", "CEG", "PLTR", "SMCI", "OKLO"):
        assert scratch not in universe, (
            f"{scratch} leaked into list_universe() — default runs would "
            f"now include the temporary cohort. tickers_scratch must NOT "
            f"affect list_universe()."
        )


def test_list_universe_still_returns_institutional_roster():
    """The 26 permanent tickers must still be present."""
    from src.registry import list_universe
    universe = set(list_universe())
    institutional = {
        "LWLG", "MRAM", "ENGN", "VELO", "SNDK", "ARM", "CRWV", "NBIS",
        "INOD", "CRDO", "ANAB", "ASTS", "RKLB", "PL", "SATS", "GHM",
        "MRVL", "INTC", "IPGP", "LITE", "MU", "STX", "AMAT", "MOG-A",
        "GLW", "LRCX",
    }
    missing = institutional - universe
    assert not missing, f"Institutional roster lost tickers: {missing}"


# =============================================================================
# 2. Registry lookups FALL BACK to scratch
# =============================================================================

def test_get_ticker_finds_scratch_entries():
    """get_ticker() must resolve scratch entries (so the engine's
    σ-class reconcile and sector sanity check work on --tickers runs)."""
    from src.registry import get_ticker
    for symbol, expected_class in [
        ("VRT", "HIGH"), ("CEG", "MID"), ("PLTR", "EXTREME"),
        ("SMCI", "EXTREME"), ("OKLO", "EXTREME"),
    ]:
        cfg = get_ticker(symbol)
        assert cfg.sigma_class == expected_class


def test_resolve_peers_uses_scratch_entries():
    """peer_rs signal needs the configured peer list. Without scratch
    lookup, resolve_peers() returns [] and the signal degrades to
    _none_signal even when peers ARE configured in YAML."""
    from src.registry import resolve_peers
    assert resolve_peers("VRT") == ["ETN", "PWR"]
    assert resolve_peers("CEG") == ["VST", "TLN"]
    assert resolve_peers("PLTR") == ["NBIS", "CRWV"]
    assert resolve_peers("SMCI") == ["DELL", "ANET"]
    # OKLO has no stock_peers AND no etf_peer — graceful degradation
    assert resolve_peers("OKLO") == []


def test_classify_uses_scratch_entries():
    """σ-class reconciliation in engine.run_pipeline reads the registry
    hint. Without scratch lookup, the auto-detected class has no hint
    to reconcile against — fine functionally, but loses the audit
    mismatch flag."""
    from src.registry import classify
    assert classify("VRT") == "HIGH"
    assert classify("CEG") == "MID"
    assert classify("OKLO") == "EXTREME"


def test_expected_sector_uses_scratch_entries():
    """sector sanity check against FMP's profile.sector field needs
    the scratch entry's sector_expected string."""
    from src.registry import expected_sector
    assert expected_sector("CEG") == "Utilities / Utilities - Renewable"
    assert expected_sector("PLTR") == "Technology / Software - Infrastructure"


# =============================================================================
# 3. Institutional roster takes precedence on collision
# =============================================================================

def test_institutional_roster_wins_on_collision():
    """If a symbol appears in BOTH `tickers:` and `tickers_scratch:`,
    the institutional entry wins. (Defensive — current YAML has no
    overlap, but a future operator might paste-duplicate.)"""
    from src.registry import get_ticker
    # MU is institutional — must resolve to the institutional entry's
    # σ-class (MID), not whatever a future scratch overlay might set.
    cfg = get_ticker("MU")
    assert cfg.sigma_class == "MID"


# =============================================================================
# 4. Unknown ticker still raises TickerNotInUniverse
# =============================================================================

def test_unknown_ticker_raises():
    """A ticker that's in NEITHER namespace must raise the typed
    exception so callers can produce a useful error message."""
    from src.registry import get_ticker, TickerNotInUniverse
    import pytest
    with pytest.raises(TickerNotInUniverse):
        get_ticker("XYZZY_NOT_A_TICKER")

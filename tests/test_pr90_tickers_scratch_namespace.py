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
    for scratch in ("ADBE", "ORCL", "DE", "IBM", "AVGO"):
        assert scratch not in universe, (
            f"{scratch} leaked into list_universe() — default runs would "
            f"now include the temporary cohort. tickers_scratch must NOT "
            f"affect list_universe()."
        )


def test_list_universe_returns_active_cohort():
    """The active daily roster — currently the 5-name iteration cohort
    (2026-05-30 cull, pending 0-BUY clearance before re-expansion).
    Originally 26; the 21 sidelined names live in tickers_scratch with
    full metadata so re-promotion is a YAML edit only (sacred #17)."""
    from src.registry import list_universe
    universe = set(list_universe())
    active_cohort = {"LWLG", "ARM", "RKLB", "MU", "AMAT"}
    missing = active_cohort - universe
    assert not missing, f"Active cohort lost tickers: {missing}"
    # Active roster must NOT contain sidelined names — that would mean
    # the cull was reverted in code, not via YAML.
    sidelined_extreme = {
        "MRAM", "ENGN", "VELO", "SNDK", "CRWV", "NBIS",
        "INOD", "CRDO", "ANAB",
    }
    sidelined_high = {"ASTS", "PL", "SATS", "GHM", "MRVL"}
    sidelined_mid = {
        "INTC", "IPGP", "LITE", "STX", "MOG-A", "GLW", "LRCX",
    }
    leaked = (sidelined_extreme | sidelined_high | sidelined_mid) & universe
    assert not leaked, (
        f"Sidelined tickers leaked back into list_universe(): {leaked}"
    )


def test_sidelined_tickers_still_resolvable_via_scratch():
    """Cull preserves metadata: every sidelined name must still resolve
    via get_ticker() so an explicit `--tickers SNDK` run gets full peer
    + σ-class + sector support. Nothing lost, only relocated."""
    from src.registry import get_ticker
    sidelined = (
        "MRAM", "ENGN", "VELO", "SNDK", "CRWV", "NBIS",
        "INOD", "CRDO", "ANAB",
        "ASTS", "PL", "SATS", "GHM", "MRVL",
        "INTC", "IPGP", "LITE", "STX", "MOG-A", "GLW", "LRCX",
    )
    for symbol in sidelined:
        cfg = get_ticker(symbol)
        assert cfg.sigma_class in ("MID", "HIGH", "EXTREME"), (
            f"sidelined {symbol} lost σ-class metadata"
        )


# =============================================================================
# 2. Registry lookups FALL BACK to scratch
# =============================================================================

def test_get_ticker_finds_scratch_entries():
    """get_ticker() must resolve scratch entries (so the engine's
    σ-class reconcile and sector sanity check work on --tickers runs)."""
    from src.registry import get_ticker
    for symbol in ("ADBE", "ORCL", "DE", "IBM", "AVGO"):
        cfg = get_ticker(symbol)
        # Current cohort is all MID σ-class hints; the test guards the
        # lookup mechanism, not the specific σ-class — that's a
        # cohort-selection choice that changes per validation batch.
        assert cfg.sigma_class in ("MID", "HIGH", "EXTREME")


def test_resolve_peers_uses_scratch_entries():
    """peer_rs signal needs the configured peer list. Without scratch
    lookup, resolve_peers() returns [] and the signal degrades to
    _none_signal even when peers ARE configured in YAML."""
    from src.registry import resolve_peers
    # Spot-check a few: every cohort ticker must yield SOMETHING (peers
    # configured) or [] (etf_peer empty) — never error.
    for symbol in ("ADBE", "ORCL", "DE", "IBM", "AVGO"):
        peers = resolve_peers(symbol)
        assert isinstance(peers, list)
        assert len(peers) > 0, f"{symbol} configured peers should resolve"


def test_classify_uses_scratch_entries():
    """σ-class reconciliation in engine.run_pipeline reads the registry
    hint. Without scratch lookup, the auto-detected class has no hint
    to reconcile against — fine functionally, but loses the audit
    mismatch flag."""
    from src.registry import classify
    for symbol in ("ADBE", "ORCL", "DE", "IBM", "AVGO"):
        assert classify(symbol) in ("MID", "HIGH", "EXTREME")


def test_expected_sector_uses_scratch_entries():
    """sector sanity check against FMP's profile.sector field needs
    the scratch entry's sector_expected string."""
    from src.registry import expected_sector
    for symbol in ("ADBE", "ORCL", "DE", "IBM", "AVGO"):
        sector = expected_sector(symbol)
        assert sector is not None and len(sector) > 0


# =============================================================================
# 3. Institutional roster takes precedence on collision
# =============================================================================

def test_institutional_roster_wins_on_collision():
    """If a symbol appears in BOTH `tickers:` and `tickers_scratch:`,
    the institutional entry wins. (Defensive — current YAML has no
    overlap, but a future operator might paste-duplicate.)"""
    from src.registry import get_ticker
    # MU is institutional — must resolve to the institutional entry's
    # σ-class (HIGH, as of PR #92 Defect G fix), not whatever a future
    # scratch overlay might set.
    cfg = get_ticker("MU")
    assert cfg.sigma_class == "HIGH"


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

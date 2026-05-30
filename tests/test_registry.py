"""Tests for src/registry.py — the per-ticker registry (D-W2-1).

Sacred decision #17 + universe-is-config: the 17-ticker roster is data in
config/diprally.yaml, not code. These tests verify:

  - All current universe tickers load correctly
  - Peer resolution follows the documented order (stock_peers → etf_peer → [])
  - σ-class hints match CLAUDE.md
  - Unknown tickers raise informative errors (don't silently corrupt)
  - Case-insensitive lookups work (operator-friendly)
  - No SNDK hardcode anywhere (sacred #4)
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.registry import (
    TickerNotInUniverse,
    classify,
    expected_sector,
    get_ticker,
    list_universe,
    resolve_peers,
)


# ---------- Universe membership ----------

def test_universe_has_5_tickers():
    """2026-05-30 cull: active daily roster reduced from 26 → 5 representative
    names until 0-BUY is reproducibly cleared on the iteration cohort. 21
    sidelined names live in tickers_scratch with full metadata. NOT a hard
    constraint on universe size (universe is config) — just a check that the
    YAML hasn't drifted from the documented roster without an explicit update."""
    universe = list_universe()
    assert len(universe) == 5, f"Universe drifted: {len(universe)} tickers"


def test_extreme_class_membership():
    """EXTREME-class active roster (2026-05-30 cull): LWLG (photonics
    narrative, thin peer data) + ARM (semi-IP, rich AI catalysts). The
    9 sidelined EXTREME names live in tickers_scratch."""
    extreme = {t for t in list_universe() if classify(t) == "EXTREME"}
    assert extreme == {"LWLG", "ARM"}, extreme


def test_high_class_membership():
    """HIGH-class active roster (2026-05-30 cull): MU (HIGH override per
    PR #92) + RKLB (aerospace/space, non-semi factor for portfolio gate)."""
    high = {t for t in list_universe() if classify(t) == "HIGH"}
    assert high == {"MU", "RKLB"}, high


def test_mid_class_membership():
    """MID-class active roster (2026-05-30 cull): AMAT only — the σ=0.40
    probe for MID conviction joint-reachability + low-σ EV math."""
    mid = {t for t in list_universe() if classify(t) == "MID"}
    assert mid == {"AMAT"}, mid


def test_class_counts_sum_to_universe():
    """Every ticker has exactly one class assignment."""
    universe = list_universe()
    extreme = sum(1 for t in universe if classify(t) == "EXTREME")
    high = sum(1 for t in universe if classify(t) == "HIGH")
    mid = sum(1 for t in universe if classify(t) == "MID")
    assert extreme + high + mid == len(universe)


# ---------- Peer resolution ----------

def test_resolve_peers_uses_stock_peers_when_available():
    """INTC has stock_peers [AMD, AVGO] per registry → return them."""
    peers = resolve_peers("INTC")
    assert peers == ["AMD", "AVGO"]


def test_resolve_peers_falls_back_to_etf_for_extreme_names():
    """EXTREME-class names lacking stock peers fall back to thematic ETF.
    MRAM → SOXX per registry."""
    assert resolve_peers("MRAM") == ["SOXX"]
    assert resolve_peers("LWLG") == ["SOXX"]
    assert resolve_peers("ENGN") == ["XBI"]
    assert resolve_peers("VELO") == ["PPA"]


def test_resolve_peers_handles_mog_dash_a():
    """Dot-vs-dash ticker convention: MOG-A is canonical (per CLAUDE.md +
    PR #6 finding). MOG.A would not resolve."""
    assert resolve_peers("MOG-A") == ["HEI", "TDG", "CW"]
    assert resolve_peers("MOG.A") == []  # not in universe


def test_resolve_peers_unknown_ticker_returns_empty():
    """Unknown tickers return empty peers. Caller can override with
    --peers explicitly. (SNDK was promoted to the universe in the
    2026-05-24 expansion — use a random unknown symbol instead.)"""
    assert resolve_peers("RANDOM123") == []
    assert resolve_peers("ZZZNOTREAL") == []


def test_resolve_peers_case_insensitive():
    """Operator-friendly: 'mram' and 'MRAM' resolve identically."""
    assert resolve_peers("mram") == resolve_peers("MRAM")
    assert resolve_peers("Intc") == resolve_peers("INTC")


# ---------- get_ticker + classify + expected_sector ----------

def test_get_ticker_returns_full_entry():
    """get_ticker exposes the full TickerConfig for downstream consumers
    (W3 σ-class system, W5 batch orchestrator, etc.)."""
    entry = get_ticker("INTC")
    assert entry.sigma_class == "MID"
    assert "Technology" in entry.sector_expected
    assert entry.stock_peers == ["AMD", "AVGO"]
    assert entry.etf_peer == ""


def test_get_ticker_raises_for_unknown():
    """get_ticker raises TickerNotInUniverse with an informative message
    — different from resolve_peers which silently returns []."""
    try:
        get_ticker("UNKNOWN_TICKER")
        assert False, "should have raised"
    except TickerNotInUniverse as e:
        msg = str(e)
        assert "UNKNOWN_TICKER" in msg
        assert "config/diprally.yaml" in msg


def test_classify_returns_none_for_unknown():
    """classify silently returns None — caller can branch on it."""
    assert classify("UNKNOWN") is None


def test_expected_sector_returns_none_for_unknown():
    assert expected_sector("UNKNOWN") is None


def test_expected_sector_matches_known_tickers():
    """Sanity check on the YAML's sector_expected fields."""
    assert "Aerospace" in expected_sector("RKLB")
    assert "Semiconductors" in expected_sector("INTC")
    assert "Biotechnology" in expected_sector("ENGN")


# ---------- Sacred #4 (no SNDK-specific hardcodes) ----------

def test_sndk_resolves_through_registry_only():
    """Sacred #4: 'No SNDK-specific hardcodes.' SNDK lives in
    tickers_scratch as of the 2026-05-30 cull (sidelined from the active
    daily roster pending 0-BUY clearance), but must still resolve
    through the same generic registry path as any other ticker. Sacred
    #4 forbids HARDCODES in code, not presence in YAML — and the
    scratch-fallback path (PR #90) is exactly the generic mechanism."""
    peers = resolve_peers("SNDK")
    # SNDK's YAML entry (now under tickers_scratch:) declares stock_peers
    assert peers == ["MU", "STX", "WDC"]


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

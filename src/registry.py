"""Per-ticker registry. Sourced from `config/diprally.yaml` via src.config.

The universe is data, not code (sacred #17 + universe-is-config). Adding or
removing tickers is a YAML edit, never a code change.

Public API:
    get_ticker(symbol) -> TickerConfig
    resolve_peers(symbol) -> list[str]
    list_universe() -> list[str]
    classify(symbol) -> str   # "EXTREME" / "HIGH" / "MID"

Closes D-W2-1 (SNDK peer-fallback shim). engine.run_pipeline and tools/run.py
both call resolve_peers() instead of carrying the W0 hardcoded ["MU", "WDC"]
fallback for SNDK only.
"""
from __future__ import annotations

from typing import Optional

from src.config import TickerConfig, _CONFIG


class TickerNotInUniverse(KeyError):
    """Raised when a ticker is not in the registered universe.

    Lets the caller distinguish 'unknown ticker' from generic KeyError so
    tools/run.py can produce a useful error: 'Ticker XYZ not in universe.
    Add it to config/diprally.yaml or pass --peers explicitly.'
    """


def get_ticker(symbol: str) -> TickerConfig:
    """Look up a ticker's registry entry. Raises TickerNotInUniverse if not
    in the loaded YAML."""
    upper = symbol.upper()
    if upper not in _CONFIG.tickers:
        raise TickerNotInUniverse(
            f"{upper} not in universe. Edit config/diprally.yaml to add, "
            f"or pass --peers explicitly to override registry."
        )
    return _CONFIG.tickers[upper]


def resolve_peers(symbol: str) -> list[str]:
    """Return the peer list for a ticker.

    Resolution order:
      1. stock_peers if non-empty
      2. [etf_peer] if etf_peer is non-empty (single-ETF anchor for EXTREME
         names that have no comparable stock peers)
      3. [] empty list — signal_from_peer_rs returns _none_signal cleanly

    Tickers not in the universe return [] (caller is responsible for either
    adding the ticker to YAML or passing --peers explicitly). Letting an
    unknown ticker silently fall through to empty peers is safer than
    raising — the engine's other 10 signals still work.
    """
    upper = symbol.upper()
    entry = _CONFIG.tickers.get(upper)
    if entry is None:
        return []
    if entry.stock_peers:
        return list(entry.stock_peers)
    if entry.etf_peer:
        return [entry.etf_peer]
    return []


def list_universe() -> list[str]:
    """Return all ticker symbols in the loaded universe, sorted."""
    return sorted(_CONFIG.tickers.keys())


def classify(symbol: str) -> Optional[str]:
    """Return the σ-class hint for a ticker (EXTREME/HIGH/MID), or None if
    not in the universe.

    σ-class is a HINT — W3's auto-detection will compute the actual class
    from realized vol + GARCH. The registry value is the starting point /
    fallback when the auto-detector lacks data.
    """
    upper = symbol.upper()
    entry = _CONFIG.tickers.get(upper)
    return entry.sigma_class if entry else None


def expected_sector(symbol: str) -> Optional[str]:
    """Return the expected sector string from the registry. Used as a
    sanity check against FMP's profile.sector field."""
    upper = symbol.upper()
    entry = _CONFIG.tickers.get(upper)
    return entry.sector_expected if entry else None


def provider_symbol(symbol: str, provider: str) -> str:
    """Translate a canonical ticker to the form required by a specific data
    provider. Sacred decision #17: per-provider translation is data in
    config/diprally.yaml.

    provider must be 'fmp' or 'yfinance' (case-insensitive). Returns the
    canonical (caller-passed) symbol if:
      - the ticker is not in the registered universe (caller responsibility)
      - the registry entry has an empty fmp_symbol / yf_symbol field
        (default — most tickers use the same symbol on both providers)

    Today's universe uses dash form (MOG-A) on both FMP and yfinance, so no
    overrides are configured. The mechanism is in place for future tickers
    where providers diverge (e.g. BRK.B on FMP vs BRK-B on Yahoo).
    """
    upper = symbol.upper()
    entry = _CONFIG.tickers.get(upper)
    if entry is None:
        return upper
    provider_lower = provider.lower()
    if provider_lower == "fmp":
        return entry.fmp_symbol or upper
    if provider_lower in ("yfinance", "yf"):
        return entry.yf_symbol or upper
    raise ValueError(f"Unknown provider {provider!r}; expected 'fmp' or 'yfinance'")

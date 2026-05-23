"""σ-class auto-detection (W3).

Classifies a ticker's σ-class (EXTREME / HIGH / MID) from its blended σ
estimate. Boundaries live in config/diprally.yaml under
sigma_class_boundaries (sacred #17).

Design decisions (per W3 plan):

  1. Live σ wins for current run; registry hint is audit trail (surfaced
     in report when mismatched). Data > config.

  2. Classification feeds on `blended_sigma` (the 4-anchor σ triangulation),
     NOT `effective_sigma` (post AI vol_regime multiplier). σ-class is a
     STRUCTURAL property of the ticker; AI vol_regime is a tactical
     adjustment. Mixing them creates chicken-and-egg with class-specific
     ai_vol_regime multipliers in later W3 PRs.

  3. Boundary semantics: >= for the lower bound. σ exactly at extreme_min
     classifies as EXTREME (the more conservative side).

  4. GARCH-fit failure path: when blended σ can't be computed, fall back
     to MID-class (most conservative thresholds, tightest grid, lowest
     friction — minimizes false-authorize risk). Caller passes σ=None.
"""
from __future__ import annotations

from typing import Optional

from src.config import SIGMA_CLASS_BOUNDARIES, SIGMA_CLASSES
from src.registry import classify as registry_classify


def classify_sigma(blended_sigma: Optional[float]) -> str:
    """Classify a ticker's σ-class from its blended σ. Returns one of
    "EXTREME", "HIGH", "MID". On σ=None (GARCH failure), returns "MID"
    (conservative fallback).
    """
    if blended_sigma is None or blended_sigma <= 0:
        return "MID"
    if blended_sigma >= SIGMA_CLASS_BOUNDARIES.extreme_min:
        return "EXTREME"
    if blended_sigma >= SIGMA_CLASS_BOUNDARIES.high_min:
        return "HIGH"
    return "MID"


def class_conviction(sigma_class: str) -> tuple[float, float]:
    """Return (dip_threshold, rally_conditional_threshold) for a σ-class.

    Class names must be one of EXTREME/HIGH/MID; KeyError on unknown.
    The class table validates non-empty at config load time.
    """
    entry = SIGMA_CLASSES[sigma_class]
    return entry.conviction.dip, entry.conviction.rally_conditional


def reconcile_with_registry(
    symbol: str, auto_class: str
) -> tuple[str, Optional[str]]:
    """Compare auto-detected class with registry hint. Returns:

        (effective_class, mismatch_note)

    effective_class is always auto_class (data wins for current run).
    mismatch_note is None on match, or a human-readable string on mismatch
    that the reporter can surface as an advisory.

    A registry miss (ticker not in universe) is treated as no-mismatch —
    the auto-detector is the only authority and the trader either adds
    the ticker to the registry or accepts that no audit trail exists yet.
    """
    registry_hint = registry_classify(symbol)
    if registry_hint is None:
        return auto_class, None
    if registry_hint == auto_class:
        return auto_class, None
    return (
        auto_class,
        f"σ-class mismatch — auto-detected {auto_class}, registry hint "
        f"{registry_hint}. Auto wins for this run; review registry if "
        f"the structural class has shifted.",
    )

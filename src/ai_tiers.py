"""AI tier resolution (W4 PR #27).

Translates a tier name ("T0" / "T1" / "T2" / "T3") into a concrete
dispatch spec the engine can consume: model IDs for each pass, token
caps, web_search bounds. Lives between the YAML schema (typed via
AITierConfig) and the engine's call sites (which need ready-to-use
model IDs, not abstract keys).

Sacred CLAUDE.md tier ladder:
  T0: math only, $0.00
  T1: Haiku Pass 1 only, ~$0.02
  T2: Sonnet Pass 1 + Sonnet Pass 2, ~$0.10
  T3: Opus Pass 1 + Sonnet Pass 2 + Haiku stress, ~$0.30

Single-ticker CLI defaults to T3 (preserves pre-W4 behavior). Multi-
ticker orchestrator (W5) will read tier assignments from the broker
(W4 PR #29) under the $2/day hard cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import AI_TIERS, MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET


@dataclass(frozen=True)
class ResolvedTier:
    """Concrete dispatch spec — engine reads these fields directly.

    A None model field means "skip this pass entirely." pass2_model is
    None for T1 (Pass 1 only, no critique). stress_model is None for
    T0/T1/T2 (only T3 runs catalyst stress).
    """
    name: str
    pass1_model: Optional[str]
    pass2_model: Optional[str]
    stress_model: Optional[str]
    pass1_web_search_max: int
    pass1_max_tokens: int
    pass2_max_tokens: int
    estimated_cost_usd: float

    @property
    def runs_ai(self) -> bool:
        """True for any tier above T0 — at least one model is configured."""
        return self.pass1_model is not None


_MODEL_LOOKUP = {
    "opus": MODEL_OPUS,
    "sonnet": MODEL_SONNET,
    "haiku": MODEL_HAIKU,
}


def _resolve_model(key: Optional[str]) -> Optional[str]:
    if key is None:
        return None
    if key not in _MODEL_LOOKUP:
        raise KeyError(
            f"Unknown ai_models key {key!r} (expected one of "
            f"{set(_MODEL_LOOKUP)} or null)"
        )
    return _MODEL_LOOKUP[key]


def resolve_tier(tier_name: str) -> ResolvedTier:
    """Look up a tier by name and resolve its model keys to concrete IDs.

    Raises KeyError if tier_name is not in AI_TIERS, or if a model key
    inside the tier doesn't resolve. Config-load validation catches the
    latter at startup; this guard catches in-process tampering.
    """
    if tier_name not in AI_TIERS:
        raise KeyError(
            f"Unknown AI tier {tier_name!r} (expected one of {sorted(AI_TIERS)})"
        )
    spec = AI_TIERS[tier_name]
    return ResolvedTier(
        name=tier_name,
        pass1_model=_resolve_model(spec.pass1_model),
        pass2_model=_resolve_model(spec.pass2_model),
        stress_model=_resolve_model(spec.stress_model),
        pass1_web_search_max=spec.pass1_web_search_max,
        pass1_max_tokens=spec.pass1_max_tokens,
        pass2_max_tokens=spec.pass2_max_tokens,
        estimated_cost_usd=spec.estimated_cost_usd,
    )


def t0() -> ResolvedTier:
    """Math-only tier — convenience shorthand for the --no-ai path."""
    return resolve_tier("T0")

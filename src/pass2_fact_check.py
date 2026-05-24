"""Pass 2 fact-discipline programmatic enforcer (PR #63).

PR #40 added a FACT DISCIPLINE block to the Pass 2 prompt that
instructs the model not to invent numeric facts (prices,
financials) outside Pass 1's JSON or the math layer's outputs.
That's prompt-level mitigation. The smoke evidence (INTC v3
audit, 2026-05-23) showed Pass 2 still occasionally invents facts:

  Pass 2 critique: "INTC's current price is far below $119 and
                    this figure is inconsistent..."
  Actual spot:     $119.84
  → Pass 2 hallucinated a contradictory price-level claim.

PR #63 adds programmatic ENFORCEMENT — a post-Pass-2 validator
that parses the critique text, extracts dollar mentions made in
price/spot contexts, and flags ones that diverge from the
math-layer ground truth.

Phase 1 (this PR): DETECT and FLAG only. Operator sees the
violation in the report; can apply skepticism. Future PR #64+
(post-data) will decide enforcement actions (auto-strip,
confidence downgrade, etc.) based on real violation patterns.

Detection strategy: regex for dollar mentions in price-context
phrases. Pure regex avoids LLM-on-LLM validation cost spiral.
False positives accepted (Pass 2 legitimately citing historical
price points, 52w high/low, etc.) — operator sees the flag with
context, judges intent. False negatives accepted (subtle
fact-hallucinations without dollar signs) — those wait for the
calibration analysis loop to surface patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Tolerance for "near spot" — anything within this fraction of spot
# is plausibly a legitimate price reference (target / 52w range / etc).
# Outside this band → hallucination flag.
SPOT_TOLERANCE_FRACTION = 0.50  # 50% in either direction

# Phrases that strongly suggest "this is the CURRENT spot" — when a
# dollar amount follows one of these and diverges from actual spot,
# it's a near-certain hallucination.
_SPOT_CONTEXT_PHRASES = [
    r"current price",
    r"spot price",
    r"spot of",
    r"current spot",
    r"trades at",
    r"trading at",
    r"is now at",
    r"current level",
    r"current level of",
    r"is currently",
    r"spot \$",
]

# Regex to extract dollar amounts from text. Handles "$119", "$1,500",
# "$119.84", "$1.5B" (but B-suffixed amounts go through a different
# code path since they're market-cap-class, not price-class).
_DOLLAR_RE = re.compile(
    r"\$\s*(\d[\d,]*(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class FactViolation:
    """One detected fact-hallucination in Pass 2's critique."""
    kind: str                # "spot_contradiction" / "outlier_price"
    claimed_value: float     # the $XXX from Pass 2
    expected_value: float    # what the math layer says (e.g. spot)
    expected_label: str      # "spot" / "dip target" / etc.
    divergence_pct: float    # |claimed - expected| / expected
    context: str             # ~40 chars surrounding the claim in critique


def _extract_dollar_mentions(text: str) -> list[tuple[float, int]]:
    """Pull all dollar amounts from text. Returns (value, position).
    Skips B/M/K suffixed amounts (market cap / revenue scale, not price).
    Filters out integers that look like years (1900-2100)."""
    if not text:
        return []
    out = []
    for m in _DOLLAR_RE.finditer(text):
        try:
            val = float(m.group(1).replace(",", ""))
        except (ValueError, TypeError):
            continue
        # Skip B/M/K suffixed amounts — market-cap / revenue scale, not price.
        tail = text[m.end():m.end() + 2].lstrip()
        if tail and tail[0] in "BMKbmk":
            continue
        # Skip likely-year values that happen to be prefixed with $.
        if 1900 <= val <= 2100 and "." not in m.group(1):
            continue
        out.append((val, m.start()))
    return out


def _find_spot_context_dollars(text: str) -> list[tuple[float, int]]:
    """Subset of dollar mentions that appear within 60 chars after a
    spot-context phrase. These are the high-confidence hallucination
    candidates."""
    if not text:
        return []
    lower = text.lower()
    spot_anchored = []
    for phrase in _SPOT_CONTEXT_PHRASES:
        for m in re.finditer(phrase, lower):
            # Look for a $ amount within 60 chars AFTER the phrase.
            window_start = m.end()
            window_end = min(len(text), window_start + 60)
            window = text[window_start:window_end]
            for amount, pos in _extract_dollar_mentions(window):
                spot_anchored.append((amount, window_start + pos))
    return spot_anchored


def _context_snippet(text: str, position: int, half_width: int = 40) -> str:
    """Pull ~80 chars around position for the violation report."""
    if not text:
        return ""
    start = max(0, position - half_width)
    end = min(len(text), position + half_width)
    snippet = text[start:end].replace("\n", " ").strip()
    return snippet


def validate_pass2_critique(critique_text: str, spot: float,
                              tolerance_fraction: float = SPOT_TOLERANCE_FRACTION
                              ) -> list[FactViolation]:
    """PR #63 — validate Pass 2's critique text against ground-truth
    spot price. Returns list of violations.

    Detection passes:
      1. Spot-context: dollar mentions following "current price", "spot of",
         etc. compared against actual spot. Any divergence > 20% flagged
         as 'spot_contradiction' (high confidence).
      2. Outlier price: any standalone $XXX > 2× or < 0.5× of spot
         flagged as 'outlier_price' (lower confidence — could be
         legitimate historical / fair-value citation, operator judges).

    Empty critique / zero spot → empty list (defensive, no false
    positives on degraded inputs).
    """
    violations: list[FactViolation] = []
    if not critique_text or not spot or spot <= 0:
        return violations

    # Pass 1: high-confidence spot contradictions.
    seen_positions = set()
    for amount, pos in _find_spot_context_dollars(critique_text):
        divergence = abs(amount - spot) / spot
        if divergence > 0.20:  # ≥20% off from actual spot is contradiction
            violations.append(FactViolation(
                kind="spot_contradiction",
                claimed_value=amount,
                expected_value=spot,
                expected_label="spot",
                divergence_pct=divergence * 100.0,
                context=_context_snippet(critique_text, pos),
            ))
            seen_positions.add(pos)

    # Pass 2: outlier-price scan (lower confidence).
    lower_bound = spot * (1 - tolerance_fraction)
    upper_bound = spot * (1 + tolerance_fraction)
    for amount, pos in _extract_dollar_mentions(critique_text):
        if pos in seen_positions:
            continue  # already flagged as spot_contradiction
        # Outlier if outside [50% spot, 150% spot] AND > 2× spot OR < 0.5× spot.
        # The intermediate band (50-150% of spot) is mostly legitimate
        # price-target citations; only flag the gross outliers.
        if amount > spot * 2.0 or amount < spot * 0.5:
            divergence = abs(amount - spot) / spot
            violations.append(FactViolation(
                kind="outlier_price",
                claimed_value=amount,
                expected_value=spot,
                expected_label="spot (loose tolerance)",
                divergence_pct=divergence * 100.0,
                context=_context_snippet(critique_text, pos),
            ))

    return violations


def format_violations(violations: list[FactViolation]) -> str:
    """Render violations for the per-ticker report. Operator-readable
    — shows each violation with context so trader can judge whether
    Pass 2 actually hallucinated or legitimately cited a far price."""
    if not violations:
        return ""
    lines = []
    lines.append(f"⚠ Pass 2 FACT-DISCIPLINE: {len(violations)} potential violation(s) detected")
    for v in violations:
        if v.kind == "spot_contradiction":
            lines.append(
                f"  ⛔ HIGH-CONF SPOT CONTRADICTION: claimed ${v.claimed_value:,.2f} "
                f"vs actual spot ${v.expected_value:,.2f} ({v.divergence_pct:.0f}% off)"
            )
        else:
            lines.append(
                f"  ⓘ outlier price: ${v.claimed_value:,.2f} vs spot "
                f"${v.expected_value:,.2f} ({v.divergence_pct:.0f}% off) — verify context"
            )
        lines.append(f"     context: \"...{v.context}...\"")
    return "\n".join(lines)

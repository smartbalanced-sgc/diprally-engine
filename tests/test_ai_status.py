"""Tests for Defect B — AI-delivery status (_compute_ai_status).

When an intended AI call fails (network / 429 / timeout / missing key /
unparseable response), the engine previously swallowed the exception and
computed a verdict on math-only signals while the report still claimed an
AI tier ran. `ai_status` is an orthogonal flag (it does NOT overwrite
verdict_state) that records whether the intended drift pipeline delivered:

  OK         — everything intended ran, OR no AI intended (T0/--no-ai), OR
               a same-day cache replay served it.
  DEGRADED   — Pass 1 ran but the intended Pass 2 critique failed; sacred #7
               ("Pass 2 wins") violated, Pass 1 drift used unrevised.
  INCOMPLETE — Pass 1 failed; verdict ran math-only despite an AI tier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ai_layer import call_ai_pass
from src.ai_tiers import resolve_tier, t0
from src.engine import _compute_ai_status, AI_STATUSES, CSV_COLUMNS

_T3 = resolve_tier("T3")   # Pass 1 + Pass 2 (+ stress + verification)
_T1 = resolve_tier("T1")   # Pass 1 only — pass2_model is None
_T0 = t0()                 # math only — runs_ai False

_FAILED = ("error", "no_client", "empty")


# ---------------------------------------------------------------------------
# OK paths
# ---------------------------------------------------------------------------

def test_t0_is_always_ok():
    """Math-only tier never intended AI — not a failure."""
    assert _compute_ai_status(
        tier=_T0, cache_hit=False,
        pass1_status="skipped", pass2_status="skipped",
    ) == "OK"


def test_cache_hit_is_ok_even_on_ai_tier():
    """Same-day cache replay served the AI outputs — nothing failed."""
    assert _compute_ai_status(
        tier=_T3, cache_hit=True,
        pass1_status="skipped", pass2_status="skipped",
    ) == "OK"


def test_both_passes_ok():
    assert _compute_ai_status(
        tier=_T3, cache_hit=False,
        pass1_status="ok", pass2_status="ok",
    ) == "OK"


def test_t1_pass1_ok_no_pass2_intended_is_ok():
    """T1 has no Pass 2 (pass2_model None) — pass2 'skipped' is expected,
    not a degradation."""
    assert _compute_ai_status(
        tier=_T1, cache_hit=False,
        pass1_status="ok", pass2_status="skipped",
    ) == "OK"


# ---------------------------------------------------------------------------
# INCOMPLETE — Pass 1 failed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p1", _FAILED)
def test_pass1_failure_is_incomplete(p1):
    assert _compute_ai_status(
        tier=_T3, cache_hit=False,
        pass1_status=p1, pass2_status="skipped",
    ) == "INCOMPLETE"


def test_pass1_failure_dominates_pass2(p1="error"):
    """If Pass 1 failed, INCOMPLETE wins even if pass2_status is also bad."""
    assert _compute_ai_status(
        tier=_T3, cache_hit=False,
        pass1_status="error", pass2_status="error",
    ) == "INCOMPLETE"


def test_t1_pass1_failure_is_incomplete():
    assert _compute_ai_status(
        tier=_T1, cache_hit=False,
        pass1_status="no_client", pass2_status="skipped",
    ) == "INCOMPLETE"


# ---------------------------------------------------------------------------
# DEGRADED — Pass 1 ok, intended Pass 2 failed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p2", _FAILED)
def test_pass1_ok_pass2_failure_is_degraded(p2):
    assert _compute_ai_status(
        tier=_T3, cache_hit=False,
        pass1_status="ok", pass2_status=p2,
    ) == "DEGRADED"


def test_t1_never_degrades_on_pass2_status():
    """A tier with no Pass 2 cannot be DEGRADED by a pass2_status value —
    it was never going to run Pass 2."""
    assert _compute_ai_status(
        tier=_T1, cache_hit=False,
        pass1_status="ok", pass2_status="error",
    ) == "OK"


# ---------------------------------------------------------------------------
# Schema + contract
# ---------------------------------------------------------------------------

def test_ai_status_column_present():
    assert "ai_status" in CSV_COLUMNS


def test_every_status_in_documented_set():
    seen = {
        _compute_ai_status(tier=_T0, cache_hit=False,
                           pass1_status="skipped", pass2_status="skipped"),
        _compute_ai_status(tier=_T3, cache_hit=False,
                           pass1_status="error", pass2_status="skipped"),
        _compute_ai_status(tier=_T3, cache_hit=False,
                           pass1_status="ok", pass2_status="error"),
    }
    assert seen == {"OK", "INCOMPLETE", "DEGRADED"}
    assert set(AI_STATUSES) == {"OK", "DEGRADED", "INCOMPLETE"}


# ---------------------------------------------------------------------------
# call_ai_pass return contract — status is the 4th element
# ---------------------------------------------------------------------------

def test_call_ai_pass_no_client_returns_failed_status(monkeypatch):
    """No API key → no client → ('no_client', $0). The 4-tuple is what
    lets the engine tell AI-unavailable apart from a real math-only run."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    parsed, cost, sources, status = call_ai_pass("prompt", pass_label="T")
    assert parsed is None
    assert cost == 0.0
    assert sources == 0
    assert status == "no_client"

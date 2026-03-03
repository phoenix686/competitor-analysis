# tests/test_simulation.py
"""
Deterministic tests for the SimulationHarness.

These tests do NOT call the LLM or any external API — they verify that:
  1. inject_adversarial_signals correctly structures contradictory signal pairs
  2. _route_after_analysis routes to reflection_agent when adversarial
     signals cause low-quality analysis output
  3. generate_golden_cases returns correctly-shaped dicts (LLM mocked)
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from harness.simulation import SimulationHarness
from graph.workflow import _route_after_analysis

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BASE_STATE = {
    "competitors": ["blinkit", "zepto", "zomato"],
    "raw_signals": [],
    "analyzed_signals": [],
    "errors": [],
    "reflection_count": 0,
}


@pytest.fixture
def harness():
    return SimulationHarness()


# ---------------------------------------------------------------------------
# Test 1: inject_adversarial_signals structure
# ---------------------------------------------------------------------------

def test_inject_adds_signals(harness):
    """Injection must add at least one signal pair to raw_signals."""
    state = harness.inject_adversarial_signals(BASE_STATE)
    assert len(state["raw_signals"]) >= 4, (
        "Expected at least 4 adversarial signals (2 contradictory pairs)"
    )


def test_inject_preserves_existing_signals(harness):
    """Injection must not discard existing raw_signals."""
    existing = [{"competitor": "zomato", "signal_type": "news",
                 "description": "existing", "confidence": 7.0}]
    state_with = {**BASE_STATE, "raw_signals": list(existing)}
    result = harness.inject_adversarial_signals(state_with)
    assert any(s["description"] == "existing" for s in result["raw_signals"]), (
        "Existing signals must be preserved after injection"
    )


def test_inject_produces_contradictory_pairs(harness):
    """Each pair must contain two signals about the same competitor."""
    state = harness.inject_adversarial_signals(BASE_STATE)
    injected = state["raw_signals"]

    # Group by competitor
    by_competitor: dict[str, list] = {}
    for sig in injected:
        comp = sig["competitor"]
        by_competitor.setdefault(comp, []).append(sig)

    # At least one competitor should have 2 signals (the contradictory pair)
    assert any(len(sigs) >= 2 for sigs in by_competitor.values()), (
        "Expected at least one competitor with contradictory signal pair"
    )


def test_inject_does_not_mutate_input(harness):
    """inject_adversarial_signals must not modify the original state dict."""
    original_len = len(BASE_STATE["raw_signals"])
    harness.inject_adversarial_signals(BASE_STATE)
    assert len(BASE_STATE["raw_signals"]) == original_len, (
        "Original state must not be mutated"
    )


# ---------------------------------------------------------------------------
# Test 2: reflection_agent triggers on adversarial analysis output
# ---------------------------------------------------------------------------

def test_reflection_triggers_on_vague_actions():
    """
    When adversarial contradictory signals lead to vague recommended_actions
    (under 30 chars for >30% of signals), reflection_agent must be triggered.
    """
    # Simulate poor-quality analysis that adversarial signals would likely produce:
    # - Contradictory claims → analyst hedges → vague actions
    poor_analyzed = [
        {"recommended_action": "Monitor", "assessment": "NEUTRAL"},       # 7 chars — vague
        {"recommended_action": "Watch closely", "assessment": "NEUTRAL"},  # 13 chars — vague
        {"recommended_action": "Investigate further", "assessment": "NEUTRAL"},  # 19 chars — vague
        {"recommended_action": "Consider options", "assessment": "THREAT"},  # 16 chars — vague
    ]
    state = {
        **BASE_STATE,
        "analyzed_signals": poor_analyzed,
        "reflection_count": 0,
    }
    route = _route_after_analysis(state)
    assert route == "reflection_agent", (
        f"Expected reflection_agent for >30% vague actions, got {route!r}"
    )


def test_reflection_skipped_when_already_reflected():
    """Once reflection_count >= 1 the router must go straight to brief_writer."""
    poor_analyzed = [
        {"recommended_action": "Monitor", "assessment": "NEUTRAL"},
        {"recommended_action": "Watch", "assessment": "NEUTRAL"},
    ]
    state = {
        **BASE_STATE,
        "analyzed_signals": poor_analyzed,
        "reflection_count": 1,   # already reflected
    }
    route = _route_after_analysis(state)
    assert route == "brief_writer", (
        f"Expected brief_writer after reflection pass, got {route!r}"
    )


def test_reflection_triggers_on_zero_signals():
    """Zero analyzed signals must always trigger reflection (first pass only)."""
    state = {**BASE_STATE, "analyzed_signals": [], "reflection_count": 0}
    assert _route_after_analysis(state) == "reflection_agent"


def test_no_reflection_on_quality_output():
    """Good-quality analysis with specific actions must bypass reflection."""
    good_analyzed = [
        {
            "recommended_action": "Launch a 30-day delivery-fee waiver in Delhi to match Blinkit",
            "assessment": "THREAT",
        },
        {
            "recommended_action": "Run targeted push notifications to Zepto churners in Bangalore",
            "assessment": "OPPORTUNITY",
        },
    ]
    state = {**BASE_STATE, "analyzed_signals": good_analyzed, "reflection_count": 0}
    assert _route_after_analysis(state) == "brief_writer"


# ---------------------------------------------------------------------------
# Test 3: generate_golden_cases (LLM mocked)
# ---------------------------------------------------------------------------

def test_generate_golden_cases_structure(harness):
    """generate_golden_cases must return a list of dicts with required keys."""
    mock_case = {
        "id": "tc_sim_001",
        "description": "Blinkit fee cut — THREAT to Delhi",
        "input": {
            "raw_signals": [{
                "competitor": "blinkit",
                "signal_type": "pricing",
                "description": "Blinkit cuts fee",
                "source": "techcrunch.com",
                "raw_evidence": "Blinkit zero fee Delhi",
                "confidence": 8.5,
            }]
        },
        "expected": {
            "assessment": "THREAT",
            "min_impact_score": 7,
            "reasoning_must_mention": ["delhi", "fee"],
        },
    }

    mock_response = MagicMock()
    mock_response.content = json.dumps([mock_case])

    state = {
        **BASE_STATE,
        "analyzed_signals": [{
            "competitor": "blinkit",
            "assessment": "THREAT",
            "impact_score": 9,
            "recommended_action": "Launch fee waiver in Delhi within 48 hours",
            "reasoning": "Blinkit fee cut threatens Delhi market share",
            "affected_markets": ["delhi"],
            "signal_type": "pricing",
            "description": "Blinkit cuts delivery fee",
            "confidence": 8.5,
        }],
    }

    # Pydantic v2 blocks instance-level setattr — patch at class level instead
    from langchain_groq import ChatGroq
    with patch.object(ChatGroq, "invoke", return_value=mock_response):
        cases = harness.generate_golden_cases(state)

    assert isinstance(cases, list), "generate_golden_cases must return a list"
    assert len(cases) == 1
    case = cases[0]
    assert "id" in case
    assert "input" in case
    assert "expected" in case
    assert "raw_signals" in case["input"]
    assert "assessment" in case["expected"]


def test_generate_golden_cases_empty_state(harness):
    """Empty analyzed_signals must return an empty list without crashing."""
    cases = harness.generate_golden_cases({**BASE_STATE, "analyzed_signals": []})
    assert cases == []

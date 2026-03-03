# tests/test_graph_replay.py
"""
Integration test for the full CompeteIQ graph using fixture data.
No live API calls — runs in seconds, fully deterministic.

Run with:
    uv run pytest tests/test_graph_replay.py -v
"""
import pytest
from harness.replay import ReplayHarness
from graph.workflow import competeiq_graph

INITIAL_STATE = {
    "messages": [],
    "competitors": ["blinkit", "zepto", "zomato"],  # subset for speed
    "raw_signals": [],
    "analyzed_signals": [],
    "final_brief": "",
    "run_id": "",
    "errors": [],
}


@pytest.fixture
def replay():
    return ReplayHarness()


def test_graph_produces_raw_signals(replay):
    """Signal collector must extract at least 1 raw signal from fixture data."""
    with replay.patch_tools():
        result = competeiq_graph.invoke(INITIAL_STATE)

    assert isinstance(result["raw_signals"], list), "raw_signals must be a list"
    assert len(result["raw_signals"]) >= 1, (
        f"Expected >= 1 raw signal, got {len(result['raw_signals'])}"
    )


def test_graph_produces_analyzed_signals(replay):
    """Analysis agent must produce at least 1 analyzed signal."""
    with replay.patch_tools():
        result = competeiq_graph.invoke(INITIAL_STATE)

    assert isinstance(result["analyzed_signals"], list)
    assert len(result["analyzed_signals"]) >= 1, (
        f"Expected >= 1 analyzed signal, got {len(result['analyzed_signals'])}"
    )


def test_analyzed_signals_have_required_fields(replay):
    """Every analyzed signal must have the required schema fields."""
    required_fields = {
        "competitor", "signal_type", "assessment",
        "affected_markets", "impact_score", "recommended_action", "confidence"
    }
    with replay.patch_tools():
        result = competeiq_graph.invoke(INITIAL_STATE)

    for signal in result["analyzed_signals"]:
        missing = required_fields - set(signal.keys())
        assert not missing, f"Signal missing fields: {missing}\n  Signal: {signal}"


def test_graph_produces_final_brief(replay):
    """Brief writer must produce a non-empty final brief."""
    with replay.patch_tools():
        result = competeiq_graph.invoke(INITIAL_STATE)

    brief = result.get("final_brief", "")
    assert isinstance(brief, str) and len(brief) > 100, (
        f"Expected a non-trivial brief, got: {repr(brief[:80])}"
    )


def test_brief_contains_priority_sections(replay):
    """Brief must contain the expected section headers."""
    with replay.patch_tools():
        result = competeiq_graph.invoke(INITIAL_STATE)

    brief = result["final_brief"]
    assert "HIGH PRIORITY" in brief or "MEDIUM PRIORITY" in brief or "OPPORTUNITIES" in brief, (
        "Brief is missing expected section headers"
    )


def test_assessment_values_are_valid(replay):
    """Assessment field must only contain THREAT, OPPORTUNITY, or NEUTRAL."""
    valid_assessments = {"THREAT", "OPPORTUNITY", "NEUTRAL"}
    with replay.patch_tools():
        result = competeiq_graph.invoke(INITIAL_STATE)

    for signal in result["analyzed_signals"]:
        assessment = signal.get("assessment")
        assert assessment in valid_assessments, (
            f"Invalid assessment value: {assessment!r}"
        )


def test_impact_scores_in_valid_range(replay):
    """Impact scores must be between 1 and 10."""
    with replay.patch_tools():
        result = competeiq_graph.invoke(INITIAL_STATE)

    for signal in result["analyzed_signals"]:
        score = signal.get("impact_score")
        assert isinstance(score, int), f"impact_score must be int, got {type(score)}"
        assert 1 <= score <= 10, f"impact_score out of range: {score}"

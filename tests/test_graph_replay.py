# tests/test_graph_replay.py
"""
Integration test for the full CompeteIQ graph using fixture data.
Tools are replaced with pre-recorded fixtures (no live Tavily/Play Store/Adzuna).
The graph runs ONCE per session (module-scoped fixture) to stay within
Groq's 12k TPM rate limit; all 7 tests share that single result.

Run with:
    uv run pytest tests/test_graph_replay.py -v
"""
import asyncio
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
    "memory_context": "",
    "momentum_summary": "",
    # Phase 3: parallel fan-out state
    "competitor_signals": {},
    "retry_count": 0,
    "tool_call_count": 0,
    "current_competitor": "",
    # Phase 5: self-correction
    "reflection_count": 0,
    # Phase 6: per-node latency tracking
    "node_latencies": {},
}


@pytest.fixture(scope="module")
def graph_result():
    """
    Run the full graph exactly once and share the result across all tests.
    module scope keeps us within Groq's TPM limit.
    Uses ainvoke() to match the async nodes in workflow.py.
    """
    harness = ReplayHarness()
    with harness.patch_tools():
        return asyncio.run(competeiq_graph.ainvoke(INITIAL_STATE))


def test_graph_produces_raw_signals(graph_result):
    """Signal collector must extract at least 1 raw signal from fixture data."""
    assert isinstance(graph_result["raw_signals"], list), "raw_signals must be a list"
    assert len(graph_result["raw_signals"]) >= 1, (
        f"Expected >= 1 raw signal, got {len(graph_result['raw_signals'])}"
    )


def test_graph_produces_analyzed_signals(graph_result):
    """Analysis agent must produce at least 1 analyzed signal."""
    assert isinstance(graph_result["analyzed_signals"], list)
    assert len(graph_result["analyzed_signals"]) >= 1, (
        f"Expected >= 1 analyzed signal, got {len(graph_result['analyzed_signals'])}"
    )


def test_analyzed_signals_have_required_fields(graph_result):
    """Every analyzed signal must have the required schema fields."""
    required_fields = {
        "competitor", "signal_type", "assessment",
        "affected_markets", "impact_score", "recommended_action", "confidence"
    }
    for signal in graph_result["analyzed_signals"]:
        missing = required_fields - set(signal.keys())
        assert not missing, f"Signal missing fields: {missing}\n  Signal: {signal}"


def test_graph_produces_final_brief(graph_result):
    """Brief writer must produce a non-empty final brief."""
    brief = graph_result.get("final_brief", "")
    assert isinstance(brief, str) and len(brief) > 100, (
        f"Expected a non-trivial brief, got: {repr(brief[:80])}"
    )


def test_brief_contains_priority_sections(graph_result):
    """Brief must contain the expected section headers."""
    brief = graph_result["final_brief"]
    assert "HIGH PRIORITY" in brief or "MEDIUM PRIORITY" in brief or "OPPORTUNITIES" in brief, (
        "Brief is missing expected section headers"
    )


def test_assessment_values_are_valid(graph_result):
    """Assessment field must only contain THREAT, OPPORTUNITY, or NEUTRAL."""
    valid_assessments = {"THREAT", "OPPORTUNITY", "NEUTRAL"}
    for signal in graph_result["analyzed_signals"]:
        assessment = signal.get("assessment")
        assert assessment in valid_assessments, (
            f"Invalid assessment value: {assessment!r}"
        )


def test_impact_scores_in_valid_range(graph_result):
    """Impact scores must be between 1 and 10."""
    for signal in graph_result["analyzed_signals"]:
        score = signal.get("impact_score")
        assert isinstance(score, int), f"impact_score must be int, got {type(score)}"
        assert 1 <= score <= 10, f"impact_score out of range: {score}"

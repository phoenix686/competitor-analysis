# graph/workflow.py
import asyncio
import uuid
import json
from langsmith import traceable
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agents.state import CompeteIQState
from agents.llm import llm
from agents.schema import RawSignalList, AnalyzedSignalList
from skills.mcp_client import get_mcp_tools
from config import COMPETITORS, MIN_CONFIDENCE_SCORE
from utils.observability import traced_node
from utils.guard import filter_hallucinated_signals
from prompts.constants import (
    ORCHESTRATOR_SYSTEM,
    COLLECTOR_EXTRACT_SYSTEM,
    ANALYSIS_SYSTEM,
    REFLECTION_SYSTEM,
    BRIEF_SYSTEM,
)
from memory.episodic import get_recent_runs, format_episodic_context
from memory.semantic import retrieve_similar, format_semantic_context, upsert_signals
from memory.context_manager import get_last_n_runs, compress_to_momentum_summary

# Structured output LLMs
llm_collector = llm.with_structured_output(RawSignalList)
llm_analyzer = llm.with_structured_output(AnalyzedSignalList)


# ---------------------------------------------------------------------------
# NODE 1: Orchestrator
# ---------------------------------------------------------------------------
@traceable(name="orchestrator", run_type="chain", tags=["competeiq"])
@traced_node("orchestrator")
def orchestrator(state: CompeteIQState) -> dict:
    # Pull last 5 run summaries from Redis for context priming
    recent_runs = get_recent_runs(5)
    memory_ctx = format_episodic_context(recent_runs)

    system = SystemMessage(content=ORCHESTRATOR_SYSTEM)
    human = HumanMessage(content=(
        f"Start competitive monitoring run for: {state['competitors']}\n\n"
        f"{memory_ctx}"
    ))
    response = llm.invoke([system, human])
    return {
        "messages": [response],
        "run_id": str(uuid.uuid4())[:8],
        "memory_context": memory_ctx,
        "tool_call_count": 0,
    }


# ---------------------------------------------------------------------------
# NODE 1b: Context Manager  (runs after orchestrator, before fan-out)
# ---------------------------------------------------------------------------
@traceable(name="context_manager", run_type="chain", tags=["competeiq"])
@traced_node("context_manager")
def context_manager(_state: CompeteIQState) -> dict:
    """
    Pulls last 10 runs from Redis and distils them into a Market Momentum
    summary via LLM. The summary is injected into both the collector extraction
    prompt and the analysis prompt so downstream nodes are aware of trends.
    """
    runs = get_last_n_runs(10)
    momentum = compress_to_momentum_summary(runs)
    return {"momentum_summary": momentum}


# ---------------------------------------------------------------------------
# NODE 2a: Collector Sub-Agent  (one instance per competitor via Send())
# ---------------------------------------------------------------------------
@traceable(name="collector_sub_agent", run_type="chain", tags=["competeiq"])
@traced_node("collector_sub_agent")
async def collector_sub_agent(state: CompeteIQState) -> dict:
    """
    Handles signal collection for exactly ONE competitor.
    Spawned in parallel by _fan_out_collectors via LangGraph Send().

    Runs all 3 tools concurrently via asyncio.gather() + asyncio.to_thread()
    then does a single structured-extraction LLM call.

    Returns:
      competitor_signals — merged by _merge_competitor_signals reducer
      tool_call_count   — accumulated with operator.add reducer
    """
    competitor = state.get("current_competitor", "unknown")

    # Resolve MCP (or fallback) tools — cached after first call
    tools = await get_mcp_tools()
    tool_map = {t.name: t for t in tools}
    search_tool  = tool_map["search_competitor"]
    reviews_tool = tool_map["get_app_reviews"]
    jobs_tool    = tool_map["get_competitor_jobs"]

    # Run all 3 data sources concurrently via MCP ainvoke
    search_q = f"{competitor} pricing delivery fee changes India 2026"
    raw_results = await asyncio.gather(
        search_tool.ainvoke(search_q),
        reviews_tool.ainvoke({"competitor": competitor}),
        jobs_tool.ainvoke({"company": competitor.capitalize()}),
        return_exceptions=True,
    )

    labels = ["Web search", "App reviews", "Jobs"]
    parts = []
    for label, result in zip(labels, raw_results):
        text = str(result) if not isinstance(result, Exception) else f"error: {result}"
        parts.append(f"{label}:\n{text[:500]}")
    all_tool_output = "\n---\n".join(parts)

    momentum = state.get("momentum_summary", "No historical momentum data available.")
    structured_response = await llm_collector.ainvoke([
        SystemMessage(content=COLLECTOR_EXTRACT_SYSTEM.format(
            min_confidence=MIN_CONFIDENCE_SCORE,
            momentum_summary=momentum,
        )),
        HumanMessage(content=(
            f"Research collected for {competitor}:\n{all_tool_output[:3000]}"
        )),
    ])

    signals = [s.model_dump() for s in structured_response.signals]

    return {
        "competitor_signals": {competitor: signals},
        "tool_call_count": 3,
    }


# ---------------------------------------------------------------------------
# NODE 2b: Aggregator  (merges all sub-agent results, applies confidence gate)
# ---------------------------------------------------------------------------
@traceable(name="aggregator", run_type="chain", tags=["competeiq"])
@traced_node("aggregator")
def aggregator(state: CompeteIQState) -> dict:
    """
    Runs after all collector_sub_agent nodes complete.
    - Flattens competitor_signals into raw_signals
    - Drops signals below MIN_CONFIDENCE_SCORE (confidence gate)
    - Records which competitors yielded zero signals (drives retry logic)
    """
    competitor_signals = state.get("competitor_signals", {})
    all_signals: list[dict] = []
    zero_coverage: list[str] = []

    for competitor in state["competitors"]:
        signals = competitor_signals.get(competitor, [])
        gated = [
            s for s in signals
            if float(s.get("confidence", 0)) >= MIN_CONFIDENCE_SCORE
        ]
        if not gated:
            zero_coverage.append(competitor)
        all_signals.extend(gated)

    errors = list(state.get("errors", []))
    if zero_coverage:
        errors.append(
            f"Zero signals after confidence gate for: {', '.join(zero_coverage)}"
        )

    return {
        "raw_signals": all_signals,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# NODE 2c: Retry Collector  (increments retry_count, resets for fresh pass)
# ---------------------------------------------------------------------------
@traceable(name="retry_collector", run_type="chain", tags=["competeiq"])
@traced_node("retry_collector")
def retry_collector(state: CompeteIQState) -> dict:
    """
    Called when at least one competitor yielded zero signals and retry_count < 2.
    - Increments retry_count
    - Resets competitor_signals via __reset__ sentinel (see state.py reducer)
    - Injects a critique HumanMessage so sub-agents know to dig deeper
    """
    zero_coverage = [
        c for c in state["competitors"]
        if not state.get("competitor_signals", {}).get(c)
    ]
    critique = (
        f"Retry {state.get('retry_count', 0) + 1}: 0 confident signals found for "
        f"{', '.join(zero_coverage)}. "
        "Look harder — check pricing changes, delivery fee updates, "
        "expansion announcements, or recent hiring trends."
    )
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        # __reset__ sentinel wipes competitor_signals before the fresh fan-out
        "competitor_signals": {"__reset__": True},
        "messages": [HumanMessage(content=critique)],
    }


# ---------------------------------------------------------------------------
# NODE 3: Analysis Agent
# ---------------------------------------------------------------------------
@traceable(name="analysis_agent", run_type="chain", tags=["competeiq"])
@traced_node("analysis_agent")
async def analysis_agent(state: CompeteIQState) -> dict:
    if not state["raw_signals"]:
        return {"analyzed_signals": []}

    # Build a short query from the first 3 signal descriptions for semantic lookup
    sample_descs = " ".join(
        s.get("description", "") for s in state["raw_signals"][:3]
    )
    similar = await asyncio.to_thread(retrieve_similar, sample_descs, 3)
    semantic_ctx = format_semantic_context(similar)

    # Truncate raw signals to keep total input under 4000 tokens
    raw_json = json.dumps(state["raw_signals"], indent=2)[:3000]

    momentum = state.get("momentum_summary", "No historical momentum data available.")
    structured_response = await llm_analyzer.ainvoke([
        SystemMessage(content=ANALYSIS_SYSTEM.format(momentum_summary=momentum)),
        HumanMessage(content=(
            f"Historical context:\n{semantic_ctx}\n\n"
            f"Analyze these signals for SwiftMart:\n{raw_json}"
        )),
    ])

    analyzed_signals = [s.model_dump() for s in structured_response.signals]

    # Hallucination guard: remove signals with invalid competitors or assessments
    clean_signals, guard_issues = filter_hallucinated_signals(
        analyzed_signals, state["competitors"]
    )

    errors = list(state.get("errors", [])) + guard_issues

    # Persist to ChromaDB in the background — brief_writer doesn't wait for it
    run_id = state.get("run_id", "unknown")
    asyncio.create_task(asyncio.to_thread(upsert_signals, clean_signals, run_id))

    return {"analyzed_signals": clean_signals, "errors": errors}


# ---------------------------------------------------------------------------
# NODE 3a: Reflection Agent  (Phase 5: self-correction on low-quality analysis)
# ---------------------------------------------------------------------------
@traceable(name="reflection_agent", run_type="chain", tags=["competeiq"])
@traced_node("reflection_agent")
async def reflection_agent(state: CompeteIQState) -> dict:
    """
    Triggered when analysis quality is poor (0 signals or vague actions).
    Re-invokes the LLM with a targeted critique to improve the output.
    Runs at most once per pipeline execution (reflection_count < 1).
    """
    analyzed = state.get("analyzed_signals", [])

    # Build a specific critique
    critiques: list[str] = []
    if not analyzed:
        critiques.append(
            "The previous analysis produced 0 signals — re-analyze and extract "
            "at least one signal from the raw data."
        )
    else:
        vague = [
            s.get("recommended_action", "")
            for s in analyzed
            if len(s.get("recommended_action", "")) < 30
        ]
        if vague:
            critiques.append(
                f"{len(vague)} of {len(analyzed)} signals have vague recommended_actions "
                f"(under 30 chars). Examples: {vague[:2]}. Make each action specific and "
                "actionable (e.g. 'Launch Delhi delivery-fee match campaign within 48 hours')."
            )
        avg_conf = sum(s.get("confidence", 0) for s in analyzed) / len(analyzed)
        if avg_conf < 4.0:
            critiques.append(
                f"Average confidence is {avg_conf:.1f}/10 — signals appear poorly grounded. "
                "Only report signals supported by concrete evidence."
            )

    critique_text = "\n".join(critiques) or "General quality improvement pass."

    raw_json = json.dumps(state["raw_signals"], indent=2)[:3000]

    structured_response = await llm_analyzer.ainvoke([
        SystemMessage(content=REFLECTION_SYSTEM.format(critique=critique_text)),
        HumanMessage(content=f"Re-analyze these signals for SwiftMart:\n{raw_json}"),
    ])

    improved = [s.model_dump() for s in structured_response.signals]

    # Apply hallucination guard to the improved signals too
    clean_signals, guard_issues = filter_hallucinated_signals(
        improved, state["competitors"]
    )

    run_id = state.get("run_id", "unknown")
    asyncio.create_task(asyncio.to_thread(upsert_signals, clean_signals, run_id))

    errors = list(state.get("errors", [])) + guard_issues

    return {
        "analyzed_signals": clean_signals,
        "reflection_count": state.get("reflection_count", 0) + 1,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# NODE 4: Brief Writer
# ---------------------------------------------------------------------------
@traceable(name="brief_writer", run_type="chain", tags=["competeiq"])
@traced_node("brief_writer")
def brief_writer(state: CompeteIQState) -> dict:
    response = llm.invoke([
        SystemMessage(content=BRIEF_SYSTEM.format(
            run_id=state.get("run_id", "unknown"),
            competitors=state["competitors"]
        )),
        HumanMessage(content=(
            f"Write brief from these analyzed signals:\n"
            f"{json.dumps(state['analyzed_signals'], indent=2)}"
        ))
    ])
    return {
        "messages": [response],
        "final_brief": response.content,
    }


# ---------------------------------------------------------------------------
# EDGE FUNCTIONS
# ---------------------------------------------------------------------------

def _fan_out_collectors(state: CompeteIQState) -> list[Send]:
    """orchestrator → collector_sub_agent (×N in parallel)."""
    return [
        Send("collector_sub_agent", {**state, "current_competitor": c})
        for c in state["competitors"]
    ]


def _route_after_aggregator(state: CompeteIQState) -> str:
    """Retry collection if any competitor has 0 signals and retries remain."""
    competitor_signals = state.get("competitor_signals", {})
    retry_count = state.get("retry_count", 0)
    zero_coverage = [
        c for c in state["competitors"]
        if not competitor_signals.get(c)
    ]
    if zero_coverage and retry_count < 2:
        return "retry_collector"
    return "analysis_agent"


def _fan_out_after_retry(state: CompeteIQState) -> list[Send]:
    """retry_collector → collector_sub_agent (×N in parallel)."""
    return [
        Send("collector_sub_agent", {**state, "current_competitor": c})
        for c in state["competitors"]
    ]


def _route_after_analysis(state: CompeteIQState) -> str:
    """
    Phase 5: reflect if quality is poor AND we haven't reflected yet.
    Quality is poor when:
      - 0 analyzed signals, OR
      - >50% of signals have vague recommended_actions (< 30 chars)
    Max one reflection pass (reflection_count < 1).
    """
    analyzed = state.get("analyzed_signals", [])
    reflection_count = state.get("reflection_count", 0)

    if reflection_count >= 1:
        return "brief_writer"

    if not analyzed:
        return "reflection_agent"

    vague_count = sum(
        1 for s in analyzed if len(s.get("recommended_action", "")) < 30
    )
    if vague_count > len(analyzed) * 0.3:
        return "reflection_agent"

    return "brief_writer"


# ---------------------------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(CompeteIQState)

    # Register nodes
    graph.add_node("orchestrator",        orchestrator)
    graph.add_node("context_manager",     context_manager)
    graph.add_node("collector_sub_agent", collector_sub_agent)
    graph.add_node("aggregator",          aggregator)
    graph.add_node("retry_collector",     retry_collector)
    graph.add_node("analysis_agent",      analysis_agent)
    graph.add_node("reflection_agent",    reflection_agent)
    graph.add_node("brief_writer",        brief_writer)

    # Linear backbone
    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "context_manager")
    graph.add_edge("reflection_agent", "brief_writer")
    graph.add_edge("brief_writer", END)

    # Fan-out: context_manager → parallel sub-agents (momentum injected into state)
    graph.add_conditional_edges(
        "context_manager",
        _fan_out_collectors,
        ["collector_sub_agent"],
    )

    # Sub-agents converge at aggregator
    graph.add_edge("collector_sub_agent", "aggregator")

    # After aggregator: retry collection or proceed to analysis
    graph.add_conditional_edges(
        "aggregator",
        _route_after_aggregator,
        ["retry_collector", "analysis_agent"],
    )

    # Retry fan-out loop
    graph.add_conditional_edges(
        "retry_collector",
        _fan_out_after_retry,
        ["collector_sub_agent"],
    )

    # After analysis: reflect if quality poor, else write brief
    graph.add_conditional_edges(
        "analysis_agent",
        _route_after_analysis,
        ["reflection_agent", "brief_writer"],
    )

    return graph.compile()


competeiq_graph = build_graph()

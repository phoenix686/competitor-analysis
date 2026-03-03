# graph/workflow.py
import uuid
import json
from langsmith import traceable
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agents.state import CompeteIQState
from agents.llm import llm
from agents.schema import RawSignalList, AnalyzedSignalList
from skills.tools import search_competitor, get_app_reviews, get_competitor_jobs
from config import COMPETITORS, MIN_CONFIDENCE_SCORE
from utils.observability import traced_node
from prompts.constants import (
    ORCHESTRATOR_SYSTEM,
    COLLECTOR_SYSTEM,
    COLLECTOR_EXTRACT_SYSTEM,
    ANALYSIS_SYSTEM,
    BRIEF_SYSTEM,
)

# Tools setup
tools = [search_competitor, get_app_reviews, get_competitor_jobs]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# Structured output LLMs
llm_collector = llm.with_structured_output(RawSignalList)
llm_analyzer = llm.with_structured_output(AnalyzedSignalList)


# NODE 1: Orchestrator
@traceable(name="orchestrator", run_type="chain", tags=["competeiq"])
@traced_node("orchestrator")
def orchestrator(state: CompeteIQState) -> CompeteIQState:
    system = SystemMessage(content=ORCHESTRATOR_SYSTEM)
    human = HumanMessage(
        content=f"Start competitive monitoring run for: {state['competitors']}"
    )
    response = llm.invoke([system, human])
    return {
        **state,
        "messages": [response],
        "run_id": str(uuid.uuid4())[:8]
    }


# NODE 2: Signal Collector
# Agentic tool-calling loop
@traceable(name="signal_collector", run_type="chain", tags=["competeiq"])
@traced_node("signal_collector")
def signal_collector(state: CompeteIQState) -> CompeteIQState:
    system = SystemMessage(
        content=COLLECTOR_SYSTEM.format(competitors=state["competitors"])
    )
    human = HumanMessage(
        content=f"Collect all signals for: {state['competitors']}"
    )
    messages = [system, human]

    # Agentic tool-calling loop
    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tool_results = tool_node.invoke({"messages": messages})
        messages.extend(tool_results["messages"])

        # Guardrail: max 8 tool calls
        tool_call_count = sum(
            1 for m in messages
            if hasattr(m, "tool_calls") and m.tool_calls
        )
        if tool_call_count >= 8:
            messages.append(HumanMessage(
                content="You have gathered enough data. Stop calling tools now."
            ))
            break

    # Extract structured signals from everything collected
    tool_outputs = []
    for m in messages:
        if hasattr(m, "content") and m.content and hasattr(m, "type") and "tool" in str(type(m).__name__).lower():
            tool_outputs.append(str(m.content)[:500])  # truncate each tool result

    all_tool_output = "\n---\n".join(tool_outputs)

    structured_response = llm_collector.invoke([
        SystemMessage(content=COLLECTOR_EXTRACT_SYSTEM.format(
            min_confidence=MIN_CONFIDENCE_SCORE
        )),
        HumanMessage(content=f"Research collected:\n{all_tool_output[:4000]}")
    ])

    raw_signals = [s.model_dump() for s in structured_response.signals]

    return {
        **state,
        "messages": messages,
        "raw_signals": raw_signals
    }


# NODE 3: Analysis Agent
@traceable(name="analysis_agent", run_type="chain", tags=["competeiq"])
@traced_node("analysis_agent")
def analysis_agent(state: CompeteIQState) -> CompeteIQState:
    if not state["raw_signals"]:
        return {**state, "analyzed_signals": []}

    structured_response = llm_analyzer.invoke([
        SystemMessage(content=ANALYSIS_SYSTEM),
        HumanMessage(content=f"""
Analyze these signals for SwiftMart:
{json.dumps(state['raw_signals'], indent=2)}
""")
    ])

    analyzed_signals = [s.model_dump() for s in structured_response.signals]

    return {
        **state,
        "analyzed_signals": analyzed_signals
    }


# NODE 4: Brief Writer
@traceable(name="brief_writer", run_type="chain", tags=["competeiq"])
@traced_node("brief_writer")
def brief_writer(state: CompeteIQState) -> CompeteIQState:
    response = llm.invoke([
        SystemMessage(content=BRIEF_SYSTEM.format(
            run_id=state.get("run_id", "unknown"),
            competitors=state["competitors"]
        )),
        HumanMessage(content=f"""
Write brief from these analyzed signals:
{json.dumps(state['analyzed_signals'], indent=2)}
""")
    ])

    return {
        **state,
        "messages": [response],
        "final_brief": response.content
    }


# BUILD GRAPH
def build_graph():
    graph = StateGraph(CompeteIQState)

    graph.add_node("orchestrator", orchestrator)
    graph.add_node("signal_collector", signal_collector)
    graph.add_node("analysis_agent", analysis_agent)
    graph.add_node("brief_writer", brief_writer)

    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "signal_collector")
    graph.add_edge("signal_collector", "analysis_agent")
    graph.add_edge("analysis_agent", "brief_writer")
    graph.add_edge("brief_writer", END)

    return graph.compile()


competeiq_graph = build_graph()

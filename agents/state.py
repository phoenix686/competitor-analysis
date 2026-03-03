# agents/state.py
import operator
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

def _merge_node_latencies(left: dict, right: dict) -> dict:
    """Reducer for node_latencies — merges timing dicts from parallel nodes."""
    return {**(left or {}), **(right or {})}


def _merge_competitor_signals(left: dict, right: dict) -> dict:
    """
    Reducer for competitor_signals during parallel fan-out.

    Each collector_sub_agent returns {"competitor_signals": {"blinkit": [...]}}
    and this reducer merges all competitors' results together.

    Special case: returning {"__reset__": True} clears the entire dict.
    retry_collector uses this to wipe results before a fresh retry cycle.
    """
    if right.get("__reset__"):
        return {}
    merged = dict(left)
    for k, v in right.items():
        if k == "__reset__":
            continue
        merged[k] = merged.get(k, []) + list(v)
    return merged


class CompeteIQState(TypedDict):
    # Conversation messages (auto-merged by LangGraph)
    messages: Annotated[list[BaseMessage], add_messages]

    # Which competitors to monitor this run
    competitors: list[str]

    # Raw signals collected by Signal Collector (set by aggregator)
    raw_signals: list[dict]

    # Analyzed signals with impact scores from Analysis Agent
    analyzed_signals: list[dict]

    # Final brief from Brief Writer
    final_brief: str

    # Run metadata
    run_id: str
    errors: list[str]

    # Memory context injected by Orchestrator (episodic + semantic summaries)
    memory_context: str

    # Market Momentum summary produced by context_manager node
    # Injected into collector extraction and analysis prompts
    momentum_summary: str

    # Phase 3: Parallel fan-out state
    # Per-competitor raw signals accumulated from parallel sub-agents
    # Uses custom reducer to merge concurrent sub-agent outputs
    competitor_signals: Annotated[dict, _merge_competitor_signals]

    # Number of retry cycles completed (0 = initial collection only)
    retry_count: int

    # Total tool calls this run, accumulated across all parallel sub-agents
    tool_call_count: Annotated[int, operator.add]

    # Active competitor for the current sub-agent invocation (set by Send())
    current_competitor: str

    # Phase 5: self-correction
    # Number of reflection passes completed (0 = no reflection yet; max 1)
    reflection_count: int

    # Per-node wall-clock latencies in milliseconds (node_name → ms)
    # Populated by traced_node decorator; merged across parallel sub-agents
    node_latencies: Annotated[dict, _merge_node_latencies]

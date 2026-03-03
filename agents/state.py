# agents/state.py
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class CompeteIQState(TypedDict):
    # Conversation messages (auto-merged by LangGraph)
    messages: Annotated[list[BaseMessage], add_messages]

    # Which competitors to monitor this run
    competitors: list[str]

    # Raw signals collected by Signal Collector
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
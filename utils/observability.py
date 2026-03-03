# utils/observability.py
"""
LangSmith tracing + structured per-node metrics for CompeteIQ.

Usage:
    from langsmith import traceable
    from utils.observability import traced_node

    @traceable(name="signal_collector", run_type="chain", tags=["competeiq"])
    @traced_node("signal_collector")
    def signal_collector(state): ...

    Apply @traceable ABOVE @traced_node. traced_node handles timing and error
    logging; @traceable creates the LangSmith span around the full wrapper.
"""
from __future__ import annotations

import functools
import logging
import time
from statistics import mean
from typing import Any, Callable

logger = logging.getLogger("competeiq.observability")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
)


# ---------------------------------------------------------------------------
# Node timing / error-logging decorator
# (LangSmith @traceable is applied explicitly in graph/workflow.py above this)
# ---------------------------------------------------------------------------

def traced_node(name: str):
    """
    Decorator that adds wall-clock latency logging and clean error surfacing
    to a LangGraph node function.

    Pair with @traceable (from langsmith) applied ABOVE this decorator so the
    LangSmith span wraps the full timed execution:

        @traceable(name="my_node", run_type="chain", tags=["competeiq"])
        @traced_node("my_node")
        def my_node(state): ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            t0 = time.perf_counter()
            try:
                result = fn(state)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.info("✅ [%s] completed in %.0fms", name, elapsed_ms)
                return result
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.error("❌ [%s] failed after %.0fms — %s", name, elapsed_ms, exc)
                raise

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Custom metric helpers
# ---------------------------------------------------------------------------

def compute_run_metrics(state: dict) -> dict[str, Any]:
    """
    Derives a structured metrics dict from final CompeteIQ state.
    Logged to run_log.jsonl and surfaced in LangSmith run outputs.
    """
    raw = state.get("raw_signals", [])
    analyzed = state.get("analyzed_signals", [])

    threats = [s for s in analyzed if s.get("assessment") == "THREAT"]
    opps    = [s for s in analyzed if s.get("assessment") == "OPPORTUNITY"]

    confidences = [s.get("confidence", 0) for s in raw if s.get("confidence")]
    impacts     = [s.get("impact_score", 0) for s in analyzed if s.get("impact_score")]

    # competitors covered = unique competitor names in raw signals
    competitors_covered = list({s.get("competitor") for s in raw if s.get("competitor")})

    # Count total tool calls made across all messages
    messages = state.get("messages", [])
    tool_calls_total = sum(
        len(m.tool_calls)
        for m in messages
        if hasattr(m, "tool_calls") and isinstance(m.tool_calls, list) and m.tool_calls
    )

    return {
        "run_id":               state.get("run_id", "unknown"),
        "signals_collected":    len(raw),
        "signals_analyzed":     len(analyzed),
        "threats":              len(threats),
        "opportunities":        len(opps),
        "neutral":              len(analyzed) - len(threats) - len(opps),
        "avg_confidence":       round(mean(confidences), 2) if confidences else 0.0,
        "avg_impact_score":     round(mean(impacts), 2)     if impacts     else 0.0,
        "competitors_covered":  competitors_covered,
        "retry_count":          state.get("retry_count", 0),
        "failed_tools":         state.get("failed_tools", []),
        "tool_calls":           state.get("tool_call_count", tool_calls_total),
    }

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
    Works with both sync and async node functions.
"""
from __future__ import annotations

import asyncio
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


def _inject_latency(result: dict, name: str, elapsed_ms: float) -> dict:
    """Merge per-node timing into the result dict returned to LangGraph."""
    if isinstance(result, dict):
        latencies = dict(result.get("node_latencies") or {})
        latencies[name] = round(elapsed_ms, 1)
        result["node_latencies"] = latencies
    return result


# Node timing / error-logging decorator — supports both sync and async nodes
def traced_node(name: str):
    """
    Decorator that adds wall-clock latency logging, clean error surfacing,
    and per-node timing (written to state["node_latencies"]) to a LangGraph
    node function.

    Pair with @traceable (from langsmith) applied ABOVE this decorator so the
    LangSmith span wraps the full timed execution:

        @traceable(name="my_node", run_type="chain", tags=["competeiq"])
        @traced_node("my_node")
        def my_node(state): ...

        # async nodes work identically:
        @traceable(name="my_async_node", run_type="chain", tags=["competeiq"])
        @traced_node("my_async_node")
        async def my_async_node(state): ...
    """
    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(state: dict) -> dict:
                t0 = time.perf_counter()
                try:
                    result = await fn(state)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    logger.info("✅ [%s] completed in %.0fms", name, elapsed_ms)
                    return _inject_latency(result, name, elapsed_ms)
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    logger.error("❌ [%s] failed after %.0fms — %s", name, elapsed_ms, exc)
                    raise
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(state: dict) -> dict:
                t0 = time.perf_counter()
                try:
                    result = fn(state)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    logger.info("✅ [%s] completed in %.0fms", name, elapsed_ms)
                    return _inject_latency(result, name, elapsed_ms)
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    logger.error("❌ [%s] failed after %.0fms — %s", name, elapsed_ms, exc)
                    raise
            return sync_wrapper
    return decorator


# Custom metric helpers
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

    # Token usage — Groq returns usage in response_metadata["token_usage"]
    total_input_tokens  = 0
    total_output_tokens = 0
    for m in messages:
        meta = getattr(m, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        total_input_tokens  += usage.get("prompt_tokens", 0)
        total_output_tokens += usage.get("completion_tokens", 0)

    # Groq llama-3.3-70b pricing: $0.59 / 1M input, $0.79 / 1M output
    _INPUT_CPM  = 0.59 / 1_000_000
    _OUTPUT_CPM = 0.79 / 1_000_000
    estimated_cost_usd = round(
        total_input_tokens * _INPUT_CPM + total_output_tokens * _OUTPUT_CPM, 6
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
        "total_input_tokens":   total_input_tokens,
        "total_output_tokens":  total_output_tokens,
        "estimated_cost_usd":   estimated_cost_usd,
    }

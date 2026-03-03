# memory/episodic.py
"""
Redis-backed episodic memory for CompeteIQ.

Stores a rolling window of run summaries so the Orchestrator can
prime the LLM with recent history — what competitors were active,
what signal categories dominated, whether retries happened.

Redis key: competeiq:run_history  (list, newest first via LPUSH)
Gracefully degrades to no-ops if Redis is unavailable.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("competeiq.memory.episodic")

_REDIS_KEY = "competeiq:run_history"
_MAX_STORED = 50  # cap total entries kept in Redis


def _get_client():
    """Lazily create a Redis client. Returns None if Redis is unreachable."""
    try:
        import redis
        from config import REDIS_URL
        client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        client.ping()
        return client
    except Exception as exc:
        logger.warning("Redis unavailable — episodic memory disabled: %s", exc)
        return None


def save_run(run_summary: dict[str, Any]) -> bool:
    """
    Persist a run summary dict to Redis.
    LPUSH keeps newest entries at index 0; LTRIM caps total stored.
    Returns True on success, False if Redis is down or the write fails.
    """
    client = _get_client()
    if client is None:
        return False
    try:
        client.lpush(_REDIS_KEY, json.dumps(run_summary, ensure_ascii=False, default=str))
        client.ltrim(_REDIS_KEY, 0, _MAX_STORED - 1)
        return True
    except Exception as exc:
        logger.warning("Failed to save run to episodic memory: %s", exc)
        return False


def get_recent_runs(n: int = 5) -> list[dict[str, Any]]:
    """
    Return the n most-recent run summaries (newest first).
    Returns [] if Redis is down or history is empty.
    """
    client = _get_client()
    if client is None:
        return []
    try:
        raw = client.lrange(_REDIS_KEY, 0, n - 1)
        return [json.loads(r) for r in raw]
    except Exception as exc:
        logger.warning("Failed to read from episodic memory: %s", exc)
        return []


def format_episodic_context(runs: list[dict]) -> str:
    """
    Format run summaries into a compact string for LLM injection.
    Kept brief to stay within the 4000-token input budget.
    """
    if not runs:
        return "No prior run history available."

    lines = ["=== Recent CompeteIQ run history (newest first) ==="]
    for i, r in enumerate(runs, 1):
        lines.append(
            f"  Run {i}: {r.get('timestamp', 'unknown')} | "
            f"id={r.get('run_id', '?')} | "
            f"signals={r.get('signals_collected', 0)} | "
            f"threats={r.get('threats', 0)} opps={r.get('opportunities', 0)} | "
            f"tool_calls={r.get('tool_calls', 0)} | "
            f"competitors={r.get('competitors_covered', [])}"
        )
    return "\n".join(lines)

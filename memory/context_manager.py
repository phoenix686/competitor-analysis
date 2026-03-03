# memory/context_manager.py
"""
Context engineering utilities for CompeteIQ.

get_last_n_runs      — pull run history from Redis episodic memory
compress_to_momentum_summary — LLM-distilled "Market Momentum" summary
apply_weighted_decay — score historical signals by recency × impact
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("competeiq.context_manager")


# ---------------------------------------------------------------------------
# Pull run history
# ---------------------------------------------------------------------------

def get_last_n_runs(n: int = 10) -> list[dict]:
    """
    Return the last n run summaries from Redis episodic memory.
    Returns an empty list if Redis is unavailable (degrades gracefully).
    """
    try:
        from memory.episodic import get_recent_runs
        return get_recent_runs(n)
    except Exception as exc:
        logger.debug("Redis unavailable for context pull: %s", exc)
        return []


# ---------------------------------------------------------------------------
# LLM compression → Market Momentum summary
# ---------------------------------------------------------------------------

_MOMENTUM_SYSTEM = """You are a market intelligence analyst for SwiftMart.

Given a JSON list of recent CompeteIQ monitoring run summaries, distil a concise
"Market Momentum" paragraph (≤120 words) covering:
1. Which competitors are most active right now
2. Which signal types are trending (pricing / hiring / sentiment / news)
3. Which markets (Delhi, Bangalore, Mumbai, Hyderabad, Pune) are most contested

Be specific and data-driven. Reference competitor names and markets.
If no data is available respond with exactly: "No historical momentum data available."
"""


def compress_to_momentum_summary(runs: list[dict]) -> str:
    """
    Distil run history into a compact Market Momentum paragraph via LLM.
    Falls back to a static string if no runs or LLM call fails.
    """
    if not runs:
        return "No historical momentum data available."

    from agents.llm import llm

    runs_text = json.dumps(runs, indent=2)[:2000]
    try:
        response = llm.invoke([
            SystemMessage(content=_MOMENTUM_SYSTEM),
            HumanMessage(content=f"Recent runs:\n{runs_text}"),
        ])
        return response.content.strip()
    except Exception as exc:
        logger.warning("compress_to_momentum_summary failed: %s", exc)
        return "No historical momentum data available."


# ---------------------------------------------------------------------------
# Weighted decay scoring for historical signals
# ---------------------------------------------------------------------------

def apply_weighted_decay(
    signals: list[dict],
    decay_factor: float = 0.85,
) -> list[dict]:
    """
    Score each historical signal by:
        decayed_score = impact_score × (decay_factor ** days_ago)

    Recent high-impact signals score highest; old low-impact ones score near zero.
    Returns signals sorted descending by decayed_score.

    Expects each signal to have an optional "run_date" key (ISO date string).
    Signals without a date are treated as from today (days_ago = 0).
    """
    now = datetime.datetime.now()
    scored: list[dict] = []

    for signal in signals:
        run_date_str = signal.get("run_date", "")
        try:
            signal_date = datetime.datetime.fromisoformat(run_date_str)
            days_ago = max(0, (now - signal_date).days)
        except (ValueError, TypeError):
            days_ago = 0

        impact = float(signal.get("impact_score", 5))
        decayed = round(impact * (decay_factor ** days_ago), 3)
        scored.append({**signal, "decayed_score": decayed})

    return sorted(scored, key=lambda s: s["decayed_score"], reverse=True)

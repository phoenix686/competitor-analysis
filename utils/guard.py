# utils/guard.py
"""
Hallucination guard for CompeteIQ analyzed signals.

Validates that each AnalyzedSignal produced by the LLM:
  1. References a competitor that was actually in the monitoring list
  2. Has a non-vague recommended_action (at least 30 chars, not just "monitor")
  3. Has a valid assessment value
  4. Has an impact_score in [1, 10]

Signals that fail validation are removed and their issues are returned as
strings so the caller can append them to state["errors"].

Usage:
    from utils.guard import filter_hallucinated_signals

    clean, issues = filter_hallucinated_signals(analyzed_signals, state["competitors"])
    # issues is a list of human-readable problem descriptions
"""
from __future__ import annotations

import logging

logger = logging.getLogger("competeiq.guard")

_VALID_ASSESSMENTS = {"THREAT", "OPPORTUNITY", "NEUTRAL"}
_VAGUE_ACTIONS = {"monitor", "watch", "track", "observe", "wait and see", "keep an eye"}


def filter_hallucinated_signals(
    signals: list[dict],
    valid_competitors: list[str],
) -> tuple[list[dict], list[str]]:
    """
    Remove signals that appear hallucinated or low-quality.

    Args:
        signals:           List of analyzed signal dicts from the LLM.
        valid_competitors: Competitor names from state["competitors"].

    Returns:
        (clean_signals, issues) where issues is a list of problem descriptions.
        Signals with critical problems are dropped; minor problems are logged only.
    """
    valid_comps_lower = {c.lower() for c in valid_competitors}
    clean: list[dict] = []
    issues: list[str] = []

    for sig in signals:
        problems: list[str] = []

        # 1. Competitor must be one we're actually monitoring
        comp = sig.get("competitor", "").lower().replace(" ", "_")
        if comp not in valid_comps_lower:
            problems.append(
                f"competitor '{sig.get('competitor')}' not in monitored list {valid_competitors}"
            )

        # 2. Assessment must be valid (Pydantic enforces this, but guard it anyway)
        if sig.get("assessment") not in _VALID_ASSESSMENTS:
            problems.append(f"invalid assessment: {sig.get('assessment')!r}")

        # 3. Impact score must be 1–10
        impact = sig.get("impact_score")
        if not isinstance(impact, (int, float)) or not (1 <= impact <= 10):
            problems.append(f"impact_score out of range: {impact!r}")

        # 4. Recommended action must be substantive
        action = sig.get("recommended_action", "").strip()
        action_lower = action.lower()
        if len(action) < 30 or any(action_lower == v for v in _VAGUE_ACTIONS):
            problems.append(
                f"vague recommended_action ({len(action)} chars): {action!r}"
            )

        if problems:
            issue_str = f"[{sig.get('competitor', '?')}] hallucination guard: {'; '.join(problems)}"
            issues.append(issue_str)
            logger.warning(issue_str)
            # Drop signals with unknown competitor or invalid assessment (unfixable)
            # Keep signals with only minor issues (vague action, borderline impact)
            critical = any(
                "not in monitored list" in p or "invalid assessment" in p
                for p in problems
            )
            if not critical:
                clean.append(sig)
        else:
            clean.append(sig)

    return clean, issues

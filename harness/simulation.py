# harness/simulation.py
"""
SimulationHarness — tools for stress-testing the CompeteIQ pipeline.

inject_adversarial_signals: injects contradictory signal pairs so the
  reflection_agent must reconcile conflicting evidence.

generate_golden_cases: inverts analyzed_signals from a successful run into
  new golden dataset entries in evals/golden_dataset.json format.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("competeiq.simulation")

# ---------------------------------------------------------------------------
# Adversarial signal pairs — contradictory claims from different sources
# ---------------------------------------------------------------------------
_ADVERSARIAL_PAIRS: list[tuple[dict, dict]] = [
    (
        # Claim A: Blinkit cuts fees (THREAT)
        {
            "competitor": "blinkit",
            "signal_type": "pricing",
            "description": "Blinkit has cut delivery fees to Rs0 for orders above Rs99 in Delhi",
            "source": "techcrunch.com",
            "raw_evidence": "Blinkit announces Zero Fee Delhi campaign starting March 2026",
            "confidence": 8.0,
        },
        # Claim B: Blinkit raises minimum order (contradicts fee cut narrative)
        {
            "competitor": "blinkit",
            "signal_type": "pricing",
            "description": "Blinkit raises minimum order value to Rs399 in Delhi, effectively increasing cost for low-basket users",
            "source": "economictimes.com",
            "raw_evidence": "Blinkit quietly raises basket minimum from Rs99 to Rs399 in NCR, reducing eligibility for free delivery",
            "confidence": 7.5,
        },
    ),
    (
        # Claim A: Zepto exits Bangalore
        {
            "competitor": "zepto",
            "signal_type": "news",
            "description": "Zepto confirms withdrawal from Bangalore citing unsustainable unit economics",
            "source": "businessinsider.in",
            "raw_evidence": "Zepto CEO confirms Bangalore exit: 'unit economics not viable at current scale in South India'",
            "confidence": 7.5,
        },
        # Claim B: Zepto expanding in Bangalore (contradicts exit)
        {
            "competitor": "zepto",
            "signal_type": "hiring",
            "description": "Zepto hiring 50 delivery partners and warehouse staff specifically for Bangalore expansion",
            "source": "linkedin.com/jobs",
            "raw_evidence": "Zepto job postings: 'Warehouse Associate Bangalore', 'Last-Mile Delivery Partner HSR Layout', posted this week",
            "confidence": 7.0,
        },
    ),
]


class SimulationHarness:
    """
    Utilities for testing pipeline robustness with adversarial inputs
    and for generating new golden dataset cases from production runs.
    """

    def inject_adversarial_signals(self, state: dict) -> dict:
        """
        Return a copy of state with contradictory signal pairs appended to
        raw_signals. The pairs represent plausible-but-conflicting claims
        from different sources on the same day about the same competitor.

        The analysis_agent or reflection_agent must either:
        - Produce a signal with lower confidence acknowledging the conflict, OR
        - Flag the contradiction in state["errors"]
        """
        injected: list[dict] = []
        for claim_a, claim_b in _ADVERSARIAL_PAIRS:
            injected.extend([claim_a, claim_b])

        new_state = dict(state)
        new_state["raw_signals"] = list(state.get("raw_signals", [])) + injected

        logger.info(
            "Injected %d adversarial signals (%d contradictory pairs)",
            len(injected),
            len(_ADVERSARIAL_PAIRS),
        )
        return new_state

    def generate_golden_cases(self, state: dict) -> list[dict]:
        """
        Take analyzed_signals from a successful run and use the LLM to
        generate new golden dataset entries in evals/golden_dataset.json format.

        Each golden case tests that the analysis_agent produces the correct
        assessment for a raw signal derived from the analyzed signal.

        Returns a list of dicts ready to be appended to golden_dataset.json
        test_cases.
        """
        from agents.llm import llm

        analyzed = state.get("analyzed_signals", [])
        if not analyzed:
            logger.warning("generate_golden_cases: no analyzed_signals in state")
            return []

        signals_json = json.dumps(analyzed, indent=2)[:3000]

        response = llm.invoke([
            SystemMessage(content=_GOLDEN_GEN_SYSTEM),
            HumanMessage(content=(
                f"Generate golden test cases from these analyzed signals:\n{signals_json}"
            )),
        ])

        try:
            cases = json.loads(response.content)
            if not isinstance(cases, list):
                cases = []
        except (json.JSONDecodeError, AttributeError):
            logger.warning("generate_golden_cases: LLM returned non-JSON output")
            cases = []

        logger.info("Generated %d golden test cases", len(cases))
        return cases


# ---------------------------------------------------------------------------
# Prompt for golden case generation
# ---------------------------------------------------------------------------
_GOLDEN_GEN_SYSTEM = """You are a test engineer for CompeteIQ.

Given a list of analyzed signals, create golden dataset test cases.
Each test case must follow this exact JSON structure:

{
  "id": "tc_XXX",
  "description": "one sentence describing what is being tested",
  "input": {
    "raw_signals": [
      {
        "competitor": "...",
        "signal_type": "pricing|feature|sentiment|hiring|news",
        "description": "one sentence raw signal",
        "source": "domain.com",
        "raw_evidence": "direct quote or data point",
        "confidence": 7.5
      }
    ]
  },
  "expected": {
    "assessment": "THREAT|OPPORTUNITY|NEUTRAL",
    "min_impact_score": 6,
    "reasoning_must_mention": ["keyword1", "keyword2"]
  }
}

Return ONLY a valid JSON array of test case objects. No prose, no markdown fences.
Generate one test case per analyzed signal provided."""

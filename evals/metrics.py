# evals/metrics.py
"""
DeepEval metrics for CompeteIQ analysis quality.

Two GEval metrics are defined:
  - AssessmentCorrectness : does the agent pick the right THREAT/OPPORTUNITY/NEUTRAL?
  - ReasoningQuality      : is the reasoning grounded and actionable?

Both use GroqJudge (llama-3.3-70b-versatile) so no OpenAI key is needed.
"""
from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from evals.groq_judge import GroqJudge

_judge = GroqJudge()

# ---------------------------------------------------------------------------
# Metric 1 — Assessment Correctness
# Checks that the LLM labelled the signal as THREAT / OPPORTUNITY / NEUTRAL
# in a way that matches the expected label given in the golden dataset.
# ---------------------------------------------------------------------------
assessment_correctness = GEval(
    name="AssessmentCorrectness",
    criteria=(
        "You are reviewing an AI competitive-intelligence analyst.\n"
        "The EXPECTED OUTPUT contains the correct assessment label "
        "(THREAT, OPPORTUNITY, or NEUTRAL) for a competitive signal.\n"
        "The ACTUAL OUTPUT is what the analyst produced.\n\n"
        "Score 1.0 if the actual assessment matches the expected label.\n"
        "Score 0.5 if the label is defensible but not ideal.\n"
        "Score 0.0 if the label is clearly wrong."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=_judge,
    threshold=0.5,
)

# ---------------------------------------------------------------------------
# Metric 2 — Reasoning Quality
# Checks that the reasoning is grounded in SwiftMart's specific context and
# the recommended action is concrete and implementable.
# ---------------------------------------------------------------------------
reasoning_quality = GEval(
    name="ReasoningQuality",
    criteria=(
        "You are reviewing an AI competitive-intelligence analyst.\n"
        "The INPUT is a competitive signal about a quick-commerce rival of SwiftMart.\n"
        "The ACTUAL OUTPUT is the analyst's full assessment (reasoning + recommended action).\n\n"
        "Score 1.0 if: the reasoning references SwiftMart's specific markets or context, "
        "AND the recommended action is concrete (not vague like 'monitor closely').\n"
        "Score 0.5 if: reasoning is generic but plausible, or action lacks specificity.\n"
        "Score 0.0 if: reasoning is hallucinated, irrelevant, or action is absent."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    model=_judge,
    threshold=0.5,
)

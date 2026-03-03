# tests/test_evals.py
"""
Phase 4 evaluation harness for CompeteIQ.

Runs the Analysis Agent against a golden dataset of known competitive signals
and asserts both deterministic correctness and LLM-judged quality.

Strategy
--------
For each test case in evals/golden_dataset.json:

1. DETERMINISTIC checks (fast, no LLM):
   - Expected assessment matches (THREAT / OPPORTUNITY / NEUTRAL)
   - Impact score is in the expected range
   - Reasoning mentions required keywords

2. DEEPEVAL GEval checks (LLM-as-judge via Groq):
   - AssessmentCorrectness  — label matches expected
   - ReasoningQuality       — grounded, actionable reasoning

REGRESSION GATE
---------------
test_regression_gate() aggregates all per-case scores and fails if the
average falls below PASS_THRESHOLD (default 0.6).  This prevents silent
quality regressions when prompts or models are updated.

Run:
    uv run pytest tests/test_evals.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from deepeval.test_case import LLMTestCase

from agents.schema import AnalyzedSignalList
from agents.llm import llm
from langchain_core.messages import HumanMessage, SystemMessage
from prompts.constants import ANALYSIS_SYSTEM
from evals.metrics import assessment_correctness, reasoning_quality

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GOLDEN_DATASET_PATH = Path(__file__).parent.parent / "evals" / "golden_dataset.json"
PASS_THRESHOLD = 0.6   # avg GEval score required to pass regression gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)["test_cases"]


def _run_analysis(raw_signals: list[dict]) -> list[dict]:
    """Invoke the analysis agent in isolation on a list of raw signals."""
    llm_analyzer = llm.with_structured_output(AnalyzedSignalList)
    raw_json = json.dumps(raw_signals, indent=2)
    response = llm_analyzer.invoke([
        SystemMessage(content=ANALYSIS_SYSTEM),
        HumanMessage(content=f"Analyze these signals for SwiftMart:\n{raw_json}"),
    ])
    return [s.model_dump() for s in response.signals]


def _score_case(tc: dict, analyzed: list[dict]) -> dict:
    """
    Return a score dict for one test case.

    Deterministic sub-scores:
      - assessment_match  : 1.0 if expected assessment matches, else 0.0
      - impact_range      : 1.0 if impact score is in expected range, else 0.0
      - keywords_present  : fraction of required keywords found in reasoning

    Returns a dict with sub-scores and the first analyzed signal for GEval.
    """
    if not analyzed:
        return {"assessment_match": 0.0, "impact_range": 0.0, "keywords": 0.0, "signal": None}

    sig = analyzed[0]  # golden dataset has one signal per case
    expected = tc["expected"]

    # Assessment match
    if "assessment" in expected:
        assessment_match = 1.0 if sig.get("assessment") == expected["assessment"] else 0.0
    elif "assessment_one_of" in expected:
        assessment_match = 1.0 if sig.get("assessment") in expected["assessment_one_of"] else 0.0
    else:
        assessment_match = 1.0  # no expectation = pass

    # Impact score range
    impact = sig.get("impact_score", 0)
    min_ok = impact >= expected.get("min_impact_score", 1)
    max_ok = impact <= expected.get("max_impact_score", 10)
    impact_range = 1.0 if (min_ok and max_ok) else 0.0

    # Required keywords in reasoning
    reasoning = (sig.get("reasoning", "") + " " + sig.get("recommended_action", "")).lower()
    keywords = expected.get("reasoning_must_mention", [])
    kw_score = sum(1 for k in keywords if k.lower() in reasoning) / len(keywords) if keywords else 1.0

    return {
        "assessment_match": assessment_match,
        "impact_range":     impact_range,
        "keywords":         kw_score,
        "signal":           sig,
    }


# ---------------------------------------------------------------------------
# Module-scoped fixture: run analysis once per case, share across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eval_results() -> list[dict]:
    """
    Run the analysis agent on every golden test case.
    Results are cached for the whole test session (module scope).
    Returns a list of {tc, analyzed, scores} dicts.
    """
    cases = _load_golden_dataset()
    results = []
    for tc in cases:
        analyzed = _run_analysis(tc["input"]["raw_signals"])
        scores = _score_case(tc, analyzed)
        results.append({"tc": tc, "analyzed": analyzed, "scores": scores})
    return results


# ---------------------------------------------------------------------------
# Test 1 — Deterministic: assessment labels
# ---------------------------------------------------------------------------

def test_assessment_labels_correct(eval_results):
    """Every golden test case must produce the expected assessment label."""
    failures = []
    for r in eval_results:
        score = r["scores"]["assessment_match"]
        if score < 1.0:
            tc_id = r["tc"]["id"]
            got = r["analyzed"][0].get("assessment") if r["analyzed"] else "none"
            expected = r["tc"]["expected"].get("assessment") or r["tc"]["expected"].get("assessment_one_of")
            failures.append(f"  {tc_id}: expected {expected}, got {got!r}")

    assert not failures, "Assessment label mismatches:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Test 2 — Deterministic: impact score ranges
# ---------------------------------------------------------------------------

def test_impact_scores_in_expected_range(eval_results):
    """Impact scores must fall within each test case's expected min/max."""
    failures = []
    for r in eval_results:
        if r["scores"]["impact_range"] < 1.0:
            tc = r["tc"]
            sig = r["analyzed"][0] if r["analyzed"] else {}
            failures.append(
                f"  {tc['id']}: impact={sig.get('impact_score')}, "
                f"expected [{tc['expected'].get('min_impact_score', 1)}"
                f"–{tc['expected'].get('max_impact_score', 10)}]"
            )
    assert not failures, "Impact score out of range:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Test 3 — Deterministic: reasoning keywords
# ---------------------------------------------------------------------------

def test_reasoning_mentions_required_keywords(eval_results):
    """Reasoning must mention context-specific keywords from the golden dataset."""
    failures = []
    for r in eval_results:
        score = r["scores"]["keywords"]
        if score < 1.0:
            tc = r["tc"]
            sig = r["analyzed"][0] if r["analyzed"] else {}
            reasoning = sig.get("reasoning", "")
            missing = [
                k for k in tc["expected"].get("reasoning_must_mention", [])
                if k.lower() not in reasoning.lower()
            ]
            failures.append(f"  {tc['id']}: reasoning missing keywords {missing}")
    assert not failures, "Reasoning keyword check failed:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Test 4 — GEval: AssessmentCorrectness (LLM judge)
# ---------------------------------------------------------------------------

def test_geval_assessment_correctness(eval_results):
    """GEval: LLM judge must agree that the assessment label is correct."""
    from deepeval import assert_test

    for r in eval_results:
        sig = r["scores"]["signal"]
        if sig is None:
            pytest.fail(f"{r['tc']['id']}: no analyzed signal produced")

        expected = r["tc"]["expected"]
        expected_label = expected.get("assessment") or "/".join(expected.get("assessment_one_of", []))

        test_case = LLMTestCase(
            input=r["tc"]["description"],
            actual_output=f"Assessment: {sig.get('assessment')}. Reasoning: {sig.get('reasoning', '')}",
            expected_output=f"Expected assessment: {expected_label}",
        )
        assert_test(test_case, [assessment_correctness])


# ---------------------------------------------------------------------------
# Test 5 — GEval: ReasoningQuality (LLM judge)
# ---------------------------------------------------------------------------

def test_geval_reasoning_quality(eval_results):
    """GEval: reasoning must be grounded and the recommended action must be concrete."""
    from deepeval import assert_test

    for r in eval_results:
        sig = r["scores"]["signal"]
        if sig is None:
            pytest.fail(f"{r['tc']['id']}: no analyzed signal produced")

        test_case = LLMTestCase(
            input=r["tc"]["description"],
            actual_output=(
                f"Reasoning: {sig.get('reasoning', '')} "
                f"Recommended action: {sig.get('recommended_action', '')}"
            ),
        )
        assert_test(test_case, [reasoning_quality])


# ---------------------------------------------------------------------------
# Test 6 — REGRESSION GATE
# ---------------------------------------------------------------------------

def test_regression_gate(eval_results):
    """
    Aggregate deterministic scores across all test cases.
    Fails if the average drops below PASS_THRESHOLD (default 0.6).

    This is the CI gate: update the golden dataset to lock in quality gains.
    """
    all_scores = []
    for r in eval_results:
        s = r["scores"]
        case_score = (
            s["assessment_match"] * 0.5
            + s["impact_range"]   * 0.3
            + s["keywords"]       * 0.2
        )
        all_scores.append(case_score)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    per_case = "\n".join(
        f"  {r['tc']['id']}: {score:.2f}"
        for r, score in zip(eval_results, all_scores)
    )

    assert avg >= PASS_THRESHOLD, (
        f"Regression gate FAILED — avg score {avg:.2f} < threshold {PASS_THRESHOLD}\n"
        f"Per-case scores:\n{per_case}"
    )
    print(f"\nRegression gate PASSED — avg score {avg:.2f} (threshold {PASS_THRESHOLD})")

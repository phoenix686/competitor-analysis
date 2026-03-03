# main.py
# ── stdlib only at the top ───────────────────────────────────────────────────
import os
import time
import argparse


def _apply_provider_arg() -> None:
    """
    Parse --provider BEFORE any project imports so that os.environ["LLM_PROVIDER"]
    is set before config.py → agents/llm.py read it at import time.

    Uses parse_known_args so pytest flags don't cause errors when this file
    is imported indirectly during testing.
    """
    parser = argparse.ArgumentParser(
        description="CompeteIQ competitive intelligence run",
        add_help=False,  # let the real imports handle --help if needed
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "groq"],
        default=None,
        help="LLM provider: 'ollama' (local, default) or 'groq' (cloud)",
    )
    args, _ = parser.parse_known_args()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider


_apply_provider_arg()  # ← must run before any project import below

# ── project imports (after env var is set) ───────────────────────────────────
from graph.workflow import competeiq_graph          # noqa: E402
from config import COMPETITORS                      # noqa: E402
from data.run_log import log_run                    # noqa: E402
from memory.episodic import save_run                # noqa: E402
from utils.observability import compute_run_metrics # noqa: E402


def run() -> None:
    provider = os.getenv("LLM_PROVIDER", "ollama")
    print(f"Starting CompeteIQ monitoring run... [LLM: {provider}]\n")

    initial_state = {
        "messages": [],
        "competitors": COMPETITORS,
        "raw_signals": [],
        "analyzed_signals": [],
        "final_brief": "",
        "run_id": "",
        "errors": [],
        "memory_context": "",
    }

    t0 = time.perf_counter()
    result = competeiq_graph.invoke(initial_state)
    latency_ms = (time.perf_counter() - t0) * 1000

    print("=" * 60)
    print(result["final_brief"])
    print("=" * 60)
    print(f"\nSignals collected : {len(result['raw_signals'])}")
    print(f"Signals analyzed  : {len(result['analyzed_signals'])}")
    print(f"Total latency     : {latency_ms:.0f}ms")
    print(f"LLM provider      : {provider}")

    # Off-critical-path logging
    log_run(result, latency_ms=latency_ms)
    print("Run logged to data/run_log.jsonl")

    metrics = compute_run_metrics(result)
    metrics["latency_ms"] = round(latency_ms, 1)
    saved = save_run(metrics)
    print("Run saved to episodic memory (Redis)" if saved else "Episodic memory skipped (Redis unavailable)")


if __name__ == "__main__":
    run()

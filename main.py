# main.py
import time
from graph.workflow import competeiq_graph
from config import COMPETITORS
from data.run_log import log_run
from memory.episodic import save_run
from utils.observability import compute_run_metrics


def run() -> None:
    print("Starting CompeteIQ monitoring run... [LLM: Groq llama-3.3-70b-versatile]\n")

    initial_state = {
        "messages": [],
        "competitors": COMPETITORS,
        "raw_signals": [],
        "analyzed_signals": [],
        "final_brief": "",
        "run_id": "",
        "errors": [],
        "memory_context": "",
        # Phase 3: parallel fan-out state
        "competitor_signals": {},
        "retry_count": 0,
        "tool_call_count": 0,
        "current_competitor": "",
        "reflection_count": 0,
    }

    t0 = time.perf_counter()
    result = competeiq_graph.invoke(initial_state)
    latency_ms = (time.perf_counter() - t0) * 1000

    print("=" * 60)
    print(result["final_brief"])
    print("=" * 60)
    print(f"\nSignals collected : {len(result['raw_signals'])}")
    print(f"Signals analyzed  : {len(result['analyzed_signals'])}")
    print(f"Tool calls made   : {result.get('tool_call_count', 0)}")
    print(f"Retry count       : {result.get('retry_count', 0)}")
    print(f"Total latency     : {latency_ms:.0f}ms")

    # Off-critical-path logging
    log_run(result, latency_ms=latency_ms)
    print("Run logged to data/run_log.jsonl")

    metrics = compute_run_metrics(result)
    metrics["latency_ms"] = round(latency_ms, 1)
    saved = save_run(metrics)
    print("Run saved to episodic memory (Redis)" if saved else "Episodic memory skipped (Redis unavailable)")


if __name__ == "__main__":
    run()

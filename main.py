# main.py
import time
from graph.workflow import competeiq_graph
from config import COMPETITORS
from data.run_log import log_run


def run():
    print("🚀 Starting CompeteIQ monitoring run...\n")

    initial_state = {
        "messages": [],
        "competitors": COMPETITORS,
        "raw_signals": [],
        "analyzed_signals": [],
        "final_brief": "",
        "run_id": "",
        "errors": [],
    }

    t0 = time.perf_counter()
    result = competeiq_graph.invoke(initial_state)
    latency_ms = (time.perf_counter() - t0) * 1000

    print("=" * 60)
    print(result["final_brief"])
    print("=" * 60)
    print(f"\n✅ Signals collected: {len(result['raw_signals'])}")
    print(f"✅ Signals analyzed: {len(result['analyzed_signals'])}")
    print(f"⏱️  Total latency: {latency_ms:.0f}ms")

    # Write run log AFTER brief is printed — never on the critical path
    log_run(result, latency_ms=latency_ms)
    print(f"📝 Run logged to data/run_log.jsonl")


if __name__ == "__main__":
    run()
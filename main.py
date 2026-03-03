# main.py
import argparse
import asyncio
import time
from graph.workflow import competeiq_graph
from config import COMPETITORS
from data.run_log import log_run
from memory.episodic import save_run
from utils.observability import compute_run_metrics


async def run() -> None:
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
        "momentum_summary": "",
        # Phase 3: parallel fan-out state
        "competitor_signals": {},
        "retry_count": 0,
        "tool_call_count": 0,
        "current_competitor": "",
        # Phase 5: self-correction
        "reflection_count": 0,
        # Phase 6: per-node latency tracking
        "node_latencies": {},
    }

    t0 = time.perf_counter()
    result = await competeiq_graph.ainvoke(initial_state)
    latency_ms = (time.perf_counter() - t0) * 1000

    print("=" * 60)
    print(result["final_brief"])
    print("=" * 60)
    print(f"\nSignals collected : {len(result['raw_signals'])}")
    print(f"Signals analyzed  : {len(result['analyzed_signals'])}")
    print(f"Tool calls made   : {result.get('tool_call_count', 0)}")
    print(f"Retry count       : {result.get('retry_count', 0)}")
    print(f"Total latency     : {latency_ms:.0f}ms")

    node_latencies = result.get("node_latencies") or {}
    if node_latencies:
        print("\nPer-node latencies:")
        for node, ms in sorted(node_latencies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {node:<30} {ms:>7.0f}ms")

    # Off-critical-path logging
    log_run(result, latency_ms=latency_ms)
    print("\nRun logged to data/run_log.jsonl")

    metrics = compute_run_metrics(result)
    metrics["latency_ms"] = round(latency_ms, 1)

    inp  = metrics.get("total_input_tokens", 0)
    out  = metrics.get("total_output_tokens", 0)
    cost = metrics.get("estimated_cost_usd", 0.0)
    if inp or out:
        print(f"\nToken usage:")
        print(f"  Input tokens    : {inp:,}")
        print(f"  Output tokens   : {out:,}")
        print(f"  Estimated cost  : ${cost:.4f} USD")

    saved = save_run(metrics)
    print("Run saved to episodic memory (Redis)" if saved else "Episodic memory skipped (Redis unavailable)")


async def _scheduled_loop() -> None:
    """Run the pipeline every 6 hours using APScheduler."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    scheduler = AsyncIOScheduler()
    scheduler.add_job(run, "interval", hours=6, id="competeiq_run")
    scheduler.start()
    print("Scheduler started — pipeline runs every 6 hours. Press Ctrl+C to stop.\n")

    # Fire immediately on startup so we don't wait 6 hours for the first run
    await run()

    try:
        await asyncio.Event().wait()   # block until KeyboardInterrupt
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("\nScheduler stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CompeteIQ competitive intelligence agent")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run the pipeline every 6 hours (fires once immediately on startup)",
    )
    args = parser.parse_args()

    if args.schedule:
        asyncio.run(_scheduled_loop())
    else:
        asyncio.run(run())

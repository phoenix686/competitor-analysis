# data/run_log.py
"""
Append-only JSONL run logger for CompeteIQ.

Each run appends one JSON line to data/run_log.jsonl.
Written AFTER the brief is printed — never on the critical path.

Usage:
    from data.run_log import log_run
    log_run(state, latency_ms=1240)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from utils.observability import compute_run_metrics

LOG_PATH = Path(__file__).parent / "run_log.jsonl"


def log_run(state: dict, latency_ms: float = 0.0) -> None:
    """Append a structured JSON record for this run to run_log.jsonl."""
    metrics = compute_run_metrics(state)
    record = {
        **metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "latency_ms": round(latency_ms, 1),
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

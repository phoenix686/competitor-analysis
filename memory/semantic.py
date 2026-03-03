# memory/semantic.py
"""
ChromaDB-backed semantic memory for CompeteIQ.

Stores analyzed signals as vector embeddings so the Analysis Agent can
retrieve historically similar signals before classifying new ones —
improving consistency and grounding new analysis in past findings.

Collection : competeiq_signals
Document   : "{competitor} — {signal_type} — {description}. Reasoning: … Action: …"
Metadata   : competitor, signal_type, assessment, impact_score, run_id
ID         : sha256(competitor|description|run_id)[:16]  — dedup-safe

Gracefully degrades to no-ops if ChromaDB is unavailable.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger("competeiq.memory.semantic")

_COLLECTION_NAME = "competeiq_signals"


def _get_collection():
    """
    Return the ChromaDB collection, creating it on first use.
    Returns None on any failure so callers can degrade gracefully.
    """
    try:
        import chromadb
        from config import CHROMA_PERSIST_DIR

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        # Uses chromadb's built-in ONNX embedding model (no extra installs needed)
        return client.get_or_create_collection(name=_COLLECTION_NAME)
    except Exception as exc:
        logger.warning("ChromaDB unavailable — semantic memory disabled: %s", exc)
        return None


def _signal_id(signal: dict, run_id: str) -> str:
    """Stable, dedup-safe ID: first 16 hex chars of sha256(key)."""
    key = f"{signal.get('competitor')}|{signal.get('description')}|{run_id}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _signal_document(signal: dict) -> str:
    """Human-readable document used for embedding and retrieval display."""
    return (
        f"{signal.get('competitor', '')} — "
        f"{signal.get('signal_type', '')} — "
        f"{signal.get('description', '')}. "
        f"Reasoning: {signal.get('reasoning', '')}. "
        f"Action: {signal.get('recommended_action', '')}"
    )


def upsert_signal(signal: dict[str, Any], run_id: str) -> None:
    """
    Upsert a single analyzed signal into ChromaDB.
    Silently no-ops if ChromaDB is unavailable.
    """
    collection = _get_collection()
    if collection is None:
        return
    try:
        collection.upsert(
            ids=[_signal_id(signal, run_id)],
            documents=[_signal_document(signal)],
            metadatas=[{
                "competitor":  str(signal.get("competitor", "")),
                "signal_type": str(signal.get("signal_type", "")),
                "assessment":  str(signal.get("assessment", "")),
                "impact_score": int(signal.get("impact_score", 0)),
                "run_id":      run_id,
            }],
        )
    except Exception as exc:
        logger.warning("Failed to upsert signal to ChromaDB: %s", exc)


def upsert_signals(signals: list[dict], run_id: str) -> None:
    """Batch upsert all analyzed signals from a completed run."""
    for signal in signals:
        upsert_signal(signal, run_id)
    if signals:
        collection = _get_collection()
        total = collection.count() if collection else "?"
        logger.info("ChromaDB: upserted %d signals (run=%s, total stored=%s)", len(signals), run_id, total)


def retrieve_similar(query: str, k: int = 3) -> list[dict[str, Any]]:
    """
    Retrieve top-k historically similar signals from ChromaDB.
    Returns list of metadata dicts enriched with the document text.
    Returns [] if ChromaDB is down or the collection is empty.
    """
    collection = _get_collection()
    if collection is None:
        return []
    try:
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            items.append({**meta, "document": doc})
        return items
    except Exception as exc:
        logger.warning("Failed to query ChromaDB: %s", exc)
        return []


def format_semantic_context(similar: list[dict]) -> str:
    """
    Format retrieved signals into a compact string for LLM injection.
    Kept brief to stay within the 4000-token input budget.
    """
    if not similar:
        return "No similar historical signals found."

    lines = ["=== Historically similar signals (for context, do not repeat verbatim) ==="]
    for s in similar:
        lines.append(
            f"  [{s.get('assessment', '?')} | {s.get('competitor', '?')} | "
            f"impact={s.get('impact_score', '?')}] "
            f"{s.get('document', '')[:200]}"
        )
    return "\n".join(lines)

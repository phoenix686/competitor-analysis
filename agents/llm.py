# agents/llm.py
"""
LLM provider factory for CompeteIQ.

Default: Ollama (local, no rate limits)   → llama3.2:3b
Switch:  Groq  (cloud, rate-limited)      → llama-3.3-70b-versatile

Override at runtime:
    python main.py --provider groq
    OLLAMA_MODEL=llama3.1:8b python main.py   # bigger local model
"""
import logging
from config import LLM_PROVIDER, GROQ_API_KEY

logger = logging.getLogger("competeiq.llm")

# OLLAMA_MODEL is also read from env so you can override without editing code:
#   OLLAMA_MODEL=llama3.1:8b python main.py
import os
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
_GROQ_MODEL   = "llama-3.3-70b-versatile"


def _build_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        logger.info("LLM provider: Groq (%s)", _GROQ_MODEL)
        return ChatGroq(
            model=_GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0,
        )

    # Default: Ollama — local, zero rate limits
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama is not installed. "
            "Run: uv add langchain-ollama  (or switch to Groq with --provider groq)"
        )
    logger.info("LLM provider: Ollama local (%s)", _OLLAMA_MODEL)
    return ChatOllama(model=_OLLAMA_MODEL, temperature=0)


llm = _build_llm()

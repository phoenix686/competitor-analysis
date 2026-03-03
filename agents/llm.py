# agents/llm.py
"""
LLM provider for CompeteIQ — Groq (cloud).
Model: llama-3.3-70b-versatile
"""
import logging
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

logger = logging.getLogger("competeiq.llm")

_GROQ_MODEL = "llama-3.3-70b-versatile"

logger.info("LLM provider: Groq (%s)", _GROQ_MODEL)

llm = ChatGroq(
    model=_GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)

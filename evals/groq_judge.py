# evals/groq_judge.py
"""
Groq-backed DeepEval model wrapper.

Lets DeepEval's GEval metric use Groq (llama-3.3-70b-versatile) as the
LLM judge instead of OpenAI, so no separate OpenAI key is needed.
"""
from __future__ import annotations

from deepeval.models.base_model import DeepEvalBaseLLM


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval-compatible LLM wrapper that delegates to Groq via LangChain."""

    def __init__(self) -> None:
        from langchain_groq import ChatGroq
        from config import GROQ_API_KEY
        self._chat = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0,
        )

    def load_model(self):
        return self._chat

    def generate(self, prompt: str, schema=None) -> str:
        from langchain_core.messages import HumanMessage
        if schema is not None:
            # DeepEval sometimes requests structured output
            structured = self._chat.with_structured_output(schema)
            return structured.invoke([HumanMessage(content=prompt)])
        return self._chat.invoke([HumanMessage(content=prompt)]).content

    async def a_generate(self, prompt: str, schema=None) -> str:
        # Synchronous fallback — DeepEval will call the sync version if needed
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return "groq/llama-3.3-70b-versatile"

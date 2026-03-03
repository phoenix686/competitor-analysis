# skills/mcp_client.py
"""
MCP client for CompeteIQ tools.

Launches skills/mcp_server.py as a subprocess (stdio transport) and returns
LangChain-compatible BaseTool objects that graph nodes can call with ainvoke().

The client and server subprocess are lazily initialized on first call and
remain alive for the lifetime of the process. Falls back to direct tool
imports if the MCP server cannot start (e.g. in test environments).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("competeiq.mcp_client")

_tools: list | None = None
_client = None


async def get_mcp_tools() -> list:
    """
    Return LangChain-compatible tools sourced from the MCP server.

    First call: starts the MCP server subprocess, connects, fetches tools.
    Subsequent calls: return the cached tool list (no subprocess overhead).
    """
    global _client, _tools
    if _tools is not None:
        return _tools

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        _client = MultiServerMCPClient({
            "competeiq": {
                "command": "python",
                "args": ["-m", "skills.mcp_server"],
                "transport": "stdio",
            }
        })
        # langchain-mcp-adapters >= 0.1.0: get_tools() is async, no context manager
        _tools = await _client.get_tools()
        logger.info("MCP client ready: %d tools loaded", len(_tools))
        return _tools

    except Exception as exc:
        logger.warning(
            "MCP server unavailable (%s) — falling back to direct tool imports", exc
        )
        return _get_fallback_tools()


def _get_fallback_tools() -> list:
    """Return direct LangChain @tool objects when MCP is unavailable."""
    from skills.tools import search_competitor, get_app_reviews, get_competitor_jobs
    return [search_competitor, get_app_reviews, get_competitor_jobs]


async def close() -> None:
    """Reset cached tools (server subprocess exits with the main process)."""
    global _client, _tools
    _client = None
    _tools = None

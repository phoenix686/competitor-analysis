# harness/replay.py
"""
Deterministic replay harness for CompeteIQ.

Replaces live external calls (Tavily, Play Store, Adzuna) with pre-recorded
fixture responses, enabling fast, offline, reproducible tests.

Strategy:
  - Tavily: patch TavilySearch._run at the CLASS level.
    Pydantic v2 blocks setattr on model *instances* (including bound methods
    like .invoke and api_wrapper.raw_results), so we must patch the unbound
    class method instead.
  - Play Store: patch google_play_scraper functions via module string.
  - Adzuna: patch httpx.get via module string.

Usage in pytest:
    from harness.replay import ReplayHarness

    def test_signal_collection():
        harness = ReplayHarness()
        with harness.patch_tools():
            result = competeiq_graph.invoke(initial_state)
        assert result["raw_signals"]
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_tavily import TavilySearch

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "tool_responses"


def _load_fixture(filename: str) -> dict:
    """Load a fixture JSON file and return parsed dict."""
    fixture_path = FIXTURES_DIR / filename
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Fixture not found: {fixture_path}. "
            "Run a live session with ReplayHarness.record_run() first."
        )
    with open(fixture_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fake implementations for each underlying external call
# ---------------------------------------------------------------------------

def _fake_tavily_run(self, query: str, **kwargs) -> dict:
    """
    Replace TavilySearch._run (class-level patch).
    Returns fixture dict keyed by competitor name in query.
    Pydantic v2 blocks instance-level setattr so we patch the class method.
    """
    q = query.lower() if isinstance(query, str) else ""
    routing = [
        ("blinkit",   "blinkit_search_2026_03_01.json"),
        ("zepto",     "blinkit_search_2026_03_01.json"),
        ("zomato",    "blinkit_search_2026_03_01.json"),
        ("amazon",    "blinkit_search_2026_03_01.json"),
        ("instamart", "blinkit_search_2026_03_01.json"),
    ]
    for keyword, fixture in routing:
        if keyword in q:
            return _load_fixture(fixture)
    return _load_fixture("blinkit_search_2026_03_01.json")


def _make_fake_app_info(app_id: str, lang: str = "en", country: str = "in") -> dict:
    return {"score": 4.2, "ratings": 1_840_000}


def _make_fake_reviews(
    app_id: str, lang: str = "en", country: str = "in", sort=None, count: int = 30
) -> tuple[list[dict], None]:
    data = _load_fixture("zepto_reviews_2026_03_01.json")
    raw_reviews = [
        {
            "score": r["score"],
            "content": r["content"],
            "at": None,
        }
        for r in data.get("recent_reviews", [])
    ]
    return raw_reviews, None


def _make_fake_httpx_response(url: str, **kwargs) -> MagicMock:
    data = _load_fixture("zomato_jobs_2026_03_01.json")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {
                "title": j["title"],
                "description": j["description"],
                "created": j["created"] + "T00:00:00Z",
                "company": {"display_name": j.get("company", "Zomato")},
            }
            for j in data.get("jobs", [])
        ]
    }
    return mock_resp


# ---------------------------------------------------------------------------
# ReplayHarness context manager
# ---------------------------------------------------------------------------

class ReplayHarness:
    """
    Patches all external I/O in CompeteIQ tools to return fixture data.

    Tavily is patched at the CLASS level (TavilySearch._run) because Pydantic v2
    blocks setattr on model instances — instance-level patches like
    patch("skills.tools._web_search.invoke", ...) raise AttributeError.
    """

    @contextmanager
    def patch_tools(self):
        """
        Patches:
          - langchain_tavily.TavilySearch._run  → fake Tavily (class-level, Pydantic v2 safe)
          - skills.tools.get_app_info           → fake Play Store app info
          - skills.tools.reviews                → fake Play Store review list
          - skills.tools.httpx.get              → fake Adzuna HTTP response
        """
        with (
            patch.object(TavilySearch, "_run",        _fake_tavily_run),
            patch("skills.tools.get_app_info",         side_effect=_make_fake_app_info),
            patch("skills.tools.reviews",              side_effect=_make_fake_reviews),
            patch("skills.tools.httpx.get",            side_effect=_make_fake_httpx_response),
        ):
            yield

    def record_run(self, run_id: str | None = None) -> None:
        """
        (Future) Intercept a live run and save tool responses as new fixtures.
        Useful for refreshing fixtures after APIs change.
        """
        raise NotImplementedError(
            "record_run() is not yet implemented. "
            "Run manually and save output to harness/fixtures/tool_responses/."
        )

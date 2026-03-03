# skills/tools.py
import logging
import os
import httpx
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from google_play_scraper import reviews, app as get_app_info, Sort
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from config import ADZUNA_APP_ID, ADZUNA_APP_KEY, ADZUNA_COUNTRY, COMPETITOR_APP_IDS

_log = logging.getLogger("competeiq.tools")

# Shared retry policy — 3 attempts, exponential backoff (1s → 2s → 4s max 8s)
_retry_kwargs = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    before_sleep=before_sleep_log(_log, logging.WARNING),
    reraise=True,
)

# Initialize Tavily search instances
_web_search = TavilySearch(max_results=5, topic="general")


# ---------------------------------------------------------------------------
# Retry-wrapped inner functions — raise on failure so tenacity can retry
# ---------------------------------------------------------------------------

@retry(**_retry_kwargs)
def _tavily_search(query: str):
    return _web_search.invoke(query)


@retry(**_retry_kwargs)
def _fetch_app_info(app_id: str) -> dict:
    return get_app_info(app_id, lang="en", country="in")


@retry(**_retry_kwargs)
def _fetch_reviews(app_id: str, count: int) -> tuple:
    return reviews(app_id, lang="en", country="in", sort=Sort.NEWEST, count=count)


@retry(**_retry_kwargs)
def _fetch_jobs(url: str, params: dict) -> dict:
    response = httpx.get(url, params=params, timeout=10.0)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Public LangChain tools — graceful error strings on final failure
# ---------------------------------------------------------------------------

@tool
def search_competitor(query: str) -> str:
    """
    Search for any information about Indian quick commerce competitors.
    Use for pricing changes, news, feature launches, partnerships, funding.
    Always include competitor name and be specific.
    Examples:
    - 'Blinkit delivery fee cut Gurgaon 2026'
    - 'Zepto new city launch India 2026'
    - 'Zomato Gold subscription price change'
    """
    try:
        results = _tavily_search(query)
        return str(results)
    except Exception as e:
        return f"search_error: {str(e)}"


@tool
def get_app_reviews(competitor: str, count: int = 30) -> str:
    """
    Get recent Play Store reviews and rating for a competitor app.
    Use to detect sentiment shifts, feature complaints, or service quality changes.
    competitor must be one of: zomato, blinkit, zepto, amazon_fresh, instamart
    """
    app_id = COMPETITOR_APP_IDS.get(competitor)
    if not app_id:
        return f"error: unknown competitor '{competitor}'. Choose from {list(COMPETITOR_APP_IDS.keys())}"

    try:
        app_data = _fetch_app_info(app_id)
        current_rating = app_data.get("score", "unknown")
        total_ratings = app_data.get("ratings", "unknown")

        review_results, _ = _fetch_reviews(app_id, count)

        compressed = [
            {
                "score": r["score"],
                "content": r["content"][:200],
                "date": str(r["at"].date()) if r.get("at") else "unknown",
            }
            for r in review_results
        ]

        return str({
            "competitor": competitor,
            "current_rating": current_rating,
            "total_ratings": total_ratings,
            "recent_reviews": compressed,
        })

    except Exception as e:
        return f"appstore_error: {str(e)}"


@tool
def get_competitor_jobs(company: str) -> str:
    """
    Get ALL recent job postings from a competitor company to infer strategic priorities.
    Do not filter by role — fetch everything and let analysis determine the pattern.
    High volume in any area reveals strategic intent:
    - Many ML/AI/data roles = building personalization or recommendations
    - Many ops/supply chain roles = city expansion underway
    - Many product roles = major feature development
    - Many finance/legal roles = preparing for IPO or acquisition
    company: competitor name e.g. 'Blinkit', 'Zepto', 'Zomato', 'Amazon'
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return "error: Adzuna API credentials not configured"

    try:
        url = f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/1"
        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_APP_KEY,
            "results_per_page": 50,
            "what": company,
            "content-type": "application/json",
        }

        data = _fetch_jobs(url, params)
        jobs = data.get("results", [])

        company_lower = company.lower()
        filtered = [
            j for j in jobs
            if any(
                word in j.get("company", {}).get("display_name", "").lower()
                for word in company_lower.split()
            )
        ]

        compressed = [
            {
                "title": j.get("title", ""),
                "description": j.get("description", "")[:150],
                "created": j.get("created", "")[:10],
            }
            for j in filtered
        ]

        return str({
            "company": company,
            "total_open_roles": len(filtered),
            "jobs": compressed,
            "note": "Analyze the distribution of job titles to infer strategic priorities",
        })

    except Exception as e:
        return f"jobs_error: {str(e)}"

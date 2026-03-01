# skills/tools.py
import os
import httpx
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from google_play_scraper import reviews, app as get_app_info, Sort
from config import ADZUNA_APP_ID, ADZUNA_APP_KEY, ADZUNA_COUNTRY, COMPETITOR_APP_IDS

# Initialize Tavily search instances
_web_search = TavilySearch(max_results=5, topic="general")


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
        results = _web_search.invoke(query)
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
        app_data = get_app_info(app_id, lang='en', country='in')
        current_rating = app_data.get('score', 'unknown')
        total_ratings = app_data.get('ratings', 'unknown')

        review_results, _ = reviews(
            app_id,
            lang='en',
            country='in',
            sort=Sort.NEWEST,
            count=count
        )

        compressed = [
            {
                "score": r["score"],
                "content": r["content"][:200],
                "date": str(r["at"].date()) if r.get("at") else "unknown"
            }
            for r in review_results
        ]

        return str({
            "competitor": competitor,
            "current_rating": current_rating,
            "total_ratings": total_ratings,
            "recent_reviews": compressed
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
            "content-type": "application/json"
        }

        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        jobs = data.get("results", [])

        # Filter to actual company jobs only
        # Replace this line:
        company_lower = company.lower()
        filtered = [
            j for j in jobs
            if company_lower in j.get("company", {}).get("display_name", "").lower()
        ]

        # With this:
        company_lower = company.lower()
        filtered = [
            j for j in jobs
            if any(
                word in j.get("company", {}).get("display_name", "").lower()
                for word in company_lower.split()
            )
        ]
        # Compress — title is the key signal, short description for context
        compressed = [
            {
                "title": j.get("title", ""),
                "description": j.get("description", "")[:150],
                "created": j.get("created", "")[:10]
            }
            for j in filtered
        ]

        # Group by rough category for easier LLM reasoning
        return str({
            "company": company,
            "total_open_roles": len(filtered),
            "jobs": compressed,
            "note": "Analyze the distribution of job titles to infer strategic priorities"
        })

    except Exception as e:
        return f"jobs_error: {str(e)}"
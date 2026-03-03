# skills/mcp_server.py
"""
FastMCP server exposing CompeteIQ's 3 tools over MCP protocol.

The tool logic, retry decorators, and error handling live entirely in
skills/tools.py — this module only adds the MCP transport layer.

Start manually (stdio, for debugging):
    python -m skills.mcp_server

Launched automatically as a subprocess by skills/mcp_client.py.
"""
import sys
import os

# Ensure project root is on sys.path when executed as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from skills.tools import _tavily_search, _fetch_app_info, _fetch_reviews, _fetch_jobs
from config import ADZUNA_APP_ID, ADZUNA_APP_KEY, ADZUNA_COUNTRY, COMPETITOR_APP_IDS
from google_play_scraper import Sort

mcp = FastMCP("competeiq-tools")


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def get_competitor_jobs(company: str) -> str:
    """
    Get recent job postings from a competitor company to infer strategic priorities.
    High volume in any area reveals strategic intent:
    - Many ML/AI/data roles = building personalisation or recommendations
    - Many ops/supply chain roles = city expansion underway
    - Many product roles = major feature development
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
            "note": "Analyse the distribution of job titles to infer strategic priorities",
        })

    except Exception as e:
        return f"jobs_error: {str(e)}"


if __name__ == "__main__":
    mcp.run()  # default: stdio transport

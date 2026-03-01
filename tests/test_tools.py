#usage:  uv run python -m tests.test_tools.py  
# test_tools.py
from skills.tools import search_competitor, get_app_reviews, get_competitor_jobs

# print("Testing competitor search...")
# print(search_competitor.invoke("Blinkit delivery fee India 2026"))

# print("\nTesting app reviews...")
# print(get_app_reviews.invoke({"competitor": "blinkit", "count": 5}))

print("\nTesting jobs...")
print(get_competitor_jobs.invoke({"company": "Zepto"}))
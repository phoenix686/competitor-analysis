# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LangSmith observability
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "competeiq")

# Memory
CHROMA_PERSIST_DIR = "./chroma_db"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Guardrails
MIN_CONFIDENCE_SCORE = 6.0
MAX_TOOL_RETRIES = 3

#"zepto", "amazon_fresh", "instamart"
# Competitors
COMPETITORS = ["zomato", "blinkit", ]

# Competitor app IDs (Play Store)
COMPETITOR_APP_IDS = {
    "zomato": "com.application.zomato",
    "blinkit": "com.grofers.customerapp",
    "zepto": "com.zepto.app",
    "amazon_fresh": "in.amazon.mShop.android.shopping",
    "instamart": "bundle.swiggy.in"
}

# Adzuna API
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
ADZUNA_COUNTRY = "in"

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LLM settings
TEMPERATURE = 0
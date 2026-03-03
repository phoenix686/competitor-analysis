# CompeteIQ

CompeteIQ is an autonomous competitive intelligence agent for **SwiftMart**, a quick-commerce company competing with Blinkit, Zepto, Zomato, Amazon Fresh, and Instamart in Indian metro markets. It runs a fully automated pipeline: parallel competitor signal collection, confidence-gated aggregation, LLM-powered analysis with hallucination guards, self-correction via a reflection loop, and an executive intelligence brief. Every run is traced with LangSmith, timed at the node level, and stored in Redis (episodic memory) and ChromaDB (semantic memory) so future runs benefit from historical context.

---

## Architecture

```mermaid
flowchart TD
    START([START]) --> ORC[Orchestrator\nfetch Redis episodic context]
    ORC -->|Send x N| C1[collector_sub_agent\nBlinkit]
    ORC -->|Send x N| C2[collector_sub_agent\nZepto]
    ORC -->|Send x N| C3[collector_sub_agent\nZomato ...]
    C1 & C2 & C3 --> AGG[Aggregator\nconfidence gate >= 6.0]
    AGG -->|zero coverage\n& retries < 2| RET[retry_collector\nreset + critique]
    RET -->|Send x N| C1
    AGG -->|all covered\nor retries exhausted| ANA[analysis_agent\nChromaDB context + guard]
    ANA -->|> 30% vague actions\nor 0 signals| REF[reflection_agent\nself-correction]
    REF --> BRF[brief_writer]
    ANA -->|quality OK| BRF
    BRF --> END_([END])

    subgraph Memory
        REDIS[(Redis\nepisodic)]
        CHROMA[(ChromaDB\nsemantic)]
    end
    ORC -.->|read| REDIS
    ANA -.->|read + write BG| CHROMA
    REF -.->|write BG| CHROMA
```

**Node summary**

| Node | Role |
|---|---|
| `orchestrator` | Seeds the run, pulls last-5-run summary from Redis |
| `collector_sub_agent` | One per competitor (parallel via `Send()`); calls 3 tools concurrently with `asyncio.gather()` |
| `aggregator` | Merges parallel results, drops signals below confidence 6.0, tracks zero-coverage |
| `retry_collector` | Increments retry counter, resets signals, injects critique for deeper search (max 2 retries) |
| `analysis_agent` | Classifies signals (THREAT/OPPORTUNITY/NEUTRAL), applies hallucination guard, writes ChromaDB in background |
| `reflection_agent` | Re-runs analysis with targeted critique when > 30% of actions are vague or 0 signals produced (max 1 pass) |
| `brief_writer` | Formats final executive brief in structured markdown |

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph 1.x (`StateGraph`, `Send()` fan-out) |
| LLM | Groq `llama-3.3-70b-versatile` via `langchain-groq` |
| Web search | Tavily (`langchain-tavily`) with tenacity retry (3x, exp backoff) |
| App reviews | `google-play-scraper` |
| Job signals | Adzuna Jobs API |
| Episodic memory | Redis (LPUSH/LRANGE run history) |
| Semantic memory | ChromaDB (`PersistentClient`, cosine similarity) |
| Tracing | LangSmith (`@traceable`) + custom `traced_node` (per-node ms) |
| Evals | DeepEval GEval with `GroqJudge` (custom `DeepEvalBaseLLM`) |
| Package manager | `uv` |

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```env
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
ADZUNA_APP_ID=your_id
ADZUNA_APP_KEY=your_key
LANGCHAIN_API_KEY=ls__...        # optional — enables LangSmith tracing
LANGCHAIN_TRACING_V2=true        # optional
REDIS_URL=redis://localhost:6379  # optional — skipped gracefully if unavailable
```

**Required:** `GROQ_API_KEY`, `TAVILY_API_KEY`
**Optional:** `ADZUNA_APP_ID` + `ADZUNA_APP_KEY` (job signals degrade gracefully without them), `LANGCHAIN_API_KEY` (tracing), `REDIS_URL` (episodic memory)

### 3. Start Redis (optional)

```bash
docker run -d -p 6379:6379 redis:7-alpine
```

ChromaDB persists locally to `./chroma_db/` automatically.

---

## Running

### Live run (calls real APIs)

```bash
uv run python main.py
```

### Integration tests (replay harness — no live API calls)

```bash
uv run pytest tests/test_graph_replay.py -v
```

### Eval suite (GEval quality metrics against golden dataset)

```bash
uv run pytest tests/test_evals.py -v
```

> **Rate limits:** The free Groq tier is 12,000 TPM / 100,000 TPD. Tests use a module-scoped fixture so the graph runs exactly once per `pytest` session. Run test suites on separate days if you hit the daily limit.

---

## Sample Output

```
Starting CompeteIQ monitoring run... [LLM: Groq llama-3.3-70b-versatile]

============================================================
## CompeteIQ Intelligence Brief
**Run ID:** a3f7c2b1 | **Competitors:** ['zomato', 'blinkit']

### HIGH PRIORITY (Impact 8-10)
- **Blinkit Zero-Fee Delhi** (Impact 9): Blinkit cut delivery fees to Rs0 on
  orders >= Rs99 across Delhi. SwiftMart's Rs25 base fee is now uncompetitive
  in our 67%-share market. Launch a 30-day fee waiver campaign immediately.

### MEDIUM PRIORITY (Impact 5-7)
- **Zomato ML Hiring** (Impact 6): 15 ML engineer roles targeting ETA
  prediction — signals a push to close Zomato's speed gap with quick-commerce.
  Monitor delivery SLA changes over the next 90 days.

### OPPORTUNITIES
- **Zepto App Degradation** (Impact 7): Zepto Bangalore rating dropped 4.2->3.5
  with 500+ 1-star reviews. Targeted SwiftMart push notifications to lapsed
  Zepto users in Bangalore could capture churned customers this week.

### Summary
- Signals detected: 4
- Threats: 1 | Opportunities: 2 | Neutral: 1
- Most urgent action: Launch Delhi delivery-fee match within 48 hours
============================================================

Signals collected : 4
Signals analyzed  : 4
Tool calls made   : 6
Retry count       : 0
Total latency     : 6842ms

Per-node latencies:
  collector_sub_agent            4201ms
  analysis_agent                 1893ms
  orchestrator                    512ms
  brief_writer                    236ms
  aggregator                        1ms

Run logged to data/run_log.jsonl
Run saved to episodic memory (Redis)
```

---

## Project Structure

```
competitor-analysis/
  agents/
    llm.py            # Groq-only LLM singleton
    schema.py         # RawSignal / AnalyzedSignal Pydantic models
    state.py          # CompeteIQState TypedDict with custom reducers
  data/
    mock_signals.json # Realistic mock signals for offline testing
    run_log.py        # Append-only JSONL logger
  evals/
    golden_dataset.json   # 4 hand-labeled test cases
    groq_judge.py         # DeepEvalBaseLLM wrapper using Groq
    metrics.py            # AssessmentCorrectness + ReasoningQuality GEval metrics
  graph/
    workflow.py       # 7 nodes + 4 edge functions (full pipeline)
  harness/
    replay.py         # Deterministic test harness (patches live tools with fixtures)
  memory/
    episodic.py       # Redis run history store
    semantic.py       # ChromaDB signal store
  prompts/
    constants.py      # All system prompt strings
  skills/
    tools.py          # 3 LangChain tools with tenacity retry
  tests/
    test_graph_replay.py  # 7 integration tests (module-scoped fixture)
    test_evals.py         # 6 eval tests + regression gate (threshold 0.6)
  utils/
    guard.py          # Hallucination guard (competitor + assessment validation)
    observability.py  # traced_node decorator + compute_run_metrics
  config.py           # Env vars + constants
  main.py             # Async entrypoint (asyncio.run)
```

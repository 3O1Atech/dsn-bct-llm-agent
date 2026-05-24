# Nigerian Multi-Agent Swarm — DSN x BCT LLM Agent Challenge

A multi-agent system for cross-domain recommendation and Nigerian-contextualised review generation, built for the DSN x BCT Hackathon 3.0.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SHARED INFRASTRUCTURE                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ ChromaDB     │  │ Persona      │  │ Nigerian Context │  │
│  │ Vector Store │  │ Engine       │  │ Knowledge Base   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌──────────────────────┐              ┌──────────────────────────┐
│     TASK A SWARM     │              │       TASK B SWARM        │
│                      │              │                            │
│  1. PersonaAgent     │              │  1. CoordinatorAgent       │
│  2. RetrieverAgent   │              │  2. HistoryAgent           │
│  3. RatingAgent      │              │  3. SemanticAgent          │
│  4. ReviewAgent      │              │  4. CrossDomainAgent       │
│  5. NigerianVoice    │              │  5. NigerianContextAgent   │
│  6. ConsistencyAgent │              │  6. GraderAgent            │
│                      │              │  7. RankerAgent            │
│                      │              │  8. ExplanationAgent       │
└──────────────────────┘              └──────────────────────────┘
```

### Task A — User Modeling Pipeline
Each agent passes a shared `state` dict through the chain:

| Agent | Input | Output |
|---|---|---|
| PersonaAgent | user_id, raw persona | enriched_persona (from vector DB history) |
| RetrieverAgent | enriched_persona, item_metadata | few_shot_examples (similar past reviews) |
| RatingAgent | persona, item, few_shot | predicted_rating (calibrated to user bias) |
| ReviewAgent | persona, item, few_shot, context | raw_review |
| NigerianVoiceAgent | raw_review, persona | nigerianized_review (Pidgin/local refs injected) |
| ConsistencyAgent | review, rating | final_review, final_rating (sentiment-rating aligned) |

### Task B — Recommendation Pipeline

| Agent | Role |
|---|---|
| CoordinatorAgent | Detects cross-domain intent from context |
| HistoryAgent | Builds taste profile; flags cold-start if <3 interactions |
| SemanticAgent | Queries ChromaDB item_metadata for top-30 candidates |
| CrossDomainAgent | Projects taste preferences into target domain via LLM |
| NigerianContextAgent | Boosts items with local relevance (city match, Nigerian tags) |
| GraderAgent | LLM scores each candidate 0-10 for user relevance |
| RankerAgent | Composite score: 0.4×semantic + 0.3×grader + 0.2×nigerian + 0.1×popularity |
| ExplanationAgent | Generates personalised explanation per recommendation |

---

## Datasets

This system was trained and evaluated on three public datasets:

| Dataset | Source | Usage |
|---|---|---|
| Amazon Reviews | [Amazon Review Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) | Primary user history + items |
| Goodreads | [UCSD Goodreads](https://mengtingwan.github.io/data/goodreads.html) | Cross-domain (books) |
| Yelp Nigerian | Yelp Open Dataset (Nigerian businesses) | Cultural localization |

The `data/` folder contains:
- `nigerian_context.json` — Nigerian cultural KB (Pidgin phrases, local references, seeding docs)
- `nigerian_items.json` / `nigerian_user_histories.json` — Nigerian sample data for cold-start and demo
- `sample_items.json` / `sample_user_history.json` — Generic samples for local testing

Raw datasets are not included (too large). Download them from the sources above, place in `raw_data/`, then run `python scripts/data_ingestion.py`.

---

## Quickstart

### Option 1 — Docker (recommended)

```bash
cp .env.example .env
# Add your API key to .env
docker-compose up --build
```

### Option 2 — Local

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your API key to .env

export PYTHONPATH=$(pwd):$PYTHONPATH
python scripts/data_ingestion.py   # process raw datasets
python scripts/seed_chroma.py      # build vector store
uvicorn main:app --host 0.0.0.0 --port 8000
```

Visit **http://localhost:8000/docs** for the interactive API explorer.

---

## Environment Variables

Copy `.env.example` to `.env`:

| Variable | Default | Description |
|---|---|---|
| `USE_KIMI` | `false` | Use Moonshot AI (Kimi) as LLM backend |
| `KIMI_API_KEY` | — | Required if USE_KIMI=true |
| `KIMI_MODEL` | `moonshot-v1-8k` | Kimi model variant |
| `USE_GEMINI` | `false` | Use Google Gemini as LLM backend |
| `GEMINI_API_KEY` | — | Required if USE_GEMINI=true |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model variant |
| `USE_OPENAI` | `false` | Use OpenAI as LLM backend |
| `OPENAI_API_KEY` | — | Required if USE_OPENAI=true |
| `MODEL_NAME` | `HuggingFaceH4/zephyr-7b-beta` | Local HF model (fallback) |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Vector DB persistence path |
| `NIGERIAN_INJECTION_RATE` | `0.4` | Localization intensity (0.0–1.0) |

---

## API Reference

### Task A — Generate Review

```bash
curl -X POST http://localhost:8000/api/v1/task-a/generate-review \
  -H "Content-Type: application/json" \
  -d '{
    "user_persona": {
      "user_id": "lagos_hustler_01",
      "demographics": "25yo, Lagos Island",
      "linguistic_register": "pidgin",
      "rating_bias": 0.3,
      "category_preferences": ["food", "transport"],
      "typical_review_length": "short",
      "emotional_triggers": ["NEPA", "traffic"]
    },
    "item_metadata": {
      "item_id": "rest_001",
      "name": "Mama Put Jollof Spot",
      "category": "food",
      "description": "Local buka serving smoky party jollof and grilled turkey",
      "attributes": {"cuisine": "Nigerian", "price_range": "₦", "location": "Yaba"}
    },
    "context": {
      "time_of_day": "afternoon",
      "day_of_week": "weekday",
      "mood_hint": "hungry after danfo struggle"
    }
  }'
```

### Task B — Recommend

```bash
curl -X POST http://localhost:8000/api/v1/task-b/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_persona": {
      "user_id": "lagos_hustler_01",
      "demographics": "25yo, Lagos Island",
      "linguistic_register": "pidgin",
      "taste_vector": {"food": ["spicy", "cheap", "local"], "movies": ["Nollywood", "comedy"]},
      "cross_domain_signals": ["likes_local_culture", "budget_conscious"]
    },
    "context": {
      "conversation_history": [],
      "current_intent": "weekend movie night",
      "domain": "movies",
      "constraints": {"budget": "₦2000", "location": "Lagos"}
    },
    "top_k": 3
  }'
```

---

## Evaluation

Run the full baseline evaluation (requires processed data and seeded ChromaDB):

```bash
python scripts/baseline_eval.py
```

Results are saved to `results/baseline_task_a.json`, `results/baseline_task_b.json`, and `results/baseline_summary.csv`.

---

## Project Structure

```
bluechip/
├── main.py                  # FastAPI app entry point
├── task_a/                  # Task A — User Modeling
│   ├── agents.py            # 6-agent pipeline (Persona → Consistency)
│   ├── pipeline.py          # TaskASwarm orchestrator
│   └── api.py               # FastAPI router
├── task_b/                  # Task B — Recommendation
│   ├── agents.py            # 8-agent pipeline (Coordinator → Explanation)
│   ├── pipeline.py          # TaskBSwarm orchestrator
│   └── api.py               # FastAPI router
├── shared/
│   ├── llm_backend.py       # LLM abstraction (Kimi / Gemini / OpenAI / local)
│   ├── vector_store.py      # ChromaDB wrapper
│   ├── persona_engine.py    # Behavioural persona builder
│   ├── nigerian_kb.py       # Nigerian cultural knowledge base
│   └── config.py            # Environment config + scoring weights
├── eval/
│   └── metrics.py           # RMSE, ROUGE-L, BERTScore, NDCG, Hit Rate
├── scripts/
│   ├── data_ingestion.py    # Amazon + Goodreads + Yelp processing
│   ├── seed_chroma.py       # ChromaDB population
│   └── baseline_eval.py     # Full evaluation runner
├── data/                    # Sample + Nigerian reference data
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

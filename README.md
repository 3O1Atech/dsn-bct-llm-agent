# 🇳🇬 Nigerian Multi-Agent Swarm

Cross-domain recommendation and review generation system with deep Nigerian localization.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SHARED INFRASTRUCTURE                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ ChromaDB     │  │ User Persona │  │ Nigerian Context │  │
│  │ (Vector DB)  │  │ Engine       │  │ Knowledge Base   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌──────────────────────┐              ┌──────────────────────────┐
│   TASK A SWARM       │              │    TASK B SWARM            │
│  Persona → Retriever │              │  Coordinator → History     │
│  → Rating → Review   │              │  → Semantic → CrossDomain  │
│  → NigerianVoice     │              │  → NigerianContext → Grader│
│  → Consistency       │              │  → Ranker → Explanation    │
└──────────────────────┘              └──────────────────────────┘
```

## One-Command Setup

```bash
docker-compose up --build
```

## API Examples

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

## Environment Variables

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `HuggingFaceH4/zephyr-7b-beta` | HF model path |
| `USE_OPENAI` | `false` | Toggle OpenAI fallback |
| `OPENAI_API_KEY` | - | Required if USE_OPENAI=true |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Vector DB path |
| `NIGERIAN_INJECTION_RATE` | `0.4` | Localization intensity |

## Team

- Built for the Cross-Domain Agent Swarm Hackathon

## Notes

- First run seeds ChromaDB from `data/` automatically.
- If no GPU is available and local model loading fails, set `USE_OPENAI=true`.
- Nigerian signals are injected via `NigerianVoiceAgent` (Task A) and `NigerianContextAgent` (Task B).

from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from task_b.pipeline import TaskBSwarm

router = APIRouter(prefix="/task-b", tags=["Task B — Recommendation"])

swarm = TaskBSwarm()


class UserPersonaB(BaseModel):
    user_id: str
    demographics: Optional[str] = ""
    linguistic_register: Optional[str] = "mixed"
    taste_vector: Optional[dict] = {}
    cross_domain_signals: Optional[list] = []


class ContextB(BaseModel):
    conversation_history: Optional[list] = Field(default_factory=list)
    current_intent: Optional[str] = ""
    domain: Optional[str] = ""
    constraints: Optional[dict] = {}


class RecommendRequest(BaseModel):
    """
    Example Nigerian request:
    {
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
    }
    """
    user_persona: UserPersonaB
    context: ContextB
    top_k: Optional[int] = 10


class RecommendationItem(BaseModel):
    rank: int
    item_id: str
    item_name: str
    domain: str
    score: float
    explanation: str
    cold_start_derived: bool


class RecommendResponse(BaseModel):
    """
    Example response:
    {
      "recommendations": [
        {
          "rank": 1,
          "item_id": "mov_001",
          "item_name": "The Wedding Party",
          "domain": "movies",
          "score": 0.93,
          "explanation": "You love Nollywood and comedy...",
          "cold_start_derived": false
        }
      ],
      "reasoning_chain": "User profile shows strong Nollywood preference...",
      "cold_start_flag": false,
      "inference_time_ms": 850
    }
    """
    recommendations: list
    reasoning_chain: str
    cold_start_flag: bool
    inference_time_ms: int


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    input_data = {
        "user_persona": request.user_persona.model_dump(),
        "context": request.context.model_dump(),
        "top_k": request.top_k,
    }
    result = swarm.run(input_data)
    return RecommendResponse(**result)

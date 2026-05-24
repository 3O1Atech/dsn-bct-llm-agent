from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from task_a.pipeline import TaskASwarm

router = APIRouter(prefix="/task-a", tags=["Task A — Review Generation"])

swarm = TaskASwarm()


class UserPersona(BaseModel):
    user_id: str
    demographics: Optional[str] = ""
    linguistic_register: Optional[str] = "mixed"
    rating_bias: Optional[float] = 0.0
    category_preferences: Optional[list] = []
    typical_review_length: Optional[str] = "medium"
    emotional_triggers: Optional[list] = []


class ItemMetadata(BaseModel):
    item_id: str
    name: str
    category: str
    description: Optional[str] = ""
    attributes: Optional[dict] = {}


class Context(BaseModel):
    time_of_day: Optional[str] = ""
    day_of_week: Optional[str] = ""
    mood_hint: Optional[str] = ""


class GenerateReviewRequest(BaseModel):
    """
    Example Nigerian request:
    {
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
    }
    """
    user_persona: UserPersona
    item_metadata: ItemMetadata
    context: Optional[Context] = Field(default_factory=Context)


class BehavioralMetadata(BaseModel):
    length_category: str
    pidgin_ratio: float
    emoji_count: int
    caps_lock_usage: bool
    local_references: list


class GenerateReviewResponse(BaseModel):
    """
    Example response:
    {
      "rating": 4.5,
      "review_text": "This jollof sweet die! Smoke dey enter the rice well well...",
      "persona_confidence": 0.91,
      "behavioral_metadata": {
        "length_category": "short",
        "pidgin_ratio": 0.42,
        "emoji_count": 0,
        "caps_lock_usage": false,
        "local_references": ["NEPA", "danfo", "jollof", "correct guy"]
      },
      "generation_time_ms": 1200
    }
    """
    rating: float
    review_text: str
    persona_confidence: float
    behavioral_metadata: BehavioralMetadata
    generation_time_ms: int


@router.post("/generate-review", response_model=GenerateReviewResponse)
async def generate_review(request: GenerateReviewRequest):
    input_data = {
        "user_persona": request.user_persona.model_dump(),
        "item_metadata": request.item_metadata.model_dump(),
        "context": request.context.model_dump(),
    }
    result = swarm.run(input_data)
    return GenerateReviewResponse(**result)

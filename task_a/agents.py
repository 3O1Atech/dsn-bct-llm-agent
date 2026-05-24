import re
import random
from typing import Dict, List

from shared.llm_backend import get_llm
from shared.vector_store import get_chroma
from shared.persona_engine import PersonaEngine
from shared.nigerian_kb import inject_nigerian_context


class PersonaAgent:
    def run(self, state: Dict) -> Dict:
        try:
            user_persona = state.get("user_persona", {})
            user_id = user_persona.get("user_id", "unknown")
            chroma = get_chroma()
            # Enrich with past reviews from vector DB
            results = chroma.query("user_reviews", f"user {user_id} past reviews", n_results=5)
            past_reviews = []
            if results and results.get("documents"):
                for doc_list, meta_list in zip(results["documents"], results["metadatas"]):
                    for doc, meta in zip(doc_list, meta_list):
                        if meta.get("user_id") == user_id:
                            past_reviews.append({"text": doc, **meta})

            engine = PersonaEngine()
            enriched = engine.build_persona(user_id, past_reviews)
            # Merge with provided persona (provided takes precedence)
            for k, v in user_persona.items():
                enriched[k] = v
            state["enriched_persona"] = enriched
        except Exception as e:
            print(f"PersonaAgent error: {e}")
            state["enriched_persona"] = state.get("user_persona", {})
        return state


class RetrieverAgent:
    def run(self, state: Dict) -> Dict:
        try:
            item_metadata = state.get("item_metadata", {})
            category = item_metadata.get("category", "general")
            query_text = f"{item_metadata.get('name', '')} {item_metadata.get('description', '')} {category}"
            user_id = state.get("enriched_persona", {}).get("user_id", "unknown")

            chroma = get_chroma()
            where_filter = {"user_id": user_id} if user_id != "unknown" else None
            results = chroma.query("user_reviews", query_text, n_results=10, where=where_filter)

            few_shot = []
            if results and results.get("documents"):
                for doc_list, meta_list in zip(results["documents"], results["metadatas"]):
                    for doc, meta in zip(doc_list, meta_list):
                        few_shot.append({"text": doc, "rating": meta.get("rating", 3), "item": meta.get("item_id", "unknown")})
                        if len(few_shot) >= 3:
                            break
                    if len(few_shot) >= 3:
                        break

            state["few_shot_examples"] = few_shot
        except Exception as e:
            print(f"RetrieverAgent error: {e}")
            state["few_shot_examples"] = []
        return state


class RatingAgent:
    def run(self, state: Dict) -> Dict:
        try:
            persona = state.get("enriched_persona", {})
            item = state.get("item_metadata", {})
            few_shot = state.get("few_shot_examples", [])
            bias = persona.get("rating_bias", 0.0)
            avg_rating = persona.get("avg_rating", 3.0)

            prompt = self._build_prompt(persona, item, few_shot)
            llm = get_llm()
            raw = llm.generate(prompt, max_tokens=32, temperature=0.3)
            rating = self._parse_rating(raw)

            # Calibrate with persona bias and average
            calibrated = rating + bias
            # Pull toward personal average
            calibrated = (calibrated + avg_rating) / 2
            calibrated = max(1.0, min(5.0, calibrated))
            # Round to nearest 0.5
            calibrated = round(calibrated * 2) / 2

            state["predicted_rating"] = calibrated
        except Exception as e:
            print(f"RatingAgent error: {e}")
            state["predicted_rating"] = 3.0
        return state

    def _build_prompt(self, persona: Dict, item: Dict, few_shot: List[Dict]) -> str:
        lines = [
            "You are a rating prediction model. Output ONLY a number between 1.0 and 5.0 (half-stars allowed).",
            f"User avg rating: {persona.get('avg_rating', 3.0)}. Category preferences: {persona.get('category_preferences', [])}.",
            f"Item: {item.get('name', '')} | Category: {item.get('category', '')} | Description: {str(item.get('description', ''))[:100]}.",
            "Past reviews:"
        ]
        for ex in few_shot[:3]:
            text = str(ex.get('text', ''))[:120]
            lines.append(f"- Rating {ex.get('rating', 3)}: {text}")
        lines.append("Predicted rating (number only):")
        return "\n".join(lines)

    def _parse_rating(self, text: str) -> float:
        text = text.strip()
        # Try to find first float/number
        m = re.search(r"(\d+\.\d+|\d+)", text)
        if m:
            val = float(m.group(1))
            return max(1.0, min(5.0, val))
        return 3.0


class ReviewAgent:
    def run(self, state: Dict) -> Dict:
        try:
            persona = state.get("enriched_persona", {})
            item = state.get("item_metadata", {})
            few_shot = state.get("few_shot_examples", [])
            context = state.get("context", {})
            length = persona.get("typical_review_length", "medium")
            register = persona.get("linguistic_register", "mixed")

            prompt = self._build_prompt(persona, item, few_shot, context, length, register)
            llm = get_llm()
            raw_review = llm.generate(prompt, max_tokens=256, temperature=0.8)
            state["raw_review"] = raw_review.strip()
        except Exception as e:
            print(f"ReviewAgent error: {e}")
            state["raw_review"] = f"The {item.get('name', 'item')} was okay."
        return state

    def _build_prompt(self, persona: Dict, item: Dict, few_shot: List[Dict], context: Dict, length: str, register: str) -> str:
        lines = [
            f"You are this user: {persona.get('demographics', '')}. Register: {register}. Typical length: {length}.",
            "Here are 3 of their past reviews:"
        ]
        for ex in few_shot[:3]:
            lines.append(f"- {str(ex.get('text', ''))[:150]}")
        lines.append(f"Write a review for: {item.get('name', '')} ({item.get('category', '')}).")
        lines.append(f"Description: {str(item.get('description', ''))[:100]}.")
        lines.append(f"Context: time={context.get('time_of_day', '')}, mood={context.get('mood_hint', '')}.")
        lines.append(f"Rules: Match their typical length ({length}). Match their tone ({register}). Do not mention you are an AI.")
        return "\n".join(lines)


class NigerianVoiceAgent:
    def run(self, state: Dict) -> Dict:
        try:
            raw_review = state.get("raw_review", "")
            persona = state.get("enriched_persona", {})
            item = state.get("item_metadata", {})
            category = item.get("category", "general")

            nigerianized = inject_nigerian_context(raw_review, persona, category)
            state["nigerianized_review"] = nigerianized
        except Exception as e:
            print(f"NigerianVoiceAgent error: {e}")
            state["nigerianized_review"] = state.get("raw_review", "")
        return state


class ConsistencyAgent:
    def run(self, state: Dict) -> Dict:
        try:
            review = state.get("nigerianized_review", "")
            rating = state.get("predicted_rating", 3.0)

            # Simple sentiment heuristic
            positive_words = ["good", "great", "excellent", "love", "best", "amazing", "sweet", "perfect", "awesome", "fantastic", "make sense", "correct", "enjoy"]
            negative_words = ["bad", "terrible", "worst", "hate", "disappointing", "poor", "awful", "nonsense", "wahala", "fail", "no dey", "yeye", "buru"]
            lower = review.lower()
            pos = sum(1 for w in positive_words if w in lower)
            neg = sum(1 for w in negative_words if w in lower)

            sentiment_score = (pos - neg) / max(pos + neg, 1)
            # Normalize sentiment_score to 0-1 where 0=very negative, 1=very positive
            normalized_sentiment = (sentiment_score + 1) / 2
            rating_norm = (rating - 1) / 4  # 1-5 -> 0-1

            alignment = 1.0 - abs(normalized_sentiment - rating_norm)
            confidence = max(0.0, min(1.0, alignment))

            # Flag and regenerate if severe mismatch
            if rating >= 4.0 and normalized_sentiment < 0.3 and confidence < 0.5:
                # Regenerate once
                try:
                    llm = get_llm()
                    prompt = (
                        f"The user gave a rating of {rating} stars but the review sounds too negative. "
                        f"Rewrite this review to be genuinely positive while keeping Nigerian flavor: {review}"
                    )
                    review = llm.generate(prompt, max_tokens=256, temperature=0.7)
                    confidence = min(1.0, confidence + 0.2)
                except Exception as regen_err:
                    print(f"Regeneration failed: {regen_err}")

            if rating <= 2.0 and normalized_sentiment > 0.7 and confidence < 0.5:
                try:
                    llm = get_llm()
                    prompt = (
                        f"The user gave a rating of {rating} stars but the review sounds too positive. "
                        f"Rewrite this review to be genuinely critical while keeping Nigerian flavor: {review}"
                    )
                    review = llm.generate(prompt, max_tokens=256, temperature=0.7)
                    confidence = min(1.0, confidence + 0.2)
                except Exception as regen_err:
                    print(f"Regeneration failed: {regen_err}")

            state["final_review"] = review.strip()
            state["final_rating"] = rating
            state["confidence_score"] = round(confidence, 2)
        except Exception as e:
            print(f"ConsistencyAgent error: {e}")
            state["final_review"] = state.get("nigerianized_review", "")
            state["final_rating"] = state.get("predicted_rating", 3.0)
            state["confidence_score"] = 0.5
        return state

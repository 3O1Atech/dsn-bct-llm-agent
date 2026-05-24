import json
import math
import re
from typing import List, Dict
from collections import Counter


class PersonaEngine:
    def build_persona(self, user_id: str, raw_history: List[Dict]) -> Dict:
        if not raw_history:
            return self._default_persona(user_id)

        ratings = [r["rating"] for r in raw_history if "rating" in r]
        texts = [r.get("text", "") for r in raw_history]
        domains = [r.get("domain", "unknown") for r in raw_history]
        categories = [r.get("category", "general") for r in raw_history]

        avg_rating = sum(ratings) / len(ratings) if ratings else 3.0
        rating_std = math.sqrt(sum((r - avg_rating) ** 2 for r in ratings) / len(ratings)) if ratings else 0.0
        review_count = len(raw_history)

        cat_dist = dict(Counter(categories))
        domain_dist = dict(Counter(domains))

        all_text = " ".join(texts).lower()
        word_count = len(all_text.split())

        linguistic_register = self._detect_register(all_text)
        typical_length = self._typical_length(word_count, len(texts))
        emoji_rate = self._emoji_rate(all_text)
        caps_rate = self._caps_rate(all_text)
        pidgin_ratio = self._pidgin_ratio(all_text)
        emotional_triggers = self._detect_emotional_triggers(all_text)

        persona = {
            "user_id": user_id,
            "avg_rating": round(avg_rating, 2),
            "rating_std": round(rating_std, 2),
            "review_count": review_count,
            "category_distribution": cat_dist,
            "domain_distribution": domain_dist,
            "linguistic_register": linguistic_register,
            "typical_length": typical_length,
            "emoji_rate": round(emoji_rate, 3),
            "caps_rate": round(caps_rate, 3),
            "pidgin_ratio": round(pidgin_ratio, 3),
            "emotional_triggers": emotional_triggers,
            "computed_at": "",
        }
        return persona

    def _default_persona(self, user_id: str) -> Dict:
        return {
            "user_id": user_id,
            "avg_rating": 3.0,
            "rating_std": 0.0,
            "review_count": 0,
            "category_distribution": {},
            "domain_distribution": {},
            "linguistic_register": "mixed",
            "typical_length": "medium",
            "emoji_rate": 0.0,
            "caps_rate": 0.0,
            "pidgin_ratio": 0.0,
            "emotional_triggers": [],
            "computed_at": "",
        }

    def _detect_register(self, text: str) -> str:
        pidgin_markers = ["dey", "wahala", "shey", "abeg", "omo", "sha", "sotey", "sabi", "no dey", "sweet die", "correct guy"]
        sarcasm_markers = ["wow", "such", "much ", "oscar-worthy", "revolutionary", "miracles", "innovation", "👏"]
        formal_markers = ["excellent", "remarkable", "disappointing", "commendable", "attentive", "operational"]

        pidgin_score = sum(1 for m in pidgin_markers if m in text)
        sarcasm_score = sum(1 for m in sarcasm_markers if m in text)
        formal_score = sum(1 for m in formal_markers if m in text)

        scores = {"pidgin": pidgin_score, "sarcastic": sarcasm_score, "formal": formal_score}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "mixed"
        return best

    def _typical_length(self, total_words: int, count: int) -> str:
        if count == 0:
            return "medium"
        avg = total_words / count
        if avg < 50:
            return "short"
        if avg > 150:
            return "long"
        return "medium"

    def _emoji_rate(self, text: str) -> float:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        emojis = emoji_pattern.findall(text)
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        return sum(len(e) for e in emojis) / total_chars

    def _caps_rate(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        caps = [c for c in letters if c.isupper()]
        return len(caps) / len(letters)

    def _pidgin_ratio(self, text: str) -> float:
        pidgin_words = ["dey", "wahala", "shey", "abeg", "omo", "sha", "sotey", "sabi", "no dey", "sweet die",
                        "correct guy", "make sense", "how far", "na", "sotey", "well well", "small small",
                        "gbas gbos", "chop life", "no wahala"]
        words = text.split()
        if not words:
            return 0.0
        matches = sum(1 for w in words if w in pidgin_words)
        return matches / len(words)

    def _detect_emotional_triggers(self, text: str) -> List[str]:
        triggers = {
            "NEPA": ["nepa", "phcn", "power", "light", "outage", "generator"],
            "traffic": ["traffic", "go-slow", "danfo", "okada", "bridge", "expressway"],
            "fuel": ["fuel", "petrol", "scarcity", "queue", "pump"],
            "price": ["price", "cost", "expensive", "cheap", "naira", "₦"],
            "service": ["service", "wait", "rude", "staff", "customer"],
        }
        found = []
        lower = text.lower()
        for trigger, keywords in triggers.items():
            if any(kw in lower for kw in keywords):
                found.append(trigger)
        return found

    def serialize(self, persona: Dict) -> str:
        return json.dumps(persona, ensure_ascii=False, indent=2)

    def deserialize(self, raw: str) -> Dict:
        return json.loads(raw)

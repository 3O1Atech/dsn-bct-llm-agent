import time
from typing import Dict

from task_a.agents import (
    PersonaAgent,
    RetrieverAgent,
    RatingAgent,
    ReviewAgent,
    NigerianVoiceAgent,
    ConsistencyAgent,
)


class TaskASwarm:
    def __init__(self):
        self.persona_agent = PersonaAgent()
        self.retriever_agent = RetrieverAgent()
        self.rating_agent = RatingAgent()
        self.review_agent = ReviewAgent()
        self.nigerian_voice_agent = NigerianVoiceAgent()
        self.consistency_agent = ConsistencyAgent()

    def run(self, input_data: Dict) -> Dict:
        start = time.time()
        state = dict(input_data)

        state = self.persona_agent.run(state)
        state = self.retriever_agent.run(state)
        state = self.rating_agent.run(state)
        state = self.review_agent.run(state)
        state = self.nigerian_voice_agent.run(state)
        state = self.consistency_agent.run(state)

        elapsed = int((time.time() - start) * 1000)

        final_review = state.get("final_review", "")
        persona = state.get("enriched_persona", {})

        # Compute behavioral metadata
        local_refs = self._extract_local_refs(final_review)
        pidgin_ratio = persona.get("pidgin_ratio", 0.0)
        caps = any(c.isupper() for c in final_review if c.isalpha())
        emoji_count = len([c for c in final_review if c in "😀😃😄😁😆😅😂🤣😊😇🙂🙃😉😌😍🥰😘😗😙😚😋😛😝😜🤪🤨🧐🤓😎🥸🤩🥳😏😒😞😔😟😕🙁☹️😣😖😫😩🥺😢😭😤😠😡🤬🤯😳🥵🥶😱😨😰😥😓🤗🤔🤭🤫🤥😶😐😑😬🙄😯😦😧😮😲🥱😴🤤😪😵🤐🥴🤢🤮🤧😷🤒🤕🤑🤠😈👿👹👺🤡💩👻💀☠️👽👾🤖🎃😺😸😹😻😼😽🙀😿😾🤲👐🙌👏🤝👍👎👊✊🤛🤜🤞✌️🤟🤘👌🤏☝️👆👇👉👈✋🤚🖐🖖👋🤙💪🦾🖕✍️🙏🦶🦵🦿🦻👂🦻👃🧠🦷🦴👀👁👅👄💋🩸🔥💥✨🌟💫💯💢💥💫💦💨🕳💣💬👁️‍🗨️🗨🗯💭💤"])
        words = final_review.split()
        length_cat = "short" if len(words) < 50 else ("long" if len(words) > 150 else "medium")

        return {
            "rating": state.get("final_rating", 3.0),
            "review_text": final_review,
            "persona_confidence": state.get("confidence_score", 0.5),
            "behavioral_metadata": {
                "length_category": length_cat,
                "pidgin_ratio": round(pidgin_ratio, 2),
                "emoji_count": emoji_count,
                "caps_lock_usage": caps,
                "local_references": local_refs,
            },
            "generation_time_ms": elapsed,
        }

    def _extract_local_refs(self, text: str) -> list:
        from shared.nigerian_kb import LOCAL_REFERENCES
        found = []
        lower = text.lower()
        for cat, refs in LOCAL_REFERENCES.items():
            for ref in refs:
                if ref.lower() in lower and ref not in found:
                    found.append(ref)
        cities = ["Lagos", "Abuja", "Port Harcourt", "Ibadan", "Kano"]
        for city in cities:
            if city.lower() in lower and city not in found:
                found.append(city)
        return found

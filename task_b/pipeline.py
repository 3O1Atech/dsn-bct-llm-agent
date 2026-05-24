import time
from typing import Dict

from task_b.agents import (
    CoordinatorAgent,
    HistoryAgent,
    SemanticAgent,
    CrossDomainAgent,
    NigerianContextAgent,
    GraderAgent,
    RankerAgent,
    ExplanationAgent,
)


class TaskBSwarm:
    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.history = HistoryAgent()
        self.semantic = SemanticAgent()
        self.cross_domain = CrossDomainAgent()
        self.nigerian = NigerianContextAgent()
        self.grader = GraderAgent()
        self.ranker = RankerAgent()
        self.explanation = ExplanationAgent()

    def run(self, input_data: Dict) -> Dict:
        start = time.time()
        state = dict(input_data)

        state = self.coordinator.run(state)
        state = self.history.run(state)

        if state.get("cold_start", False):
            # Cold-start path: skip deep history, use demographic baseline + popularity + nigerian context
            state["taste_profile"] = state.get("taste_profile", {"preferred_genres": ["general"]})

        state = self.semantic.run(state)
        state = self.cross_domain.run(state)
        state = self.nigerian.run(state)
        state = self.grader.run(state)
        state = self.ranker.run(state)
        state = self.explanation.run(state)

        elapsed = int((time.time() - start) * 1000)

        top_k = input_data.get("top_k", 10)
        ranked = state.get("ranked_items", [])[:top_k]
        explanations = state.get("explanations", [])
        exp_map = {e["item_id"]: e["explanation"] for e in explanations}

        recommendations = []
        for idx, item in enumerate(ranked, start=1):
            recommendations.append({
                "rank": idx,
                "item_id": item["item_id"],
                "item_name": item["name"],
                "domain": item["domain"],
                "score": item.get("composite_score", 0.0),
                "explanation": exp_map.get(item["item_id"], "Recommended based on your profile."),
                "cold_start_derived": state.get("cold_start", False),
            })

        reasoning = self._build_reasoning(state)

        return {
            "recommendations": recommendations,
            "reasoning_chain": reasoning,
            "cold_start_flag": state.get("cold_start", False),
            "inference_time_ms": elapsed,
        }

    def _build_reasoning(self, state: Dict) -> str:
        parts = []
        if state.get("cold_start", False):
            parts.append("Cold-start user; using demographic baseline and popularity.")
        else:
            parts.append("User profile shows established taste preferences.")
        if state.get("needs_cross_domain", False):
            parts.append("Cross-domain projection applied to map tastes to new domain.")
        parts.append("Items filtered by LLM grader and ranked by composite score.")
        return " ".join(parts)

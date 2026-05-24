import re
import random
from typing import Dict, List, Optional

from shared.llm_backend import get_llm
from shared.vector_store import get_chroma
from shared.nigerian_kb import LOCAL_REFERENCES, CITIES
import shared.config as config


class CoordinatorAgent:
    def run(self, state: Dict) -> Dict:
        try:
            context = state.get("context", {})
            history = context.get("conversation_history", [])
            current_intent = context.get("current_intent", "")
            domain = context.get("domain", "")
            user_domains = state.get("user_persona", {}).get("cross_domain_signals", [])

            needs_cross = False
            if domain:
                # Simple heuristic: if domain not in user's known history domains
                # We can approximate by checking if domain word appears in taste_vector keys
                taste = state.get("user_persona", {}).get("taste_vector", {})
                if taste and domain not in taste:
                    needs_cross = True

            state["needs_cross_domain"] = needs_cross
            state["current_intent"] = current_intent
            state["conversation_history"] = history
            state["target_domain"] = domain
        except Exception as e:
            print(f"CoordinatorAgent error: {e}")
            state["needs_cross_domain"] = False
            state["current_intent"] = ""
            state["conversation_history"] = []
            state["target_domain"] = ""
        return state


class HistoryAgent:
    def run(self, state: Dict) -> Dict:
        try:
            persona = state.get("user_persona", {})
            user_id = persona.get("user_id", "unknown")
            chroma = get_chroma()
            results = chroma.query("user_reviews", f"user {user_id} interactions", n_results=10)
            interactions = []
            if results and results.get("documents"):
                for doc_list, meta_list in zip(results["documents"], results["metadatas"]):
                    for doc, meta in zip(doc_list, meta_list):
                        if meta.get("user_id") == user_id:
                            interactions.append({"text": doc, **meta})

            cold_start = len(interactions) < 3  # threshold for meaningful history
            if cold_start:
                baseline = {
                    "taste_profile": {
                        "preferred_genres": ["general"],
                        "price_sensitivity": "medium",
                        "location_preference": persona.get("demographics", ""),
                    },
                    "cold_start": True,
                }
                state["taste_profile"] = baseline["taste_profile"]
                state["cold_start"] = True
            else:
                domains = {}
                for inter in interactions:
                    d = inter.get("domain", "general")
                    domains[d] = domains.get(d, 0) + 1
                taste_profile = {
                    "preferred_genres": list(domains.keys()),
                    "price_sensitivity": "medium",
                    "location_preference": persona.get("demographics", ""),
                    "interaction_count": len(interactions),
                }
                state["taste_profile"] = taste_profile
                state["cold_start"] = False
        except Exception as e:
            print(f"HistoryAgent error: {e}")
            state["taste_profile"] = {"preferred_genres": ["general"], "price_sensitivity": "medium"}
            state["cold_start"] = True
        return state


class SemanticAgent:
    def run(self, state: Dict) -> Dict:
        try:
            taste = state.get("taste_profile", {})
            target_domain = state.get("target_domain", "")
            query = " ".join(taste.get("preferred_genres", ["general"]))
            if target_domain:
                query += f" {target_domain}"

            chroma = get_chroma()
            results = chroma.query("item_metadata", query, n_results=30)
            candidates = []
            if results and results.get("documents"):
                for doc_list, meta_list, dist_list in zip(results["documents"], results["metadatas"], results["distances"]):
                    for doc, meta, dist in zip(doc_list, meta_list, dist_list):
                        # Convert distance to similarity score (approximate)
                        sim = max(0.0, 1.0 - float(dist))
                        candidates.append({
                            "item_id": meta.get("item_id", ""),
                            "name": meta.get("name", ""),
                            "domain": meta.get("domain", ""),
                            "category": meta.get("category", ""),
                            "description": doc,
                            "attributes": meta.get("attributes", {}),
                            "semantic_score": sim,
                        })
            seen = set()
            uniq = []
            for c in candidates:
                if c["item_id"] not in seen:
                    seen.add(c["item_id"])
                    uniq.append(c)
            state["semantic_candidates"] = uniq[:20]
        except Exception as e:
            print(f"SemanticAgent error: {e}")
            state["semantic_candidates"] = []
        return state


class CrossDomainAgent:
    def run(self, state: Dict) -> Dict:
        try:
            if not state.get("needs_cross_domain", False):
                state["projected_preferences"] = state.get("taste_profile", {})
                return state

            taste = state.get("taste_profile", {})
            target = state.get("target_domain", "")
            llm = get_llm()
            prompt = (
                f"User likes these categories: {taste.get('preferred_genres', [])}. "
                f"What {target} genres or types match these tastes? Output a short comma-separated list only."
            )
            raw = llm.generate(prompt, max_tokens=64, temperature=0.5)
            genres = [g.strip() for g in raw.split(",") if g.strip()]
            projected = dict(taste)
            projected["projected_genres"] = genres
            projected["original_domain"] = taste.get("preferred_genres", [])
            state["projected_preferences"] = projected
        except Exception as e:
            print(f"CrossDomainAgent error: {e}")
            state["projected_preferences"] = state.get("taste_profile", {})
        return state


class NigerianContextAgent:
    def run(self, state: Dict) -> Dict:
        try:
            candidates = state.get("semantic_candidates", [])
            persona = state.get("user_persona", {})
            demographics = persona.get("demographics", "")
            register = persona.get("linguistic_register", "mixed")

            user_city = None
            for city in CITIES:
                if city.lower() in demographics.lower():
                    user_city = city
                    break

            for cand in candidates:
                boost = 0.0
                attrs = cand.get("attributes", {})
                tags = attrs.get("tags", [])
                loc = attrs.get("location", "")

                if user_city and (user_city.lower() in str(loc).lower() or any(user_city.lower() in str(t).lower() for t in tags)):
                    boost += 0.15

                if register in ["pidgin", "mixed"]:
                    naija_tags = sum(1 for t in tags if any(ref.lower() in str(t).lower() for ref in sum(LOCAL_REFERENCES.values(), [])))
                    boost += 0.05 * naija_tags

                cand["nigerian_boost"] = round(min(0.3, boost), 3)

            state["localized_candidates"] = candidates
        except Exception as e:
            print(f"NigerianContextAgent error: {e}")
            state["localized_candidates"] = state.get("semantic_candidates", [])
        return state


class GraderAgent:
    def run(self, state: Dict) -> Dict:
        try:
            candidates = state.get("localized_candidates", [])
            persona = state.get("user_persona", {})
            llm = get_llm()
            filtered = []

            for cand in candidates:
                prompt = (
                    f"Rate relevance of this item to the user on a scale 0-10. "
                    f"User taste: {persona.get('taste_vector', {})}. "
                    f"Item: {cand['name']} | {cand['description']}. "
                    f"Output ONLY a number 0-10."
                )
                raw = llm.generate(prompt, max_tokens=16, temperature=0.2)
                score = self._parse_score(raw)
                if score >= 5.0:
                    cand["grader_score"] = score
                    filtered.append(cand)

            state["filtered_candidates"] = filtered
        except Exception as e:
            print(f"GraderAgent error: {e}")
            state["filtered_candidates"] = state.get("localized_candidates", [])
        return state

    def _parse_score(self, text: str) -> float:
        m = re.search(r"(\d+\.\d+|\d+)", text.strip())
        if m:
            return max(0.0, min(10.0, float(m.group(1))))
        return 5.0


class RankerAgent:
    def run(self, state: Dict) -> Dict:
        try:
            candidates = state.get("filtered_candidates", [])
            ranked = []
            for cand in candidates:
                sem = cand.get("semantic_score", 0.0)
                grd = cand.get("grader_score", 5.0) / 10.0  # normalize to 0-1
                nig = cand.get("nigerian_boost", 0.0)
                pop = cand.get("popularity", 0.5)
                score = (config.WEIGHT_SEMANTIC * sem + config.WEIGHT_GRADER * grd
                         + config.WEIGHT_NIGERIAN * nig + config.WEIGHT_POPULARITY * pop)
                cand["composite_score"] = round(score, 3)
                ranked.append(cand)

            ranked.sort(key=lambda x: x["composite_score"], reverse=True)
            state["ranked_items"] = ranked[:10]
        except Exception as e:
            print(f"RankerAgent error: {e}")
            state["ranked_items"] = []
        return state


class ExplanationAgent:
    def run(self, state: Dict) -> Dict:
        try:
            items = state.get("ranked_items", [])
            persona = state.get("user_persona", {})
            taste = persona.get("taste_vector", {})
            history = state.get("conversation_history", [])
            explanations = []

            for item in items:
                reasons = []
                item_tags = item.get("attributes", {}).get("tags", [])
                for domain, prefs in taste.items():
                    if isinstance(prefs, list):
                        for p in prefs:
                            if any(p.lower() in str(t).lower() for t in item_tags):
                                reasons.append(f"you like {p} in {domain}")

                if not reasons:
                    reasons.append("it matches your general interests")

                if history:
                    last_turn = history[-1]
                    reasons.append(f"based on your recent interest in '{last_turn}'")

                exp = f"Because {', '.join(reasons)}. {item['name']} is a strong match for your profile."
                if persona.get("linguistic_register") == "pidgin":
                    exp = f"You love {', '.join(taste.get('movies', taste.get('food', ['this kind thing'])))}. This one go sweet you well well."

                explanations.append({
                    "item_id": item["item_id"],
                    "explanation": exp,
                })

            state["explanations"] = explanations
        except Exception as e:
            print(f"ExplanationAgent error: {e}")
            state["explanations"] = []
        return state

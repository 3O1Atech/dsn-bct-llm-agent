import random
from typing import List, Dict
from shared.llm_backend import get_llm


def blind_test(real_reviews: List[str], generated_reviews: List[str], n_evaluators: int = 3) -> Dict:
    """
    Shuffle real and generated reviews. Return structure for evaluators to label.
    """
    pool = []
    for r in real_reviews:
        pool.append({"text": r, "source": "real"})
    for g in generated_reviews:
        pool.append({"text": g, "source": "generated"})
    random.shuffle(pool)

    # Simulate evaluator judgments with a heuristic LLM prompt
    llm = get_llm()
    judgments = []
    for item in pool:
        prompt = (
            f"Is the following review written by a real human or AI-generated? "
            f"Reply ONLY 'human' or 'ai'. Review: {item['text'][:300]}"
        )
        label = llm.generate(prompt, max_tokens=10, temperature=0.1).lower()
        ai_detected = "ai" in label or "generated" in label
        judgments.append({
            "text_snippet": item["text"][:100],
            "true_source": item["source"],
            "evaluator_label": "ai" if ai_detected else "human",
            "correct": (ai_detected and item["source"] == "generated") or (not ai_detected and item["source"] == "real"),
        })

    accuracy = sum(1 for j in judgments if j["correct"]) / max(len(judgments), 1)
    return {
        "evaluator_count": n_evaluators,
        "total_judgments": len(judgments),
        "accuracy": round(accuracy, 2),
        "judgments": judgments,
    }


def nigerianness_score(reviews: List[str]) -> Dict:
    """
    Heuristic + LLM prompt to score Nigerian cultural presence 1-5.
    """
    from shared.nigerian_kb import PIDGIN_PHRASES, LOCAL_REFERENCES
    scores = []
    for review in reviews:
        lower = review.lower()
        hits = 0
        for phrase_list in PIDGIN_PHRASES.values():
            for ph in phrase_list:
                if ph.lower() in lower:
                    hits += 1
        for ref_list in LOCAL_REFERENCES.values():
            for ref in ref_list:
                if ref.lower() in lower:
                    hits += 1
        # Normalize roughly to 1-5
        heuristic = min(5, 1 + hits)
        scores.append(heuristic)

    # LLM refinement on a sample
    llm = get_llm()
    sample = reviews[0] if reviews else ""
    prompt = (
        f"Rate how strongly Nigerian (pidgin, local references, culture) this review is on a scale 1-5. "
        f"Output ONLY a number. Review: {sample[:300]}"
    )
    llm_raw = llm.generate(prompt, max_tokens=10, temperature=0.2)
    try:
        llm_score = max(1, min(5, int("".join([c for c in llm_raw if c.isdigit()])[:1])))
    except Exception:
        llm_score = 3

    avg_score = round(sum(scores) / max(len(scores), 1), 2)
    return {
        "average_heuristic": avg_score,
        "llm_sample_score": llm_score,
        "individual_scores": scores,
    }

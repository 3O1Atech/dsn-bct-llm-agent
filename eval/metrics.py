import numpy as np
from typing import List


def compute_rmse(y_true: List[float], y_pred: List[float]) -> float:
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
    mse = np.mean([(t - p) ** 2 for t, p in zip(y_true, y_pred)])
    return float(np.sqrt(mse))


def compute_rouge_l(predictions: List[str], references: List[str]) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            scores.append(s["rougeL"].fmeasure)
        return {
            "rouge_l_f1": float(np.mean(scores)),
            "scores": scores,
        }
    except Exception as e:
        print(f"ROUGE error: {e}")
        return {"rouge_l_f1": 0.0, "scores": []}


def compute_bertscore(predictions: List[str], references: List[str]) -> dict:
    try:
        from bert_score import score
        P, R, F1 = score(predictions, references, lang="en", verbose=False)
        return {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }
    except Exception as e:
        print(f"BERTScore error: {e}")
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


def compute_ndcg(relevances: List[float], scores: List[float], k: int = 10) -> float:
    """Compute NDCG@k given true relevances and predicted scores."""
    if len(relevances) == 0 or len(scores) == 0:
        return 0.0
    # Sort by predicted scores descending
    paired = list(zip(scores, relevances))
    paired.sort(key=lambda x: x[0], reverse=True)
    ranked_rels = [r for _, r in paired[:k]]

    def dcg(rel_list):
        gain = 0.0
        for i, rel in enumerate(rel_list, start=1):
            gain += (2 ** rel - 1) / np.log2(i + 1)
        return gain

    ideal = sorted(relevances, reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    actual_dcg = dcg(ranked_rels)
    if ideal_dcg == 0:
        return 0.0
    return float(actual_dcg / ideal_dcg)


def compute_hit_rate(relevances: List[float], k: int = 10, threshold: float = 1.0) -> float:
    """Hit rate: proportion of queries with at least one relevant item in top-k."""
    if len(relevances) == 0:
        return 0.0
    # Here relevances is a flat list; for per-query we expect list of lists.
    # Simplification: treat as single query list
    top = relevances[:k]
    hits = int(any(r >= threshold for r in top))
    return float(hits)

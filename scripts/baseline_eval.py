import json
import csv
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from task_a.pipeline import TaskASwarm
from task_b.pipeline import TaskBSwarm
from shared.persona_engine import PersonaEngine
from eval.metrics import compute_rmse, compute_rouge_l, compute_bertscore, compute_hit_rate, compute_ndcg

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

WORKERS = int(os.getenv("EVAL_WORKERS", "10"))
MAX_SAMPLES = int(os.getenv("EVAL_MAX_SAMPLES", "0"))  # 0 = all

engine = PersonaEngine()


def build_taste_vector(train_history):
    taste_vector = {}
    for r in train_history:
        dom = r.get("domain", "general")
        cat = r.get("category", "general")
        taste_vector.setdefault(dom, []).append(cat)
    for k in taste_vector:
        taste_vector[k] = list(set(taste_vector[k]))
    return taste_vector


# ========================= TASK A =========================
print("\n=== Baseline Evaluation: Task A ===")

with open("processed_data/test_set_task_a.json", "r", encoding="utf-8") as f:
    test_a = json.load(f)

if MAX_SAMPLES:
    test_a = test_a[:MAX_SAMPLES]
print(f"Loaded {len(test_a)} Task A test samples  (workers={WORKERS})")


def run_task_a(args):
    i, sample = args
    swarm = TaskASwarm()
    user_id = sample["user_id"]
    train_history = sample["train_history"]
    persona = engine.build_persona(user_id, train_history)
    input_data = {
        "user_persona": {"user_id": user_id, **persona},
        "item_metadata": sample["item_metadata"],
        "context": {"time_of_day": "", "day_of_week": "", "mood_hint": ""},
    }
    result = swarm.run(input_data)
    return i, sample, result["rating"], result["review_text"]


predictions_a = [None] * len(test_a)
y_true = []
y_pred = []
text_preds = []
text_refs = []
done = 0

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(run_task_a, (i, s)): i for i, s in enumerate(test_a)}
    for fut in as_completed(futures):
        i, sample, pred_rating, pred_text = fut.result()
        predictions_a[i] = {
            "user_id": sample["user_id"],
            "item_id": sample["item_metadata"]["item_id"],
            "predicted_rating": pred_rating,
            "true_rating": sample["true_rating"],
            "predicted_text": pred_text,
            "true_text": sample["true_text"],
        }
        y_true.append(sample["true_rating"])
        y_pred.append(pred_rating)
        if sample["true_text"].strip():
            text_preds.append(pred_text)
            text_refs.append(sample["true_text"])
        done += 1
        if done % 100 == 0:
            print(f"  Task A progress: {done}/{len(test_a)}")

rmse = compute_rmse(y_true, y_pred)
rouge_res = compute_rouge_l(text_preds, text_refs) if text_preds else {"rouge_l_f1": 0.0}
bert_res = compute_bertscore(text_preds, text_refs) if text_preds else {"bertscore_f1": 0.0}

task_a_summary = {
    "rmse": rmse,
    "rouge_l_f1": rouge_res.get("rouge_l_f1", 0.0),
    "bertscore_f1": bert_res.get("bertscore_f1", 0.0),
    "num_samples": len(test_a),
    "num_text_samples": len(text_preds),
    "predictions": predictions_a,
}

with open(RESULTS_DIR / "baseline_task_a.json", "w", encoding="utf-8") as f:
    json.dump(task_a_summary, f, ensure_ascii=False, indent=2)

print(f"Task A complete -> RMSE={rmse:.4f}, ROUGE-L={task_a_summary['rouge_l_f1']:.4f}, BERTScore={task_a_summary['bertscore_f1']:.4f}")


# ========================= TASK B =========================
print("\n=== Baseline Evaluation: Task B ===")

with open("processed_data/test_set_task_b.json", "r", encoding="utf-8") as f:
    test_b = json.load(f)

if MAX_SAMPLES:
    test_b = test_b[:MAX_SAMPLES]
print(f"Loaded {len(test_b)} Task B test samples  (workers={WORKERS})")


def run_task_b(args):
    i, sample = args
    swarm = TaskBSwarm()
    user_id = sample["user_id"]
    train_history = sample["train_history"]
    persona = engine.build_persona(user_id, train_history)
    taste_vector = build_taste_vector(train_history)
    hidden_meta = sample["hidden_item_metadata"]
    input_data = {
        "user_persona": {
            "user_id": user_id,
            "demographics": "",
            "linguistic_register": persona.get("linguistic_register", "mixed"),
            "taste_vector": taste_vector,
            "cross_domain_signals": [],
        },
        "context": {
            "conversation_history": [],
            "current_intent": "recommendation",
            "domain": hidden_meta.get("domain", ""),
            "constraints": {},
        },
        "top_k": 10,
    }
    result = swarm.run(input_data)
    recs = result.get("recommendations", [])
    rec_ids = [r["item_id"] for r in recs]
    rec_scores = [r["score"] for r in recs]
    hidden_id = sample["hidden_item_id"]
    relevances = [1.0 if rid == hidden_id else 0.0 for rid in rec_ids]
    hit = compute_hit_rate(relevances, k=10, threshold=1.0)
    ndcg = compute_ndcg(relevances, rec_scores, k=10) if rec_scores else 0.0
    return i, sample["user_id"], hidden_id, rec_ids, hit, ndcg


predictions_b = [None] * len(test_b)
hit_rates = []
ndcgs = []
done = 0

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(run_task_b, (i, s)): i for i, s in enumerate(test_b)}
    for fut in as_completed(futures):
        i, user_id, hidden_id, rec_ids, hit, ndcg = fut.result()
        predictions_b[i] = {
            "user_id": user_id,
            "hidden_item_id": hidden_id,
            "recs": rec_ids,
            "hit": hit,
            "ndcg": ndcg,
        }
        hit_rates.append(hit)
        ndcgs.append(ndcg)
        done += 1
        if done % 100 == 0:
            print(f"  Task B progress: {done}/{len(test_b)}")

mean_hit = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
mean_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

task_b_summary = {
    "hit_rate_at_10": mean_hit,
    "ndcg_at_10": mean_ndcg,
    "num_samples": len(test_b),
    "predictions": predictions_b,
}

with open(RESULTS_DIR / "baseline_task_b.json", "w", encoding="utf-8") as f:
    json.dump(task_b_summary, f, ensure_ascii=False, indent=2)

print(f"Task B complete -> HitRate@10={mean_hit:.4f}, NDCG@10={mean_ndcg:.4f}")


# ========================= SUMMARY CSV =========================
print("\n=== Exporting summary CSV ===")
with open(RESULTS_DIR / "baseline_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Baseline_Value", "Target_Phase2"])
    writer.writerow(["Task_A_RMSE", f"{rmse:.4f}", "TBD"])
    writer.writerow(["Task_A_ROUGE-L", f"{task_a_summary['rouge_l_f1']:.4f}", "TBD"])
    writer.writerow(["Task_A_BERTScore", f"{task_a_summary['bertscore_f1']:.4f}", "TBD"])
    writer.writerow(["Task_B_HitRate@10", f"{mean_hit:.4f}", "TBD"])
    writer.writerow(["Task_B_NDCG@10", f"{mean_ndcg:.4f}", "TBD"])

print("All baseline evaluations saved to results/")

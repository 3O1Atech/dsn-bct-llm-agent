import json
import csv
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from task_a.pipeline import TaskASwarm
from task_b.pipeline import TaskBSwarm
from shared.persona_engine import PersonaEngine
from eval.metrics import compute_rmse, compute_rouge_l, compute_bertscore, compute_hit_rate, compute_ndcg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

WORKERS = int(os.getenv("EVAL_WORKERS", "1"))
MAX_SAMPLES = int(os.getenv("EVAL_MAX_SAMPLES", "0"))  # 0 = all
CHECKPOINT_EVERY = 100

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


def load_checkpoint(path: Path) -> dict:
    """Returns {idx: entry} for all previously completed samples."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["_idx"]: e for e in entries if e}


def save_checkpoint(path: Path, predictions: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e for e in predictions if e is not None], f, ensure_ascii=False)


# ========================= TASK A =========================
log.info("=== Baseline Evaluation: Task A ===")

with open("processed_data/test_set_task_a.json", "r", encoding="utf-8") as f:
    test_a = json.load(f)

if MAX_SAMPLES:
    test_a = test_a[:MAX_SAMPLES]

CHECKPOINT_A = RESULTS_DIR / "checkpoint_task_a.json"
ckpt_a = load_checkpoint(CHECKPOINT_A)

predictions_a = [None] * len(test_a)
y_true_a, y_pred_a, text_preds_a, text_refs_a = [], [], [], []

for idx, entry in ckpt_a.items():
    predictions_a[idx] = entry
    y_true_a.append(entry["true_rating"])
    y_pred_a.append(entry["predicted_rating"])
    if entry.get("true_text", "").strip():
        text_preds_a.append(entry["predicted_text"])
        text_refs_a.append(entry["true_text"])

done_a = len(ckpt_a)
remaining_a = [(i, s) for i, s in enumerate(test_a) if i not in ckpt_a]

log.info(f"Task A: {len(test_a)} total | {done_a} from checkpoint | {len(remaining_a)} remaining | workers={WORKERS}")


def run_task_a(args):
    i, sample = args
    swarm = TaskASwarm()
    user_id = sample["user_id"]
    persona = engine.build_persona(user_id, sample["train_history"])
    input_data = {
        "user_persona": {"user_id": user_id, **persona},
        "item_metadata": sample["item_metadata"],
        "context": {"time_of_day": "", "day_of_week": "", "mood_hint": ""},
    }
    result = swarm.run(input_data)
    return i, sample, result["rating"], result["review_text"]


with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(run_task_a, arg): arg[0] for arg in remaining_a}
    for fut in as_completed(futures):
        try:
            i, sample, pred_rating, pred_text = fut.result()
        except Exception as e:
            log.warning(f"Task A sample failed: {e}")
            continue

        entry = {
            "_idx": i,
            "user_id": sample["user_id"],
            "item_id": sample["item_metadata"]["item_id"],
            "predicted_rating": pred_rating,
            "true_rating": sample["true_rating"],
            "predicted_text": pred_text,
            "true_text": sample["true_text"],
        }
        predictions_a[i] = entry
        y_true_a.append(sample["true_rating"])
        y_pred_a.append(pred_rating)
        if sample["true_text"].strip():
            text_preds_a.append(pred_text)
            text_refs_a.append(sample["true_text"])

        done_a += 1
        if done_a % CHECKPOINT_EVERY == 0:
            save_checkpoint(CHECKPOINT_A, predictions_a)
            log.info(f"  Task A progress: {done_a}/{len(test_a)} (checkpoint saved)")

# Final checkpoint save
save_checkpoint(CHECKPOINT_A, predictions_a)

rmse = compute_rmse(y_true_a, y_pred_a)
rouge_res = compute_rouge_l(text_preds_a, text_refs_a) if text_preds_a else {"rouge_l_f1": 0.0}
bert_res = compute_bertscore(text_preds_a, text_refs_a) if text_preds_a else {"bertscore_f1": 0.0}

task_a_summary = {
    "rmse": rmse,
    "rouge_l_f1": rouge_res.get("rouge_l_f1", 0.0),
    "bertscore_f1": bert_res.get("bertscore_f1", 0.0),
    "num_samples": len(test_a),
    "num_evaluated": done_a,
    "predictions": [p for p in predictions_a if p is not None],
}

with open(RESULTS_DIR / "baseline_task_a.json", "w", encoding="utf-8") as f:
    json.dump(task_a_summary, f, ensure_ascii=False, indent=2)

log.info(f"Task A complete -> RMSE={rmse:.4f}, ROUGE-L={task_a_summary['rouge_l_f1']:.4f}, BERTScore={task_a_summary['bertscore_f1']:.4f}")


# ========================= TASK B =========================
log.info("=== Baseline Evaluation: Task B ===")

with open("processed_data/test_set_task_b.json", "r", encoding="utf-8") as f:
    test_b = json.load(f)

if MAX_SAMPLES:
    test_b = test_b[:MAX_SAMPLES]

CHECKPOINT_B = RESULTS_DIR / "checkpoint_task_b.json"
ckpt_b = load_checkpoint(CHECKPOINT_B)

predictions_b = [None] * len(test_b)
hit_rates, ndcgs = [], []

for idx, entry in ckpt_b.items():
    predictions_b[idx] = entry
    hit_rates.append(entry["hit"])
    ndcgs.append(entry["ndcg"])

done_b = len(ckpt_b)
remaining_b = [(i, s) for i, s in enumerate(test_b) if i not in ckpt_b]

log.info(f"Task B: {len(test_b)} total | {done_b} from checkpoint | {len(remaining_b)} remaining | workers={WORKERS}")


def run_task_b(args):
    i, sample = args
    swarm = TaskBSwarm()
    user_id = sample["user_id"]
    persona = engine.build_persona(user_id, sample["train_history"])
    taste_vector = build_taste_vector(sample["train_history"])
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
    return i, user_id, hidden_id, rec_ids, hit, ndcg


with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(run_task_b, arg): arg[0] for arg in remaining_b}
    for fut in as_completed(futures):
        try:
            i, user_id, hidden_id, rec_ids, hit, ndcg = fut.result()
        except Exception as e:
            log.warning(f"Task B sample failed: {e}")
            continue

        entry = {
            "_idx": i,
            "user_id": user_id,
            "hidden_item_id": hidden_id,
            "recs": rec_ids,
            "hit": hit,
            "ndcg": ndcg,
        }
        predictions_b[i] = entry
        hit_rates.append(hit)
        ndcgs.append(ndcg)
        done_b += 1

        if done_b % CHECKPOINT_EVERY == 0:
            save_checkpoint(CHECKPOINT_B, predictions_b)
            log.info(f"  Task B progress: {done_b}/{len(test_b)} (checkpoint saved)")

save_checkpoint(CHECKPOINT_B, predictions_b)

mean_hit = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
mean_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

task_b_summary = {
    "hit_rate_at_10": mean_hit,
    "ndcg_at_10": mean_ndcg,
    "num_samples": len(test_b),
    "num_evaluated": done_b,
    "predictions": [p for p in predictions_b if p is not None],
}

with open(RESULTS_DIR / "baseline_task_b.json", "w", encoding="utf-8") as f:
    json.dump(task_b_summary, f, ensure_ascii=False, indent=2)

log.info(f"Task B complete -> HitRate@10={mean_hit:.4f}, NDCG@10={mean_ndcg:.4f}")


# ========================= SUMMARY CSV =========================
with open(RESULTS_DIR / "baseline_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Baseline_Value", "Samples_Evaluated"])
    writer.writerow(["Task_A_RMSE", f"{rmse:.4f}", done_a])
    writer.writerow(["Task_A_ROUGE-L", f"{task_a_summary['rouge_l_f1']:.4f}", done_a])
    writer.writerow(["Task_A_BERTScore", f"{task_a_summary['bertscore_f1']:.4f}", done_a])
    writer.writerow(["Task_B_HitRate@10", f"{mean_hit:.4f}", done_b])
    writer.writerow(["Task_B_NDCG@10", f"{mean_ndcg:.4f}", done_b])

log.info("All baseline evaluations saved to results/")

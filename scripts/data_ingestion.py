import json
import csv
import random
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("raw_data")
PROC_DIR = Path("processed_data")
PROC_DIR.mkdir(exist_ok=True)

random.seed(42)

user_histories = {}
test_set_task_a = []
test_set_task_b = []
item_metadata = {}


def split_reviews(reviews, train_ratio=0.8):
    reviews = list(reviews)
    random.shuffle(reviews)
    n_train = int(len(reviews) * train_ratio)
    return reviews[:n_train], reviews[n_train:]


def print_domain_stats(name, users_dict, test_a, test_b, items_dict):
    n_users = len(users_dict)
    n_train = sum(len(v) for v in users_dict.values())
    n_test_a = sum(1 for s in test_a if s["user_id"].startswith(name + "_"))
    n_test_b = sum(1 for s in test_b if s["user_id"].startswith(name + "_"))
    n_items = sum(1 for m in items_dict.values() if m.get("domain") == name)
    print(f"  {name}: users={n_users}, train_reviews={n_train}, task_a={n_test_a}, task_b={n_test_b}, items={n_items}")


# ========================= AMAZON =========================
print("\n=== Processing Amazon ===")
amazon_path = RAW_DIR / "amazon" / "reviews_Movies_and_TV_5.json"
if amazon_path.exists():
    amazon_users = defaultdict(list)
    amazon_items = {}
    line_count = 0

    bad_lines = 0
    with open(amazon_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if line_count % 100000 == 0:
                print(f"  Amazon lines read: {line_count}")
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            reviewer_id = rec.get("reviewerID")
            asin = rec.get("asin")
            if not reviewer_id or not asin:
                continue

            review = {
                "review_id": f"amazon_{reviewer_id}_{asin}",
                "item_id": f"amazon_{asin}",
                "rating": int(float(rec.get("overall", 3))),
                "text": rec.get("reviewText", "") or "",
                "date": rec.get("reviewTime", ""),
                "unix_timestamp": rec.get("unixReviewTime", 0),
                "domain": "amazon",
                "category": "movies_and_tv",
            }
            amazon_users[reviewer_id].append(review)

            if review["item_id"] not in amazon_items:
                amazon_items[review["item_id"]] = {
                    "item_id": review["item_id"],
                    "name": asin,
                    "category": "movies_and_tv",
                    "description": rec.get("summary", "") or "",
                    "domain": "amazon",
                    "attributes": {},
                }

    if bad_lines:
        print(f"  [WARN] Skipped {bad_lines} malformed JSON lines")
    print(f"Amazon raw users: {len(amazon_users)}")
    amazon_users = {uid: revs for uid, revs in amazon_users.items() if 5 <= len(revs) <= 50}
    print(f"Amazon after 5-50 filter: {len(amazon_users)}")
    amazon_user_ids = list(amazon_users.keys())
    if len(amazon_user_ids) > 5000:
        amazon_user_ids = random.sample(amazon_user_ids, 5000)
    amazon_users = {uid: amazon_users[uid] for uid in amazon_user_ids}
    print(f"Amazon sampled users: {len(amazon_users)}")

    for uid, revs in amazon_users.items():
        train, test = split_reviews(revs)
        prefixed_uid = f"amazon_{uid}"
        user_histories[prefixed_uid] = train
        for t in test:
            test_set_task_a.append({
                "user_id": prefixed_uid,
                "train_history": train,
                "item_metadata": amazon_items[t["item_id"]],
                "true_rating": t["rating"],
                "true_text": t["text"],
            })
        if test:
            test_sorted = sorted(test, key=lambda x: x.get("unix_timestamp", 0))
            hidden = test_sorted[-1]
            test_set_task_b.append({
                "user_id": prefixed_uid,
                "train_history": train,
                "hidden_item_id": hidden["item_id"],
                "hidden_item_metadata": amazon_items[hidden["item_id"]],
                "true_rating": hidden["rating"],
            })

    item_metadata.update(amazon_items)
else:
    print("  [SKIP] Amazon raw data not found. Skipping Amazon domain.")


# ========================= GOODREADS =========================
print("\n=== Processing Goodreads ===")
gr_ratings_path = RAW_DIR / "goodreads" / "ratings.csv"
gr_books_path = RAW_DIR / "goodreads" / "books.csv"
if not gr_ratings_path.exists() or not gr_books_path.exists():
    raise FileNotFoundError(
        "Goodreads data not found. Run scripts/auto_download.py first."
    )

books = {}
with open(gr_books_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        book_id = row.get("book_id")
        if not book_id:
            continue
        prefixed_id = f"goodreads_{book_id}"
        books[prefixed_id] = {
            "item_id": prefixed_id,
            "name": row.get("title", ""),
            "category": "books",
            "description": f"{row.get('authors', '')} ({row.get('original_publication_year', '')})",
            "domain": "goodreads",
            "attributes": {"language_code": row.get("language_code", "")},
        }

gr_users = defaultdict(list)
with open(gr_ratings_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i % 500000 == 0:
            print(f"  Goodreads ratings read: {i}")
        user_id = row.get("user_id")
        book_id = row.get("book_id")
        rating = row.get("rating")
        if not user_id or not book_id or rating is None:
            continue
        prefixed_book_id = f"goodreads_{book_id}"
        review = {
            "review_id": f"goodreads_{user_id}_{book_id}",
            "item_id": prefixed_book_id,
            "rating": int(float(rating)),
            "text": "",
            "date": "",
            "unix_timestamp": 0,
            "domain": "goodreads",
            "category": "books",
        }
        gr_users[user_id].append(review)

print(f"Goodreads raw users: {len(gr_users)}")
gr_users = {uid: revs for uid, revs in gr_users.items() if 5 <= len(revs) <= 50}
print(f"Goodreads after 5-50 filter: {len(gr_users)}")
gr_user_ids = list(gr_users.keys())
if len(gr_user_ids) > 5000:
    gr_user_ids = random.sample(gr_user_ids, 5000)
gr_users = {uid: gr_users[uid] for uid in gr_user_ids}
print(f"Goodreads sampled users: {len(gr_users)}")

for uid, revs in gr_users.items():
    train, test = split_reviews(revs)
    prefixed_uid = f"goodreads_{uid}"
    user_histories[prefixed_uid] = train
    for t in test:
        test_set_task_a.append({
            "user_id": prefixed_uid,
            "train_history": train,
            "item_metadata": books.get(t["item_id"], {
                "item_id": t["item_id"],
                "name": t["item_id"],
                "category": "books",
                "description": "",
                "domain": "goodreads",
                "attributes": {},
            }),
            "true_rating": t["rating"],
            "true_text": t["text"],
        })
    if test:
        hidden = test[0]
        test_set_task_b.append({
            "user_id": prefixed_uid,
            "train_history": train,
            "hidden_item_id": hidden["item_id"],
            "hidden_item_metadata": books.get(hidden["item_id"], {
                "item_id": hidden["item_id"],
                "name": hidden["item_id"],
                "category": "books",
                "description": "",
                "domain": "goodreads",
                "attributes": {},
            }),
            "true_rating": hidden["rating"],
        })

item_metadata.update(books)


# ========================= NIGERIAN =========================
print("\n=== Processing Nigerian Yelp ===")
nig_hist_path = Path("data/nigerian_user_histories.json")
nig_items_path = Path("data/nigerian_items.json")
if not nig_hist_path.exists() or not nig_items_path.exists():
    raise FileNotFoundError(
        "Nigerian data files missing. Expected data/nigerian_user_histories.json and data/nigerian_items.json"
    )

with open(nig_items_path, "r", encoding="utf-8") as f:
    nig_items_raw = json.load(f)
nig_items = {}
for item in nig_items_raw.get("items", []):
    orig_id = item["item_id"]
    prefixed_id = f"yelp_nigerian_{orig_id}"
    nig_items[prefixed_id] = {
        "item_id": prefixed_id,
        "name": item.get("name", ""),
        "category": item.get("category", "general"),
        "description": item.get("description", ""),
        "domain": "yelp_nigerian",
        "attributes": item.get("attributes", {}),
    }

with open(nig_hist_path, "r", encoding="utf-8") as f:
    nig_data = json.load(f)
nig_users = {}
for user in nig_data.get("users", []):
    uid = user["user_id"]
    reviews = []
    for r in user.get("reviews", []):
        orig_item_id = r.get("item_id", "")
        prefixed_item_id = f"yelp_nigerian_{orig_item_id}"
        category = nig_items.get(prefixed_item_id, {}).get("category", "general")
        reviews.append({
            "review_id": f"yelp_nigerian_{uid}_{orig_item_id}",
            "item_id": prefixed_item_id,
            "rating": int(r.get("rating", 3)),
            "text": r.get("text", ""),
            "date": "",
            "unix_timestamp": 0,
            "domain": "yelp_nigerian",
            "category": category,
        })
    nig_users[uid] = reviews

print(f"Nigerian users: {len(nig_users)}")
for uid, revs in nig_users.items():
    train, test = split_reviews(revs)
    prefixed_uid = f"yelp_nigerian_{uid}"
    user_histories[prefixed_uid] = train
    for t in test:
        test_set_task_a.append({
            "user_id": prefixed_uid,
            "train_history": train,
            "item_metadata": nig_items.get(t["item_id"], {
                "item_id": t["item_id"],
                "name": t["item_id"],
                "category": "general",
                "description": "",
                "domain": "yelp_nigerian",
                "attributes": {},
            }),
            "true_rating": t["rating"],
            "true_text": t["text"],
        })
    if test:
        hidden = test[0]
        test_set_task_b.append({
            "user_id": prefixed_uid,
            "train_history": train,
            "hidden_item_id": hidden["item_id"],
            "hidden_item_metadata": nig_items.get(hidden["item_id"], {
                "item_id": hidden["item_id"],
                "name": hidden["item_id"],
                "category": "general",
                "description": "",
                "domain": "yelp_nigerian",
                "attributes": {},
            }),
            "true_rating": hidden["rating"],
        })

item_metadata.update(nig_items)


# ========================= SAVE =========================
print("\n=== Saving processed data ===")
with open(PROC_DIR / "user_histories.json", "w", encoding="utf-8") as f:
    json.dump(user_histories, f, ensure_ascii=False, indent=2)

with open(PROC_DIR / "test_set_task_a.json", "w", encoding="utf-8") as f:
    json.dump(test_set_task_a, f, ensure_ascii=False, indent=2)

with open(PROC_DIR / "test_set_task_b.json", "w", encoding="utf-8") as f:
    json.dump(test_set_task_b, f, ensure_ascii=False, indent=2)

with open(PROC_DIR / "item_metadata.json", "w", encoding="utf-8") as f:
    json.dump(item_metadata, f, ensure_ascii=False, indent=2)


# ========================= STATS =========================
print("\n=== Dataset Statistics ===")
if any(k.startswith("amazon_") for k in user_histories):
    print_domain_stats("amazon", {k:v for k,v in user_histories.items() if k.startswith("amazon_")}, test_set_task_a, test_set_task_b, item_metadata)
print_domain_stats("goodreads", {k:v for k,v in user_histories.items() if k.startswith("goodreads_")}, test_set_task_a, test_set_task_b, item_metadata)
print_domain_stats("yelp_nigerian", {k:v for k,v in user_histories.items() if k.startswith("yelp_nigerian_")}, test_set_task_a, test_set_task_b, item_metadata)

total_users = len(user_histories)
total_items = len(item_metadata)
total_task_a = len(test_set_task_a)
total_task_b = len(test_set_task_b)
print(f"\nTotal: users={total_users}, items={total_items}, task_a_samples={total_task_a}, task_b_samples={total_task_b}")
print("Done.")

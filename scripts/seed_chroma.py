import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 1000
CHROMA_DIR = "./chroma_data"


def main():
    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    print("Loading embedding model: all-MiniLM-L6-v2 (cached locally)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    proc_dir = Path("processed_data")
    with open(proc_dir / "user_histories.json", "r", encoding="utf-8") as f:
        user_histories = json.load(f)
    with open(proc_dir / "item_metadata.json", "r", encoding="utf-8") as f:
        item_metadata = json.load(f)
    with open("data/nigerian_context.json", "r", encoding="utf-8") as f:
        nigerian_context = json.load(f)

    # ------------------- user_reviews -------------------
    print("Building user_reviews collection...")
    coll_reviews = client.get_or_create_collection("user_reviews")

    docs = []
    metas = []
    ids = []
    for user_id, reviews in user_histories.items():
        for rev in reviews:
            docs.append(rev.get("text", "") or "")
            metas.append({
                "user_id": user_id,
                "item_id": rev.get("item_id", ""),
                "rating": rev.get("rating", 0),
                "domain": rev.get("domain", ""),
            })
            ids.append(rev.get("review_id", f"{user_id}_{rev.get('item_id', '')}"))

    total = len(docs)
    print(f"Total user reviews to embed: {total}")
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        embeddings = embedder.encode(docs[i:end], show_progress_bar=False).tolist()
        coll_reviews.upsert(
            documents=docs[i:end],
            metadatas=metas[i:end],
            ids=ids[i:end],
            embeddings=embeddings,
        )
        print(f"  user_reviews: {end}/{total}")
    print(f"user_reviews final count: {coll_reviews.count()}")

    # ------------------- item_metadata -------------------
    print("Building item_metadata collection...")
    coll_items = client.get_or_create_collection("item_metadata")

    docs = []
    metas = []
    ids = []
    for item_id, meta in item_metadata.items():
        name = meta.get("name", "")
        desc = meta.get("description", "")
        docs.append(f"{name}. {desc}")
        metas.append({
            "item_id": item_id,
            "name": meta.get("name", ""),
            "domain": meta.get("domain", ""),
            "category": meta.get("category", ""),
        })
        ids.append(item_id)

    total = len(docs)
    print(f"Total items to embed: {total}")
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        embeddings = embedder.encode(docs[i:end], show_progress_bar=False).tolist()
        coll_items.upsert(
            documents=docs[i:end],
            metadatas=metas[i:end],
            ids=ids[i:end],
            embeddings=embeddings,
        )
        print(f"  item_metadata: {end}/{total}")
    print(f"item_metadata final count: {coll_items.count()}")

    # ------------------- nigerian_refs -------------------
    print("Building nigerian_refs collection...")
    coll_nig = client.get_or_create_collection("nigerian_refs")

    nigerian_docs = []
    nigerian_docs.extend(nigerian_context.get("seeding_documents", []))

    for note_type, phrases in nigerian_context.get("cultural_notes", {}).items():
        nigerian_docs.append(f"{note_type}: {', '.join(phrases)}")

    for ref_type, refs in nigerian_context.get("local_references", {}).items():
        nigerian_docs.append(f"Nigerian {ref_type} references: {', '.join(refs)}")

    for sentiment, phrases in nigerian_context.get("pidgin_phrases", {}).items():
        nigerian_docs.append(f"Pidgin {sentiment} phrases: {', '.join(phrases)}")

    metas = [{"type": "cultural_reference"} for _ in nigerian_docs]
    ids = [f"nigerian_ref_{i}" for i in range(len(nigerian_docs))]

    if nigerian_docs:
        embeddings = embedder.encode(nigerian_docs, show_progress_bar=False).tolist()
        coll_nig.upsert(
            documents=nigerian_docs,
            metadatas=metas,
            ids=ids,
            embeddings=embeddings,
        )
    print(f"nigerian_refs final count: {coll_nig.count()}")

    print("All collections seeded successfully.")


if __name__ == "__main__":
    main()

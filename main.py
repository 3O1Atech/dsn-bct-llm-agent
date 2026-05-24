import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from task_a.api import router as task_a_router
from task_b.api import router as task_b_router
from shared.llm_backend import get_llm
from shared.vector_store import get_chroma
from shared.nigerian_kb import LOCAL_REFERENCES


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing ChromaDB collections...")
    chroma = get_chroma()
    # Ensure collections exist
    chroma.client.get_or_create_collection("user_reviews")
    chroma.client.get_or_create_collection("item_metadata")
    chroma.client.get_or_create_collection("nigerian_refs")
    print("Collections ready.")

    print(f"Nigerian KB loaded: {sum(len(v) for v in LOCAL_REFERENCES.values())} local references.")

    print("Warming up LLM...")
    llm = get_llm()
    if llm.is_ready():
        _ = llm.generate("Hello", max_tokens=10)
        print("LLM ready.")
    else:
        print("LLM not ready; check model or OpenAI config.")

    # Run seed if collections empty
    try:
        coll = chroma.client.get_collection("user_reviews")
        if coll.count() == 0:
            print("Seeding database...")
            import subprocess
            subprocess.run(["python", "scripts/seed_db.py"], check=False)
    except Exception as e:
        print(f"Seed check skipped: {e}")

    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="Nigerian Multi-Agent Swarm",
    description="Cross-domain recommendation and review generation with Nigerian localization.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(task_a_router, prefix="/api/v1")
app.include_router(task_b_router, prefix="/api/v1")


@app.get("/health")
async def health():
    llm = get_llm()
    return {
        "task_a": True,
        "task_b": True,
        "llm_ready": llm.is_ready(),
    }


@app.get("/")
async def root():
    return {"message": "Nigerian Agent Swarm is running. Visit /docs for API explorer."}

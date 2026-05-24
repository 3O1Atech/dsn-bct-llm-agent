import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
USE_KIMI = os.getenv("USE_KIMI", "false").lower() == "true"
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_MODEL = os.getenv("KIMI_MODEL", "moonshot-v1-8k")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TASK_A_PORT = int(os.getenv("TASK_A_PORT", "8000"))
TASK_B_PORT = int(os.getenv("TASK_B_PORT", "8000"))
NIGERIAN_INJECTION_RATE = float(os.getenv("NIGERIAN_INJECTION_RATE", "0.4"))
DEVICE = os.getenv("DEVICE", "auto")

# Thresholds
RATING_ALIGNMENT_THRESHOLD = 0.7
GRADER_MIN_SCORE = 5.0
SEMANTIC_CANDIDATES_TOPK = 20
RANKER_TOPK = 10

# Composite scoring weights
WEIGHT_SEMANTIC = 0.4
WEIGHT_GRADER = 0.3
WEIGHT_NIGERIAN = 0.2
WEIGHT_POPULARITY = 0.1

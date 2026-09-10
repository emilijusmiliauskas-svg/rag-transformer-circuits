from pathlib import Path


# --- LLM Model Configuration ---
# Generation runs on OpenRouter rather than Groq. Same model, different host:
# Groq's free tier caps at 200,000 tokens/day, which one four-stage ablation
# exhausts — especially once stage 2 selected 1024-token chunks, so every
# generation carries ~5k tokens of retrieved context. Keeping the model
# identical means results stay comparable across all four stages.
LLM_MODEL: str = "openai/gpt-oss-120b"
LLM_API_BASE: str = "https://openrouter.ai/api/v1"

# gpt-oss draws reasoning tokens from the same completion budget as the
# answer, so a tight cap can return an empty body rather than a short answer.
# 768 was enough on Groq; 2048 leaves headroom without changing what the
# model is asked to produce.
LLM_MAX_NEW_TOKENS: int = 2048
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
# The OpenAI-compatible API has no repetition_penalty; the equivalent knobs
# are frequency_penalty / presence_penalty. Left unset.

# --- System Prompt ---
LLM_SYSTEM_PROMPT: str = (
    "You are a research assistant answering questions about mechanistic "
    "interpretability papers. Answer using only the provided context. "
    "If the context does not contain the answer, say so plainly rather "
    "than drawing on outside knowledge or speculating. Quote the papers' "
    "own terminology where it is precise, and keep answers concise."
)

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# --- RAG/VectorStore Configuration ---
# These are the winning values selected via the evaluation pipeline
# (evaluate.py). SIMILARITY_TOP_K is the size of the initial retrieval
# pool BEFORE reranking.
SIMILARITY_TOP_K: int = 10
# 1024/200 replaced 512/50 after the corpus fix. On the original index —
# 97.7% base64 — larger chunks scored worse, because a bigger chunk swept in
# proportionally more binary. On clean prose the ranking reverses.
CHUNK_SIZE: int = 1024
CHUNK_OVERLAP: int = 200

# --- Reranker Configuration ---
# Cross-encoder used to re-score the top-k retrieved chunks; the top
# RERANKER_TOP_N are then passed to the LLM.
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_N: int = 5

# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 3900

# --- Persistent Storage Paths ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"

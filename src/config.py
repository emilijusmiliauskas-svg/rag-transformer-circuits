from pathlib import Path


# --- LLM Model Configuration ---
LLM_MODEL: str = "openai/gpt-oss-120b"
LLM_MAX_NEW_TOKENS: int = 768
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
# Groq's OpenAI-compatible API has no repetition_penalty; the equivalent
# knobs are frequency_penalty / presence_penalty. Left unset.

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
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50

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

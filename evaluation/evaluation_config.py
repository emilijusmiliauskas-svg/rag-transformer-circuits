from pathlib import Path

from ragas.metrics.base import Metric
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
)

# --- LLM Model Configuration ---
# The original judge, moonshotai/kimi-k2-instruct, was retired from Groq.
#
# The intent was a judge from a different family than the generator
# (openai/gpt-oss-120b) so the system does not grade its own work. On Groq's
# free tier that is not currently achievable: qwen3.8-27b is capped at 1000
# output tokens/minute, and AnswerCorrectness needs ~2048 to classify an
# answer without truncating. groq/compound-mini has the headroom but is an
# agentic system with tool use, which is wrong for a deterministic judge.
#
# gpt-oss-20b is therefore the judge: a different model and size from the
# generator, but the same family. That is a weaker independence guarantee
# than the original design and is recorded as a limitation in the README.
EVALUATION_LLM_MODEL: str = "openai/gpt-oss-20b"

# --- Embedding Model Configuration ---
EVALUATION_EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"

# --- Paths for Evaluation ---
EVALUATION_ROOT_PATH: Path = Path(__file__).parent
EVALUATION_RESULTS_PATH: Path = EVALUATION_ROOT_PATH / "evaluation_results/"
EXPERIMENTAL_VECTOR_STORES_PATH: Path = (
    EVALUATION_ROOT_PATH
    / "evaluation_vector_stores/"
)
EVALUATION_EMBEDDING_CACHE_PATH: Path = (
    EVALUATION_ROOT_PATH
    / "evaluation_embedding_models/"
)

# --- Ragas Evaluation Metrics ---
EVALUATION_METRICS: list[Metric] = [
    Faithfulness(),
    AnswerCorrectness(),
    ContextPrecision(),
    ContextRecall(),
]

# --- Sleep Timers for API Limits ---
SLEEP_PER_EVALUATION: int = 60
SLEEP_PER_QUESTION: int = 6

# --- Configuration for Chunking Strategy Evaluation ---
CHUNKING_STRATEGY_CONFIGS: list[dict[str, int]] = [
    {'size': 512, 'overlap': 50},
    {'size': 768, 'overlap': 115},
    {'size': 1024, 'overlap': 200},
]

# --- Cross-encoder Model for Reranking ---
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Configuration for Reranking Evaluation ---
RERANKER_CONFIGS: list[dict[str, int]] = [
    {'retriever_k': 10, 'reranker_n': 2},
    {'retriever_k': 10, 'reranker_n': 5},
    {'retriever_k': 20, 'reranker_n': 5},
]

# --- Fallback defaults ---
# These are only used if a prior evaluation stage's results CSV can't
# be found on disk. Normally the pipeline auto-picks winners from the
# latest chunking_evaluation_*.csv and reranker_evaluation_*.csv.
DEFAULT_CHUNKING_STRATEGY: dict[str, int] = {'size': 512, 'overlap': 50}
DEFAULT_RERANKER_STRATEGY: dict[str, int] = {
    'retriever_k': 10, 'reranker_n': 2
}

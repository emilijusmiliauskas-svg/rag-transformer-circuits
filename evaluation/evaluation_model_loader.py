import os
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms.base import LlamaIndexLLMWrapper

from evaluation.evaluation_config import (
    EVALUATION_LLM_MODEL,
    EVALUATION_EMBEDDING_MODEL_NAME,
    EVALUATION_EMBEDDING_CACHE_PATH,
)


# Load environment variables from the .env file
load_dotenv()


def initialise_evaluation_llm() -> Groq:
    """Initialises the Groq LLM with core parameters from config."""

    api_key: str | None = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Make sure it's set in your .env file."
        )

    # AnswerCorrectness asks the judge to classify every statement in an
    # answer against the ground truth, which runs long. Two things broke that
    # before: at 2048 tokens the JSON truncated mid-object, and gpt-oss also
    # spends part of the completion on a hidden reasoning trace drawn from the
    # same budget. Lowering reasoning_effort freed budget but made the judge
    # under-think and return an empty object, so the fix is simply a larger
    # budget at default effort. gpt-oss-20b has no output-per-minute cap on
    # the free tier (unlike qwen3.8-27b, which is limited to 1000).
    return Groq(
        api_key=api_key,
        model=EVALUATION_LLM_MODEL,
        temperature=0.0,
        max_tokens=8000,
    )


def load_ragas_models(
) -> tuple[LlamaIndexLLMWrapper, HuggingFaceEmbeddings]:
    """
    Loads the LLM and embedding models
    required for Ragas evaluation.
    """
    print("--- 🧠 Loading Ragas LLM and Embeddings ---")

    llm_for_evaluation: Groq = initialise_evaluation_llm()

    # Wrap the LlamaIndex LLM for compatibility with Ragas
    ragas_llm = LlamaIndexLLMWrapper(llm=llm_for_evaluation)

    # Initialise the embedding model Ragas will use for its metrics
    ragas_embeddings = HuggingFaceEmbeddings(
        model=EVALUATION_EMBEDDING_MODEL_NAME,
        cache_folder=EVALUATION_EMBEDDING_CACHE_PATH.as_posix()
    )

    return ragas_llm, ragas_embeddings

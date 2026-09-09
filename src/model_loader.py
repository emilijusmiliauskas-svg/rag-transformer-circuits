import os
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import (
    LLM_MODEL,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_CACHE_PATH,
)


load_dotenv()


def initialise_llm() -> Groq:
    """Initialises the Groq LLM with core parameters from config."""

    api_key: str | None = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Make sure it's set in your .env file."
        )

    # These were declared in config.py but never passed, so generation ran at
    # the provider default rather than near-deterministic — the likely cause of
    # the run-to-run variance in the first evaluation round.
    return Groq(
        api_key=api_key,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_NEW_TOKENS,
        additional_kwargs={"top_p": LLM_TOP_P},
    )


def get_embedding_model() -> HuggingFaceEmbedding:
    """Initialises and returns the HuggingFace embedding model."""

    EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix()
    )

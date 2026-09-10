import os
from dotenv import load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

from src.config import (
    LLM_API_BASE,
    LLM_MODEL,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_CACHE_PATH,
)


load_dotenv()


def initialise_llm() -> OpenAILike:
    """
    Initialises the generation model.

    OpenRouter exposes an OpenAI-compatible endpoint, so OpenAILike is used
    rather than a provider-specific client — which also means swapping hosts
    is a config change rather than a code change.
    """

    api_key: str | None = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. Make sure it's set in your .env file."
        )

    # temperature, max_tokens and top_p were declared in config.py but never
    # passed to the client, so generation ran at the provider default rather
    # than near-deterministic — the likely cause of the run-to-run variance in
    # the first evaluation round.
    return OpenAILike(
        api_key=api_key,
        api_base=LLM_API_BASE,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_NEW_TOKENS,
        is_chat_model=True,
        timeout=600.0,
        additional_kwargs={"top_p": LLM_TOP_P},
    )


def get_embedding_model() -> HuggingFaceEmbedding:
    """Initialises and returns the HuggingFace embedding model."""

    EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix()
    )

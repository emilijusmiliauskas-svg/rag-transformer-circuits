import os
from dotenv import load_dotenv

from llama_index.llms.openai_like import OpenAILike
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms.base import LlamaIndexLLMWrapper

from evaluation.evaluation_config import (
    EVALUATION_LLM_API_BASE,
    EVALUATION_LLM_MODEL,
    EVALUATION_EMBEDDING_MODEL_NAME,
    EVALUATION_EMBEDDING_CACHE_PATH,
)


# Load environment variables from the .env file
load_dotenv()


def initialise_evaluation_llm() -> OpenAILike:
    """
    Initialises the judge model.

    DeepSeek exposes an OpenAI-compatible endpoint, so OpenAILike is used
    rather than a provider-specific class — OpenAI's own wrapper rejects
    model names it does not recognise.
    """

    api_key: str | None = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not found. Set it in .env, or point "
            "load_dotenv() at an env file that defines it."
        )

    # The budget has to be generous, and for a non-obvious reason: DeepSeek V4
    # draws hidden reasoning tokens from the same allowance as the answer. Set
    # it too low and the reasoning consumes the whole budget, so the API
    # returns success with an *empty* body, which then fails to parse as JSON.
    # Measured on the Faithfulness NLI prompt, 5 calls each:
    #
    #   max_tokens=2048  -> 3/5 empty
    #   max_tokens=4096  -> 0/5 empty
    #   max_tokens=8000  -> 0/5 empty
    #   max_tokens=16000 -> 0/5 empty
    #
    # Actual content is only ~900 characters; the headroom is for the
    # reasoning, not the answer. AnswerCorrectness sends far more input than
    # the NLI prompt, so 16000 buys margin over the measured floor.
    return OpenAILike(
        api_key=api_key,
        api_base=EVALUATION_LLM_API_BASE,
        model=EVALUATION_LLM_MODEL,
        temperature=0.0,
        max_tokens=16000,
        is_chat_model=True,
        timeout=600.0,
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

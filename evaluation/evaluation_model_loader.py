import os
from dotenv import load_dotenv

from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms.base import LlamaIndexLLMWrapper

from evaluation.retrying_llm import RetryOnEmptyLLM

from evaluation.evaluation_config import (
    EVALUATION_LLM_API_BASE,
    EVALUATION_LLM_MODEL,
    EVALUATION_EMBEDDING_MODEL_NAME,
    EVALUATION_EMBEDDING_CACHE_PATH,
)


# Load environment variables from the .env file
load_dotenv()


def initialise_evaluation_llm() -> RetryOnEmptyLLM:
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

    # Thinking is disabled deliberately, and it is the fix for a failure that
    # took a while to pin down.
    #
    # DeepSeek V4 draws reasoning tokens from the same max_tokens allowance as
    # the answer. When reasoning consumes the budget the API returns a
    # well-formed HTTP 200 whose content is an empty string — no error, no
    # truncation flag. RAGAS cannot parse that, its repair prompt has nothing
    # to repair, and the whole evaluation aborts, discarding every result
    # gathered so far.
    #
    # Measured on the Faithfulness NLI prompt, 5 calls each:
    #
    #   thinking ON,  max_tokens=2048  -> 5/5 empty
    #   thinking OFF, max_tokens=2048  -> 0/5 empty
    #   thinking OFF, max_tokens=4096  -> 0/5 empty
    #
    # It is a threshold effect, not a flaky one, which is why it looked random
    # across prompts of differing length. RAGAS's prompts are classification
    # tasks and do not need a reasoning pass, so turning it off costs nothing
    # and makes the judge deterministic, cheaper and faster.
    #
    # RetryOnEmptyLLM stays as a safety net for any empty that slips through.
    return RetryOnEmptyLLM(
        api_key=api_key,
        api_base=EVALUATION_LLM_API_BASE,
        model=EVALUATION_LLM_MODEL,
        temperature=0.0,
        max_tokens=8000,
        is_chat_model=True,
        timeout=600.0,
        additional_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
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

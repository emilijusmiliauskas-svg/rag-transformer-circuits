from datasets import Dataset
from datetime import datetime
from pathlib import Path
import time
from typing import Any

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pandas as pd
from ragas.dataset_schema import EvaluationResult
from ragas.executor import Executor
from ragas.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms.base import LlamaIndexLLMWrapper

from evaluation.evaluation_config import (
    EVALUATION_RESULTS_PATH,
    EXPERIMENTAL_VECTOR_STORES_PATH,
    SLEEP_PER_QUESTION,
    SLEEP_PER_EVALUATION,
    EVALUATION_METRICS,
)
from evaluation.evaluation_questions import EVALUATION_DATA
from src.corpus import load_documents
from src.config import DATA_PATH


def get_evaluation_data() -> tuple[list[str], list[str]]:
    """
    Extracts questions and ground truths from the EVALUATION_DATA constant.
    """

    return [item["question"] for item in EVALUATION_DATA], [
        item["ground_truth"] for item in EVALUATION_DATA
    ]



# The judge produces long structured output and the free tier throttles it,
# so RAGAS's default 180s per-metric timeout and 10 retries are both too
# tight; a single AnswerCorrectness call can spend minutes in backoff.
EVALUATION_RUN_CONFIG: RunConfig = RunConfig(
    timeout=900,
    max_retries=15,
    max_wait=90,
    max_workers=2,
)

def get_best_config_from_results(
    filename_prefix: str,
    param_cols: list[str],
    score_cols: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    Finds the most recent detailed CSV for `filename_prefix` in the
    evaluation_results directory and returns the winning parameter
    combination — picked by the mean of `score_cols` (defaults to
    context_precision + context_recall).

    Returns None if no results file can be found — callers should
    handle this and fall back to a default.
    """

    score_cols = score_cols or [
        "faithfulness",
        "answer_correctness",
        "context_precision",
        "context_recall",
    ]

    results_dir: Path = EVALUATION_RESULTS_PATH
    if not results_dir.exists():
        return None

    candidates: list[Path] = sorted(
        results_dir.glob(f"{filename_prefix}_detailed_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    latest: Path = candidates[0]
    df: pd.DataFrame = pd.read_csv(latest)

    missing_params: list[str] = [c for c in param_cols if c not in df.columns]
    missing_scores: list[str] = [c for c in score_cols if c not in df.columns]
    if missing_params or missing_scores:
        return None

    grouped: pd.DataFrame = df.groupby(param_cols)[score_cols].mean()
    grouped["__combined_score__"] = grouped.mean(axis=1)
    winner_key = grouped["__combined_score__"].idxmax()

    if not isinstance(winner_key, tuple):
        winner_key = (winner_key,)

    winner: dict[str, Any] = dict(zip(param_cols, winner_key))
    winner["__source_file__"] = latest.name
    winner["__combined_score__"] = float(
        grouped.loc[winner_key[0] if len(winner_key) == 1 else winner_key,
                    "__combined_score__"]
    )
    return winner


def get_or_build_index(
    chunk_size: int, chunk_overlap: int, embed_model: HuggingFaceEmbedding
) -> VectorStoreIndex:
    """
    Checks for a persisted vector store for this experiment.
    If it exists, it loads it. If not, it builds it, persists it,
    and returns it.
    """

    vector_store_id: str = f"vs_chunk_{chunk_size}_overlap_{chunk_overlap}"
    specific_vector_store_path: Path = (
        EXPERIMENTAL_VECTOR_STORES_PATH
        / vector_store_id
    )

    if specific_vector_store_path.exists():
        print(f"--- Loading existing index from: {vector_store_id} ---")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir=str(specific_vector_store_path)
        )
        index: VectorStoreIndex = load_index_from_storage(
            storage_context, embed_model=embed_model
        )
    else:
        print(f"--- Creating new index for: {vector_store_id} ---")
        documents: list[Document] = load_documents(DATA_PATH)

        text_splitter: SentenceSplitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        index: VectorStoreIndex = VectorStoreIndex.from_documents(
            documents, transformations=[text_splitter], embed_model=embed_model
        )

        index.storage_context.persist(
            persist_dir=str(specific_vector_store_path)
        )
        print(f"--- Saved new index to: {vector_store_id} ---")
    return index


def generate_qa_dataset(
    query_engine: BaseQueryEngine,
    questions: list[str],
    ground_truths: list[str]
) -> Dataset:
    """
    Generates answers and contexts for a given query engine
    and returns a HuggingFace Dataset.
    """

    responses: list[str] = []
    contexts: list[list[str]] = []
    for question_index, question in enumerate(questions):
        print(
            "Fetching context and synthesising response for question "
            f"{question_index + 1}/{len(questions)}: '{question[:30]}...'"
        )
        response_object = query_engine.query(question)
        responses.append(str(response_object))
        contexts.append(
            [node.get_content() for node in response_object.source_nodes]
        )

        # If you are hitting API rate limits
        # You can slow down the rate with time.sleep
        #
        # if question_index + 1 < len(questions):
        #     print(
        #         f"Taking a {SLEEP_PER_QUESTION} second breather "
        #         "to keep the API happy 🐢"
        #     )
        #     time.sleep(SLEEP_PER_QUESTION)
        # else:
        #     continue

    response_data: dict[str, list[Any]] = {
        "question": questions,
        "answer": responses,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    return Dataset.from_dict(response_data)


def evaluate_without_rate_limit(
    qa_dataset: Dataset,
    ragas_llm: LlamaIndexLLMWrapper,
    ragas_embeddings: HuggingFaceEmbeddings,
) -> pd.DataFrame:
    """
    Runs Ragas evaluation on the entire dataset at once.
    Ideal for local models or APIs without strict rate limits.
    """

    print("--- ⚡ Running evaluation without rate limiting... ---")

    result: EvaluationResult | Executor = evaluate(
        dataset=qa_dataset,
        metrics=EVALUATION_METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=True,
        run_config=EVALUATION_RUN_CONFIG,
    )

    results_df: pd.DataFrame = result.to_pandas()

    print("--- ✅ Evaluation complete! ---")

    return results_df


def evaluate_with_rate_limit(
    qa_dataset: Dataset,
    ragas_llm: LlamaIndexLLMWrapper,
    ragas_embeddings: HuggingFaceEmbeddings,
) -> pd.DataFrame:
    """
    Runs Ragas evaluation row-by-row to accommodate API rate limits,
    pausing between each evaluation.
    """

    print("--- 🐢 Running evaluation with rate limiting... ---")
    number_of_questions: int = len(qa_dataset)

    partial_results_list: list[pd.DataFrame] = []
    row: dict[str, Any]
    for i, row in enumerate(qa_dataset):
        print(
            f"Evaluating response for question {i + 1}/{number_of_questions}: "
            f"'{row['question'][:50]}...'"
        )

        single_row_dataset: Dataset = Dataset.from_dict(
            {key: [value] for key, value in row.items()}
        )

        result: EvaluationResult | Executor = evaluate(
            dataset=single_row_dataset,
            metrics=EVALUATION_METRICS,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=True,
            run_config=EVALUATION_RUN_CONFIG,
        )

        partial_results_list.append(result.to_pandas())

        if i + 1 < number_of_questions:
            print(
                f"Taking a {SLEEP_PER_EVALUATION} second breather "
                "to keep the API happy."
            )
            time.sleep(SLEEP_PER_EVALUATION)

    results_df: pd.DataFrame = pd.concat(
        partial_results_list,
        ignore_index=True
    )

    print("--- ✅ Evaluation complete! ---")

    return results_df


def save_results(results_df: pd.DataFrame, filename_prefix: str) -> None:
    """Saves the evaluation results and summary to CSV files."""

    results_dir: Path = EVALUATION_RESULTS_PATH
    results_dir.mkdir(exist_ok=True, parents=True)
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    detailed_path: Path = (
        results_dir
        / f"{filename_prefix}_detailed_{timestamp}.csv"
    )
    results_df.to_csv(detailed_path, index=False)
    print(f"--- 💾 Detailed results saved to {detailed_path} ---")

    summary_path: Path = (
        results_dir
        / f"{filename_prefix}_summary_{timestamp}.csv"
    )
    param_cols: list[str] = [
        col
        for col in [
            'chunk_size',
            'chunk_overlap',
            'retriever_k',
            'reranker_n',
            'use_hyde']
        if col in results_df.columns
    ]

    if param_cols:
        avg_scores: pd.DataFrame = results_df.groupby(param_cols).mean(
            numeric_only=True
        )
        avg_scores.to_csv(summary_path)
        print(f"--- 💾 Summary of average scores saved to {summary_path} ---")

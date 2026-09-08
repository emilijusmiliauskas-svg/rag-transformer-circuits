from datasets import Dataset

from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import (
    BaseQueryEngine,
    RetrieverQueryEngine,
    TransformQueryEngine,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import pandas as pd
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms.base import LlamaIndexLLMWrapper

from evaluation.evaluation_helper_functions import (
    generate_qa_dataset,
    get_best_config_from_results,
    get_evaluation_data,
    get_or_build_index,
    save_results,
    evaluate_without_rate_limit,
    evaluate_with_rate_limit,
)
from evaluation.evaluation_config import (
    CHUNKING_STRATEGY_CONFIGS,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_RERANKER_STRATEGY,
    RERANKER_CONFIGS,
    RERANKER_MODEL_NAME,
)
from evaluation.evaluation_model_loader import load_ragas_models
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SIMILARITY_TOP_K,
)
from src.model_loader import get_embedding_model, initialise_llm


def _resolve_best_chunking() -> dict[str, int]:
    """
    Reads the latest chunking_evaluation results and picks the winning
    (chunk_size, chunk_overlap). Falls back to DEFAULT_CHUNKING_STRATEGY
    if no results file is available.
    """
    winner = get_best_config_from_results(
        filename_prefix="chunking_evaluation",
        param_cols=["chunk_size", "chunk_overlap"],
    )
    if winner is None:
        print(
            "⚠️  No chunking_evaluation results found. "
            f"Falling back to defaults: {DEFAULT_CHUNKING_STRATEGY}"
        )
        return {
            "chunk_size": DEFAULT_CHUNKING_STRATEGY["size"],
            "chunk_overlap": DEFAULT_CHUNKING_STRATEGY["overlap"],
        }

    print(
        f"✅ Auto-selected best chunking from {winner['__source_file__']}: "
        f"chunk_size={winner['chunk_size']}, "
        f"chunk_overlap={winner['chunk_overlap']} "
        f"(score={winner['__combined_score__']:.3f})"
    )
    return {
        "chunk_size": int(winner["chunk_size"]),
        "chunk_overlap": int(winner["chunk_overlap"]),
    }


def _resolve_best_reranker() -> dict[str, int]:
    """
    Reads the latest reranker_evaluation results and picks the winning
    (retriever_k, reranker_n). Falls back to DEFAULT_RERANKER_STRATEGY
    if no results file is available.
    """
    winner = get_best_config_from_results(
        filename_prefix="reranker_evaluation",
        param_cols=["retriever_k", "reranker_n"],
    )
    if winner is None:
        print(
            "⚠️  No reranker_evaluation results found. "
            f"Falling back to defaults: {DEFAULT_RERANKER_STRATEGY}"
        )
        return dict(DEFAULT_RERANKER_STRATEGY)

    print(
        f"✅ Auto-selected best reranker from {winner['__source_file__']}: "
        f"retriever_k={winner['retriever_k']}, "
        f"reranker_n={winner['reranker_n']} "
        f"(score={winner['__combined_score__']:.3f})"
    )
    return {
        "retriever_k": int(winner["retriever_k"]),
        "reranker_n": int(winner["reranker_n"]),
    }


def evaluate_baseline() -> None:
    """
    Evaluates the RAG system using only the settings from config.py.
    """

    print("--- 🚀 Stage 1: Evaluating Baseline Configuration ---")

    llm_to_test: Groq = initialise_llm()

    embed_model_to_test: HuggingFaceEmbedding = get_embedding_model()

    questions: list[str]
    ground_truths: list[str]
    questions, ground_truths = get_evaluation_data()

    index: VectorStoreIndex = get_or_build_index(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embed_model=embed_model_to_test
    )

    query_engine: BaseQueryEngine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        llm=llm_to_test
    )

    qa_dataset: Dataset = generate_qa_dataset(
        query_engine,
        questions,
        ground_truths
    )

    print("--- Running Ragas evaluation for baseline... ---")

    ragas_llm: LlamaIndexLLMWrapper
    ragas_embeddings: HuggingFaceEmbeddings
    ragas_llm, ragas_embeddings = load_ragas_models()

    # --- If you don't have a Rate per Minute limit on your API ---
    # results_df: pd.DataFrame = evaluate_without_rate_limit(
    #     qa_dataset,
    #     ragas_llm,
    #     ragas_embeddings,
    # )

    # --- If you do have a Rate per Minute API limit ---
    results_df: pd.DataFrame = evaluate_with_rate_limit(
        qa_dataset,
        ragas_llm,
        ragas_embeddings,
    )

    # Add Chunk Size and Chunk Overlap to DataFrame to help tracking
    results_df['chunk_size'] = CHUNK_SIZE
    results_df['chunk_overlap'] = CHUNK_OVERLAP

    save_results(results_df, "baseline_evaluation")

    print("--- ✅ Baseline Evaluation Complete ---")


def evaluate_chunking_strategies() -> None:
    """ Evaluates different chunk sizes and overlaps. """
    print("\n--- 🚀 Stage 2: Evaluating Chunking Strategies ---")

    llm_to_test: Groq = initialise_llm()

    embed_model_to_test: HuggingFaceEmbedding = get_embedding_model()

    questions, ground_truths = get_evaluation_data()

    ragas_llm: LlamaIndexLLMWrapper
    ragas_embeddings: HuggingFaceEmbeddings
    ragas_llm, ragas_embeddings = load_ragas_models()

    all_results: list[pd.DataFrame] = []

    for config in CHUNKING_STRATEGY_CONFIGS:

        chunk_size, chunk_overlap = config['size'], config['overlap']

        print(f"--- Testing Chunk Config: size={chunk_size}, "
              f"overlap={chunk_overlap} ---")

        index: VectorStoreIndex = get_or_build_index(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_model=embed_model_to_test
        )

        query_engine: BaseQueryEngine = index.as_query_engine(
            similarity_top_k=SIMILARITY_TOP_K,
            llm=llm_to_test
        )

        qa_dataset: Dataset = generate_qa_dataset(
            query_engine,
            questions,
            ground_truths
        )

        print("--- Running Ragas evaluation for chunking... ---")

        # --- If you don't have a Rate per Minute limit on your API ---
        # results_df: pd.DataFrame = evaluate_without_rate_limit(
        #     qa_dataset,
        #     ragas_llm,
        #     ragas_embeddings,
        # )

        # --- If you do have a Rate per Minute API limit ---
        results_df: pd.DataFrame = evaluate_with_rate_limit(
            qa_dataset,
            ragas_llm,
            ragas_embeddings,
        )

        # Add Chunk Size and Chunk Overlap to DataFrame to help tracking
        results_df['chunk_size'] = chunk_size
        results_df['chunk_overlap'] = chunk_overlap

        all_results.append(results_df)

    final_df: pd.DataFrame = pd.concat(all_results, ignore_index=True)

    save_results(final_df, "chunking_evaluation")

    print("--- ✅ Chunking Strategy Evaluation Complete ---")


def evaluate_reranker_strategies() -> None:
    """
    Evaluates different reranker settings on top of the best chunking
    strategy (auto-selected from the latest chunking_evaluation results).
    """
    print("\n--- 🚀 Stage 3: Evaluating Reranker Strategies ---")

    best_chunking: dict[str, int] = _resolve_best_chunking()
    chunk_size: int = best_chunking["chunk_size"]
    chunk_overlap: int = best_chunking["chunk_overlap"]

    llm_to_test: Groq = initialise_llm()

    embed_model_to_test: HuggingFaceEmbedding = get_embedding_model()

    questions, ground_truths = get_evaluation_data()

    ragas_llm: LlamaIndexLLMWrapper
    ragas_embeddings: HuggingFaceEmbeddings
    ragas_llm, ragas_embeddings = load_ragas_models()

    index: VectorStoreIndex = get_or_build_index(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=embed_model_to_test,
    )

    all_results: list[pd.DataFrame] = []

    for config in RERANKER_CONFIGS:
        retriever_k, reranker_n = config['retriever_k'], config['reranker_n']

        print(f"--- Testing Reranker Config: retrieve_k={retriever_k}, "
              f"rerank_n={reranker_n} ---")

        retriever = index.as_retriever(similarity_top_k=retriever_k)

        reranker = SentenceTransformerRerank(
            top_n=reranker_n, model=RERANKER_MODEL_NAME
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker],
            llm=llm_to_test,
        )

        qa_dataset: Dataset = generate_qa_dataset(
            query_engine,
            questions,
            ground_truths,
        )

        print("--- Running Ragas evaluation for reranker... ---")

        results_df: pd.DataFrame = evaluate_with_rate_limit(
            qa_dataset,
            ragas_llm,
            ragas_embeddings,
        )

        results_df['chunk_size'] = chunk_size
        results_df['chunk_overlap'] = chunk_overlap
        results_df['retriever_k'] = retriever_k
        results_df['reranker_n'] = reranker_n

        all_results.append(results_df)

    final_df: pd.DataFrame = pd.concat(all_results, ignore_index=True)

    save_results(final_df, "reranker_evaluation")

    print("--- ✅ Reranker Strategy Evaluation Complete ---")


def evaluate_query_rewriting() -> None:
    """
    Evaluates the impact of HyDE on top of the best chunking and reranker
    configurations (both auto-selected from prior stage results).
    """
    print("\n--- 🚀 Stage 4: Evaluating Query Rewriting (HyDE) ---")

    best_chunking: dict[str, int] = _resolve_best_chunking()
    chunk_size: int = best_chunking["chunk_size"]
    chunk_overlap: int = best_chunking["chunk_overlap"]

    best_reranker: dict[str, int] = _resolve_best_reranker()
    best_retriever_k: int = best_reranker["retriever_k"]
    best_reranker_n: int = best_reranker["reranker_n"]

    llm_to_test: Groq = initialise_llm()

    embed_model_to_test: HuggingFaceEmbedding = get_embedding_model()

    questions, ground_truths = get_evaluation_data()

    index: VectorStoreIndex = get_or_build_index(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=embed_model_to_test,
    )

    ragas_llm: LlamaIndexLLMWrapper
    ragas_embeddings: HuggingFaceEmbeddings
    ragas_llm, ragas_embeddings = load_ragas_models()

    all_results: list[pd.DataFrame] = []

    for use_hyde in [False, True]:
        print(f"\n--- Testing Query Rewrite Config: use_hyde={use_hyde} ---")

        retriever = index.as_retriever(similarity_top_k=best_retriever_k)

        reranker = SentenceTransformerRerank(
            top_n=best_reranker_n,
            model=RERANKER_MODEL_NAME,
        )

        base_query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker],
            llm=llm_to_test,
        )

        if use_hyde:
            hyde_transform = HyDEQueryTransform(
                llm=llm_to_test,
                include_original=True,
            )
            query_engine = TransformQueryEngine(
                base_query_engine,
                query_transform=hyde_transform,
            )
        else:
            query_engine = base_query_engine

        qa_dataset: Dataset = generate_qa_dataset(
            query_engine,
            questions,
            ground_truths,
        )

        print("--- Running Ragas evaluation for query rewriting... ---")

        results_df: pd.DataFrame = evaluate_with_rate_limit(
            qa_dataset,
            ragas_llm,
            ragas_embeddings,
        )

        results_df['chunk_size'] = chunk_size
        results_df['chunk_overlap'] = chunk_overlap
        results_df['retriever_k'] = best_retriever_k
        results_df['reranker_n'] = best_reranker_n
        results_df['use_hyde'] = use_hyde

        all_results.append(results_df)

    final_df: pd.DataFrame = pd.concat(all_results, ignore_index=True)

    save_results(final_df, "query_rewrite_evaluation")

    print("--- ✅ Query Rewrite Evaluation Complete ---")

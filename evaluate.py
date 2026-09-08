from evaluation.evaluation_engine import (
    evaluate_baseline,
    evaluate_chunking_strategies,
    evaluate_reranker_strategies,
    evaluate_query_rewriting,
)


if __name__ == "__main__":
    # Each stage auto-reads the winning config from the prior stage's
    # latest results CSV. To re-run just one stage, comment out the others.

    # Stage 1: Baseline Evaluation
    # evaluate_baseline()

    # Stage 2: Chunking Strategy Evaluation
    # evaluate_chunking_strategies()

    # Stage 3: Reranker Strategy Evaluation
    #   (auto-picks best chunk_size/chunk_overlap from stage 2)
    # evaluate_reranker_strategies()

    # Stage 4: Query Rewriter / HyDE Evaluation
    #   (auto-picks best chunking from stage 2 AND best reranker from stage 3)
    evaluate_query_rewriting()

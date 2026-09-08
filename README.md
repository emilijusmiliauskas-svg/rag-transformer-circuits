# Transformer Circuits RAG: An Ablation Study

A retrieval-augmented generation system over six Anthropic mechanistic-interpretability papers — built with LlamaIndex and Groq, then **measured at every stage** with a RAGAS evaluation pipeline rather than tuned by intuition.

Four staged experiments: baseline → chunking strategy → cross-encoder reranking → query rewriting. Each stage reads the winning configuration from the previous stage's results and sweeps one variable.

---

## What Was Actually Measured

Every number below comes from a committed CSV in [`evaluation/evaluation_results/`](evaluation/evaluation_results/).

### Stage 2 — Chunking strategy

| Chunk size | Overlap | Faithfulness | Answer correctness | Context precision |
|-----------:|--------:|:------------:|:------------------:|:-----------------:|
| **512** | 50 | 1.000 | 0.486 | 1.000 |
| 768 | 115 | 0.778 | **0.502** | 1.000 |
| 1024 | 200 | 1.000 | 0.387 | 1.000 |

Larger chunks were worse. At 1024 tokens, answer correctness fell 20% against the 512 baseline — retrieving a large block dilutes the relevant passage with surrounding text the LLM then has to filter. 768 edged out 512 on correctness but lost faithfulness, so 512 carried forward.

### Stage 3 — Cross-encoder reranking

| Retriever k | Reranker n | Faithfulness | Answer correctness | Context precision | Context recall |
|------------:|-----------:|:------------:|:------------------:|:-----------------:|:--------------:|
| 10 | 2 | 0.454 | 0.535 | 0.500 | 0.667 |
| 10 | 5 | 0.906 | 0.659 | 0.722 | **1.000** |
| 20 | 5 | **0.925** | **0.664** | 0.722 | 0.833 |

**Reranking was the single biggest win** — answer correctness rose from 0.486 to 0.659 by retrieving 10 chunks and letting a cross-encoder pick the best 5.

Cutting to the top 2 was actively harmful: faithfulness collapsed to 0.454 and recall to 0.667. Two chunks simply don't contain enough evidence, and the model fills the gap by inventing. Widening the pool to k=20 bought a marginal correctness gain while *dropping* recall to 0.833 — more candidates gave the reranker more chances to discard something it needed. `k=10, n=5` was selected.

### Stage 4 — Query rewriting (HyDE)

| HyDE | Faithfulness | Answer correctness | Context precision |
|:----:|:------------:|:------------------:|:-----------------:|
| off | 0.838 | **0.678** | **0.722** |
| on | **0.885** | 0.580 | 0.522 |

**HyDE made things worse and was rejected.** Generating a hypothetical answer to retrieve against cost 14% of answer correctness and 28% of context precision.

The likely reason is domain mismatch: HyDE works by having the LLM draft a plausible answer and embedding *that*. On narrow technical material — superposition, induction heads — the model's hypothetical answer drifts toward generic ML language, and the drafted text pulls retrieval away from the actual papers. Faithfulness rose slightly because the model grew more cautious with weaker context, which is not an improvement worth having.

## Final Configuration

The winning setup, in [`src/config.py`](src/config.py):

```python
LLM_MODEL            = "openai/gpt-oss-120b"          # via Groq
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE           = 512
CHUNK_OVERLAP        = 50
SIMILARITY_TOP_K     = 10                             # pool before reranking
RERANKER_MODEL_NAME  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_N       = 5                              # passed to the LLM
```

Evaluation deliberately uses a **different model family from generation** — `moonshotai/kimi-k2-instruct` as judge and `BAAI/bge-large-en-v1.5` for embeddings — so the system is not grading its own work with its own weights.

## Honest Limitations

- **The evaluation set is 4 questions.** That is far too small to separate close configurations, and it shows: the same 512/50 setup scored 0.573 correctness in the baseline run and 0.486 in the chunking run. Differences under roughly 0.1 in these tables should be treated as noise, and the HyDE and reranking conclusions rest on gaps large enough to survive it — the 768-vs-512 comparison does not.
- **Ground truths are hand-written by me**, so answer correctness measures agreement with my reading of the papers.
- **Single run per configuration.** No seeds, no repeats, no confidence intervals. Repeating each configuration 3–5 times would be the first thing to add.

## Architecture

```
rag_project/
├── app.py                    # Streamlit UI — chat with source inspection
├── main.py                   # Terminal chat entry point
├── evaluate.py               # Runs the four evaluation stages
├── src/
│   ├── config.py             # Production settings (the winning values above)
│   ├── model_loader.py       # Groq LLM + HuggingFace embeddings, cached
│   └── engine.py             # Vector store + CondensePlusContext chat engine
├── evaluation/
│   ├── evaluation_engine.py  # The four staged experiments
│   ├── evaluation_config.py  # Judge model, metrics, sweep grids
│   ├── evaluation_questions.py
│   └── evaluation_results/   # Committed CSVs — every number in this README
├── initial/                  # 7 notebooks: the build, step by step
└── data/                     # 6 HTML papers from transformer-circuits.pub
```

Retrieval uses `CondensePlusContextChatEngine`, which rewrites a follow-up question into a standalone query using chat history before retrieving — so multi-turn conversation works rather than each turn retrieving on a fragment.

Each evaluation stage auto-reads the previous stage's winning configuration from its results CSV, so the pipeline is re-runnable end to end without hand-copying parameters between stages.

## The Corpus

Six papers from [transformer-circuits.pub](https://transformer-circuits.pub): induction heads and in-context learning, toy models of superposition, scaling monosemanticity, interpretability dreams, and the mathematical framework for transformer circuits.

Deliberately chosen as a hard test — dense technical prose with precise terminology, where retrieving *approximately* the right passage produces a confidently wrong answer.

## Running It

```bash
conda env create -f environment.yml
conda activate rag-project-env

cp .env.example .env        # add your Groq API key
streamlit run app.py        # web UI
python main.py              # terminal chat
python evaluate.py          # re-run the evaluation stages
```

First run downloads the embedding model and builds the vector store into `local_storage/` (git-ignored, ~1.3 GB). The evaluation pipeline sleeps between calls to stay inside Groq's free-tier rate limits, so a full four-stage run takes a while.

## Notebooks

`initial/` holds the build as seven steps — bare LLM, chatbot, RAG, evaluation, chunking, reranking, query rewriting — each one adding a single component. Useful for following the reasoning; `src/` is the consolidated result.

## License

MIT

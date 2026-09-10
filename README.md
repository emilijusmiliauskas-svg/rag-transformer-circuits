# Transformer Circuits RAG: An Ablation Study

A retrieval-augmented generation system over six Anthropic mechanistic-interpretability papers — built with LlamaIndex, then **tuned by measurement rather than intuition**, and then re-run from scratch after the corpus turned out to be almost entirely noise.

The system itself is ordinary: LlamaIndex, a vector store, a cross-encoder reranker, a Streamlit front end. The point of the repository is the evaluation around it, and what happened when the evaluation was pointed at its own inputs.

---

## The Short Version

Four staged experiments — baseline → chunking → reranking → query rewriting — each feeding its winning configuration into the next, scored on four RAGAS metrics by an independent judge model.

Then the index was audited and found to be **97.7% base64 image data**. Six papers had produced 110,370 nodes; after parsing the HTML properly, they produce **290**.

Everything was re-run on the clean corpus. **One of the three original conclusions reversed. Two held.**

| Stage | Winner | vs. the original run |
|---|---|---|
| Chunking | **1024 / 200** | **Reversed** — 512 had won, 1024 had been worst |
| Reranking | **k=10, n=5** | Confirmed |
| Query rewriting (HyDE) | **off** | Confirmed |

That mix is the useful result. Not "the old numbers were all wrong," and not "nothing changed" — one specific conclusion was an artifact of the data, and the audit is what exposed it.

## What's Actually In Here

Anyone can point LlamaIndex at a folder of PDFs and get a chatbot. That part of this repository is unremarkable and took an afternoon. Here is everything else, roughly in order of how commonly you'd find it in a project this size:

| | Where to look |
|---|---|
| Embeddings, a vector store, a Streamlit chat UI | `src/engine.py`, `app.py` |
| A **cross-encoder reranker** — retrieve 10, re-score, pass the best 5 to the model. Retrieval has stages; one similarity search is rarely enough | `src/engine.py` |
| **Measurement instead of impressions** — four RAGAS metrics, every result written to CSV and committed rather than screenshotted | `evaluation/evaluation_results/` |
| **A staged design**, where each experiment inherits the previous winner instead of testing everything against defaults | `evaluation/evaluation_engine.py` |
| **A technique tried and rejected.** HyDE is fashionable and it made things worse here, so it isn't in the final configuration | Stage 4, below |
| **An audit of the inputs.** The index turned out to be 97.7% base64 image data. Finding that meant re-running everything and publishing a correction | `src/corpus.py` |
| **A statement of what the numbers can't support.** The same configuration scored 0.861 and 0.956 on two different runs, so gaps under ~0.1 are noise, and the README says so | *Honest Limitations*, below |

The last three are the ones worth your time. Tuning a retrieval pipeline is a solved, documented exercise. Noticing that a careful tuning run was measuring the wrong thing entirely — and that one of its conclusions inverts once fixed — is the part that isn't in any tutorial.

## Results

All figures come from the committed CSVs in [`evaluation/evaluation_results/`](evaluation/evaluation_results/).

### Stage 1 — Baseline (512 / 50)

| Faithfulness | Answer correctness | Context precision | Context recall |
|:---:|:---:|:---:|:---:|
| 0.632 | 0.473 | 0.703 | 0.667 |

### Stage 2 — Chunking strategy

| Chunk / overlap | Faithfulness | Answer correctness | Context precision | Context recall |
|---|:---:|:---:|:---:|:---:|
| 512 / 50 | 0.747 | 0.546 | 0.612 | 0.833 |
| 768 / 115 | 0.620 | 0.633 | 0.756 | 0.833 |
| **1024 / 200** | **0.902** | **0.682** | 0.691 | **1.000** |

**Larger chunks win, which is the opposite of the original finding.** The first run reported 1024 as the worst configuration, with answer correctness 20% below 512, and explained it as a large block diluting the relevant passage with surrounding text.

That explanation was wrong, but the measurement was real: when 97.7% of the index is base64, a bigger chunk really does sweep in proportionally more binary. Remove the binary and the effect inverts — more context helps, as you would expect.

### Stage 3 — Cross-encoder reranking (at 1024 / 200)

| retriever k | reranker n | Faithfulness | Answer correctness | Context precision |
|---:|---:|:---:|:---:|:---:|
| 10 | 2 | 0.795 | 0.494 | 0.667 |
| **10** | **5** | **0.861** | **0.620** | **0.796** |
| 20 | 5 | 0.804 | 0.601 | 0.717 |

Both failure modes from the original run reproduce. **Cutting to the top 2 is actively harmful** — two chunks don't carry enough evidence and the model fills the gap. **Widening the pool to k=20 doesn't help either**: more candidates give the reranker more opportunities to discard something it needed, and precision drops.

### Stage 4 — Query rewriting with HyDE (at 1024 / 200, k=10, n=5)

| HyDE | Faithfulness | Answer correctness | Context precision |
|:---:|:---:|:---:|:---:|
| **off** | **0.956** | **0.702** | **0.712** |
| on | 0.736 | 0.594 | 0.679 |

**Rejected, as before.** Generating a hypothetical answer to retrieve against costs 23% faithfulness and 15% answer correctness.

The likely reason is domain mismatch. HyDE works by having the model draft a plausible answer and embedding *that*. On narrow technical material — superposition, induction heads, sparse autoencoders — the draft drifts toward generic ML language and pulls retrieval away from the actual papers.

## The Corpus Problem

The papers were saved as single-file HTML with inline base64 images. `SimpleDirectoryReader` reads those files as plain text, so the image data was chunked and embedded alongside the prose.

| | Before | After |
|---|---:|---:|
| Bytes on disk | 51.3 MB | 51.3 MB |
| Extractable text | — | 596k chars (**1.16%**) |
| Nodes at 512/50 | 110,370 | **290** |
| Base64 blobs (n=1000 sample) | 97.7% | — |

`monosemanticity.html` alone is 20 MB, almost none of which is text.

[`src/corpus.py`](src/corpus.py) parses the HTML with BeautifulSoup, drops `script`, `style` and `head`, and strips `data:` URIs before chunking. Both the application and the evaluation harness load through it.

The lesson is the part worth keeping: **no amount of retrieval tuning compensates for a corpus you did not inspect.** The first round of tuning was careful, internally consistent, and pointed the wrong way on chunk size.

## Final Configuration

In [`src/config.py`](src/config.py):

```python
LLM_MODEL            = "openai/gpt-oss-120b"          # via OpenRouter
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE           = 1024
CHUNK_OVERLAP        = 200
SIMILARITY_TOP_K     = 10                             # pool before reranking
RERANKER_MODEL_NAME  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_N       = 5                              # passed to the LLM
```

Evaluation uses **DeepSeek V4 Flash** as judge and `BAAI/bge-large-en-v1.5` for its embeddings — deliberately a different model family from the generator, so the system is not grading its own work with its own weights.

## Honest Limitations

- **The evaluation set is 3 questions.** Far too small to separate close configurations.
- **Run-to-run variance is larger than several of the gaps above.** The identical configuration (1024/200, k=10, n=5, no HyDE) scored **0.861 faithfulness / 0.620 correctness** in stage 3 and **0.956 / 0.702** in stage 4. Differences smaller than roughly 0.1 should not be read as real. The chunking reversal and the HyDE rejection are large enough to survive this; the 768-vs-512 comparison is not.
- **Single run per configuration.** No repeats, no confidence intervals. Repeating each configuration 3–5 times is the first thing to add.
- **Ground truths are hand-written by me**, so answer correctness measures agreement with my reading of the papers.
- **Stage 1 ran on a different judge** (`moonshotai/kimi-k2-instruct`, since retired from Groq) than stages 2–4. Its absolute numbers are not directly comparable with the later stages; the within-stage rankings are.

## Architecture

```
rag_project/
├── app.py                    # Streamlit UI — chat with source inspection
├── main.py                   # Terminal chat entry point
├── evaluate.py               # Runs the four evaluation stages
├── src/
│   ├── corpus.py             # HTML → prose. The fix described above
│   ├── config.py             # Production settings (the winning values)
│   ├── model_loader.py       # Generator + embeddings
│   └── engine.py             # Vector store + CondensePlusContext chat engine
├── evaluation/
│   ├── evaluation_engine.py  # The four staged experiments
│   ├── evaluation_config.py  # Judge model, metrics, sweep grids
│   ├── retrying_llm.py       # Re-asks when the judge returns an empty body
│   └── evaluation_results/   # Committed CSVs — every number in this README
├── initial/                  # 7 notebooks: the build, step by step
└── data/                     # 6 HTML papers from transformer-circuits.pub
```

Retrieval uses `CondensePlusContextChatEngine`, which rewrites a follow-up question into a standalone query using chat history before retrieving — so multi-turn conversation works rather than each turn retrieving on a fragment.

Each evaluation stage auto-reads the previous stage's winning configuration from its results CSV, so the pipeline re-runs end to end without hand-copying parameters between stages.

## Two Provider Problems Worth Recording

Both cost real time, and neither is obvious from the documentation.

**Reasoning tokens are drawn from the answer's budget.** DeepSeek V4 and gpt-oss both spend hidden reasoning tokens out of `max_tokens`. When reasoning consumes the allowance, the API returns a well-formed HTTP 200 with an **empty content string** — no error, no truncation flag. RAGAS cannot parse that, its repair prompt has nothing to repair, and the entire evaluation aborts. Measured on the Faithfulness prompt: thinking on at 2048 tokens gave 5/5 empty responses; thinking off gave 0/5. Disabling thinking for the judge fixed it outright.

**Free tiers cannot carry this workload.** Groq caps at 200,000 tokens/day per model. One four-stage ablation exhausts that — especially once 1024-token chunks meant every generation carried ~5k tokens of retrieved context. Generation moved to the same model on OpenRouter, which kept results comparable across stages while removing the ceiling. The full run cost roughly one cent.

## The Corpus

Six papers from [transformer-circuits.pub](https://transformer-circuits.pub): induction heads and in-context learning, toy models of superposition, scaling monosemanticity, interpretability dreams, and the mathematical framework for transformer circuits.

Deliberately chosen as a hard test — dense technical prose with precise terminology, where retrieving *approximately* the right passage produces a confidently wrong answer.

## Running It

```bash
conda env create -f environment.yml
conda activate rag-project-env

cp .env.example .env        # add your API keys
streamlit run app.py        # web UI
python main.py              # terminal chat
python evaluate.py          # re-run the evaluation stages
```

First run downloads the embedding model and builds the vector store into `local_storage/` (git-ignored). Individual stages can be enabled or disabled by commenting entries in `evaluate.py`.

## Notebooks

`initial/` holds the build as seven steps — bare LLM, chatbot, RAG, evaluation, chunking, reranking, query rewriting — each adding a single component. Useful for following the reasoning; `src/` is the consolidated result.

## License

MIT

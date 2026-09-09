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

Evaluation uses a separate judge model and a separate embedding model from generation — see *Status* for the independence caveat the free tier forces.

## The Corpus Was 97.7% Noise — Now Fixed, Numbers Pending Re-Run

Reviewing this repository after the fact, the index turns out to be mostly garbage.

The corpus was saved as **single-file HTML with inline base64 images**, and `SimpleDirectoryReader` ingests those files as raw text. The base64 blobs get chunked and embedded alongside the prose:

| | |
|---|---:|
| Nodes in the vector store | 110,370 |
| Base64/binary blobs (sampled, n=1000) | **97.7%** |
| Estimated real prose nodes | ~2,500 |

Six papers should produce a few thousand chunks. They produced a hundred and ten thousand, because `monosemanticity.html` alone is 20 MB and almost none of that is text.

**This reframes every result above.** The tuning was real and the measurements are honest, but they were all made over an index that is 97.7% noise — which plausibly explains the pattern in the numbers:

- **Why reranking was the biggest win by such a margin.** The cross-encoder was earning its gains partly by discarding junk the embedding retriever kept surfacing. Against a clean index, its marginal value would likely be much smaller.
- **Why context precision sat at 0.72** in the later stages rather than the 1.0 the baseline reported.
- **Why HyDE degraded so sharply.** A hypothetical answer embedded against a corpus dominated by base64 has far more opportunity to land on nothing useful.

### The fix, now applied

[`src/corpus.py`](src/corpus.py) parses the HTML instead of reading it as text, strips scripts, styles and `head`, and drops `data:` URIs before chunking. Both the app and the evaluation harness now load through it.

| | Before | After |
|---|---:|---:|
| Bytes on disk | 51.3 MB | 51.3 MB |
| Text extracted | — | 596k chars (**1.16%**) |
| Nodes at 512/50 | 110,370 | **290** |

**The tables above are still the old numbers, measured over the polluted index.** They are left in place, clearly marked, because deleting them would hide the finding. The re-run is blocked on API quota rather than on anything in this repository — see *Status* below.

The lesson stands regardless: **no amount of retrieval tuning compensates for a corpus you did not inspect.**

### Other fixes applied alongside

- **Generation parameters now reach the model.** `temperature`, `max_tokens` and `top_p` are passed to `Groq()`; they were previously declared and ignored. `repetition_penalty` has no equivalent in Groq's OpenAI-compatible API and is documented as such rather than left as a dead constant.
- **The system prompt now grounds the model** — answer only from context, say so when the context does not contain the answer — replacing `"You are a helpful chatbot. Be friendly and conversational."`
- **Judge model replaced.** `moonshotai/kimi-k2-instruct` was retired from Groq. See *Status* for what replaced it and at what cost.
- **Vector-store loading** no longer treats a stray `.DS_Store` as a valid persisted index.

## Status: Re-Run Blocked on Free-Tier Quota

The clean-corpus ablation has not produced numbers yet. Five attempts each failed for a different, identifiable reason:

| Judge configuration | Failure |
|---|---|
| `qwen3.8-27b`, default budget | JSON truncated mid-object; Pydantic parse failed |
| `qwen3.8-27b`, 8000 tokens | Free tier caps this model at 1000 output tokens/minute |
| `gpt-oss-20b`, 2048 tokens | Truncated again — gpt-oss spends part of the budget on a hidden reasoning trace |
| `gpt-oss-20b`, 2048 + `reasoning_effort: low` | Judge under-thought and returned an empty `{}` |
| `gpt-oss-20b`, 8000 tokens | Validated on a single question, then exhausted the 200,000 token/day cap |

`AnswerCorrectness` asks the judge to classify every statement in an answer against the ground truth, which is a long structured generation — precisely the shape a throttled free tier handles worst. The configuration is now validated (`faithfulness 0.667, answer_correctness 0.734, context_precision 1.0, context_recall 1.0` on a single question) and the run needs only quota to complete.

## Honest Limitations

- **The evaluation set is 3 questions.** That is far too small to separate close configurations, and it shows: the same 512/50 setup scored 0.573 correctness in the baseline run and 0.486 in the chunking run. Differences under roughly 0.1 in these tables should be treated as noise, and the HyDE and reranking conclusions rest on gaps large enough to survive it — the 768-vs-512 comparison does not.
- **Ground truths are hand-written by me**, so answer correctness measures agreement with my reading of the papers.
- **Single run per configuration.** No seeds, no repeats, no confidence intervals. Repeating each configuration 3–5 times would be the first thing to add.
- **The judge is no longer from a different family than the generator.** The original design used a cross-family judge so the system would not grade its own work. On Groq's free tier that is not currently reachable, so the judge is `openai/gpt-oss-20b` against a `openai/gpt-oss-120b` generator: a different model and size, same family. That is a weaker independence guarantee and should be read as one.

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

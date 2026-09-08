# RAG From Scratch

A Retrieval-Augmented Generation system built one pipeline stage at a time, from
scratch — no LangChain, no LlamaIndex, no vector database. The goal isn't a
working demo; it's understanding what every layer of a "real" RAG stack
(chunking, embeddings, vector search, BM25, hybrid fusion, reranking,
evaluation) actually does internally, so those primitives aren't hidden
behind a framework call.

Currently implements **V1–V5** of a planned V1–V11 progression (see
[Roadmap](#roadmap)). Every stage is swappable via config and measurable
against the same retrieval-evaluation harness.

## Pipeline

```
documents/*.{txt,md}
    │
    ▼
rag/loader.py        load_documents()      →  Document(text, source)
    │
    ▼
rag/chunker.py        chunk_documents()      →  Chunk(text, source, chunk_index,
    │                                              section_title)
    │                                            3 strategies: fixed / paragraph /
    │                                            recursive (CHUNK_STRATEGY)
    ▼
rag/embedder.py        embed_chunks()         →  Gemini gemini-embedding-001
    │                                            (rate-limit-throttled)
    ▼
rag/vector_store.py     VectorStore            in-memory brute-force cosine search
rag/bm25.py             BM25Index              from-scratch keyword search (V4)
rag/hybrid_search.py    hybrid_search()        fuses vector + BM25 via Reciprocal
    │                                          Rank Fusion (RETRIEVAL_MODE=vector|hybrid)
    ▼
rag/reranker.py          rerank()               local cross-encoder reranking
    │                                          (sentence-transformers,
    │                                          ms-marco-MiniLM-L-6-v2, no API call)
    ▼
rag/prompt.py             build_prompt()         numbered context block + section-title
    │                                          citations, cite-only-from-context
    ▼
rag/generator.py           generate_answer()      Gemini gemini-flash-lite-latest
    ▼
Answer + Sources
```

`rag/pipeline.py` wires all of the above into `Pipeline.index(directory)` and
`Pipeline.ask(query)` — read that file first, it's the whole system in one
place.

`rag/config.py` is the single place that reads environment variables (via
`.env`); no other module touches `os.environ` directly.

## Why these design choices

- **No framework.** BM25, RRF fusion, chunking strategies, and the debug/eval
  harness are all hand-implemented so the underlying algorithm is visible,
  not abstracted away. See `reference/langchain_equivalent.py` for the same
  pipeline rebuilt with LangChain, annotated with what each call replaces and
  its hidden costs (double retrieval for sources, lost default observability,
  dozens of extra dependencies) — kept purely for comparison.
- **Hybrid search (vector + BM25)** exists because embeddings alone can dilute
  an exact term (an acronym like "SSTF") into the surrounding topic's
  semantic vocabulary; BM25 has no such confusion since it matches literal
  tokens. Reciprocal Rank Fusion combines them without needing raw scores on
  the same scale.
- **Reranking with a cross-encoder** exists because bi-encoder retrieval
  (embeddings) scores query and chunk independently for speed, which
  structurally caps how much cross-relevance it can capture. A cross-encoder
  scores query+chunk together on a small candidate set (N) to pick a better
  final top-K.
- **Everything is measured, not assumed.** `rag/evaluation.py` implements
  Recall@K, Precision@K, MRR, and nDCG@K from scratch against a 24-question,
  file-level ground-truth dataset (`eval/qa_dataset.py`), so every retrieval
  change (chunking strategy, hybrid vs. vector-only, reranking on/off) is
  compared quantitatively instead of by intuition. See `eval/METRICS.md` for
  what each metric means and why all four are tracked.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with:

```
GEMINI_API_KEY=your_key_here

# Optional overrides (defaults shown)
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
GEMINI_GENERATION_MODEL=gemini-flash-lite-latest
CHUNK_SIZE_CHARS=1000
CHUNK_OVERLAP_CHARS=150
CHUNK_STRATEGY=fixed          # fixed | paragraph | recursive
RETRIEVAL_MODE=vector         # vector | hybrid
RERANK_ENABLED=false
RERANK_CANDIDATE_COUNT=20
TOP_K=5
RAG_DEBUG=false                # print every intermediate pipeline value
```

## Usage

```bash
# Index a directory once, then ask questions interactively against it
python main.py documents/

# V2: retrieval-only evaluation (Recall@K / Precision@K / MRR / nDCG@K)
python run_eval.py documents/

# V3: same eval, once per chunking strategy (fixed / paragraph / recursive)
python compare_chunking.py documents/

# V4: vector-only vs. hybrid (vector + BM25) retrieval
python compare_retrieval.py documents/

# V5: plain retrieval vs. cross-encoder-reranked retrieval
python compare_reranking.py documents/
```

Set `RAG_DEBUG=true` to see every intermediate value per query: query
embedding dimensions, retrieved chunks with similarity/BM25/rerank scores,
the constructed prompt, model name, token usage, latency, final answer, and
sources — the system should never be a black box.

## Repository layout

```
rag/
  loader.py          Document loading
  chunker.py          Chunking (fixed / paragraph / recursive strategies)
  embedder.py          Gemini embeddings (query vs. document task types)
  vector_store.py      In-memory brute-force cosine similarity search
  bm25.py               From-scratch BM25 keyword search
  hybrid_search.py       Reciprocal Rank Fusion of vector + BM25
  reranker.py            Local cross-encoder reranking
  prompt.py               Prompt construction with citations
  generator.py              Gemini answer generation
  pipeline.py               Wires all stages: index() / retrieve() / ask()
  config.py                 Single source of truth for env-driven config
  evaluation.py             Recall@K / Precision@K / MRR / nDCG@K harness
eval/
  qa_dataset.py        24 hand-written questions with file-level ground truth
                       (includes questions with no answer in the corpus)
  METRICS.md            Explanation of each retrieval metric
documents/              Source corpus (OS concepts: scheduling, deadlocks,
                        memory management, processes/threads, OSTEP excerpts)
reference/
  langchain_equivalent.py  Same pipeline rebuilt with LangChain, for comparison
rag_architecture.html       Reference notes on how large-scale RAG systems are
                            architected (vocabulary/context, not a build target)
main.py                     Interactive CLI entry point
run_eval.py / compare_*.py  Evaluation scripts (V2-V5)
```

## Roadmap

| Version | Focus | Status |
|---|---|---|
| V1 | Basic RAG (load → chunk → embed → retrieve → prompt → generate) | ✅ Done |
| V2 | Retrieval fundamentals + evaluation harness | ✅ Done |
| V3 | Chunking strategies + metadata (section titles, source tracking) | ✅ Done |
| V4 | Hybrid search (BM25 + vector, RRF fusion) | ✅ Done |
| V5 | Cross-encoder reranking | ✅ Done |
| V6 | Query transformation (rewriting, HyDE, decomposition) | ⏭️ Deliberately skipped |
| V7 | Production RAG (persistence, API layer, retries, model abstraction) | 🔜 Not started |
| V8 | Security (auth, document-level permissions, tenant isolation) | Not started |
| V9 | Evaluation + observability (faithfulness, tracing, cost tracking) | Not started |
| V10 | Scaling (caching, async workers, queues) | Not started |
| V11 | Agentic RAG (tool calling, multi-step retrieval) | Not started |

V6 was explicitly skipped by design choice, not oversight — query
transformation is worth doing only once it's clear it measurably improves
retrieval, which requires V7's usage patterns to justify.

### Known gaps

- `compare_chunking.py`, `compare_retrieval.py`, and `compare_reranking.py`
  have been verified structurally/offline (e.g. a standalone rerank test
  correctly scored a relevant chunk +9.47 vs. -4.05/-8.69 for wrong chunks)
  but not yet run end-to-end against the live Gemini API for real numbers.
- No automated test suite yet (`pytest` is in `requirements.txt`, but no
  test files exist). Per-stage tests for loading, chunking, embedding,
  retrieval, ranking, and citation mapping are a deferred next step.

## Learning approach

This project intentionally avoids RAG frameworks in its early stages so the
underlying concepts (what retrieval actually computes, why hybrid search
helps, why cross-encoders reorder results bi-encoders can't) are learned
before being wrapped in an abstraction. Every architectural decision is
expected to answer: what problem does it solve, what's the simplest
solution, when does that stop being sufficient, and what does the
production alternative cost in latency/complexity/failure modes. See
`CLAUDE.md` for the full learning philosophy and version ladder this
project follows.

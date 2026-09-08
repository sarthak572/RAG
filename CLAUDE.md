# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# RAG Learning Project — Permanent Instructions

## Repository State

We are building one evolving RAG system, one module per pipeline stage, per
the version ladder below. **V1 through V5 are implemented and wired
together in `rag/pipeline.py`** (read this file first — it's the whole
system in one place):

    documents/*.{txt,md}, documents/ostep/*.txt
       ↓ rag/loader.py        load_documents() -> Document(text, source)
    rag/chunker.py             chunk_documents() -> Chunk(text, source,
                                chunk_index, section_title) -- 3 strategies
                                (fixed/paragraph/recursive, via CHUNK_STRATEGY)
    rag/embedder.py            embed_chunks() / embed_query() -> Gemini
                                gemini-embedding-001 (RETRIEVAL_DOCUMENT vs
                                RETRIEVAL_QUERY task_type), rate-limit-throttled
    rag/bm25.py                BM25Index -- from-scratch keyword search
    rag/vector_store.py        VectorStore -- in-memory brute-force cosine
    rag/hybrid_search.py       hybrid_search() -- fuses vector+BM25 via
                                Reciprocal Rank Fusion (RETRIEVAL_MODE=vector|hybrid)
    rag/reranker.py            rerank() -- local cross-encoder
                                (sentence-transformers, ms-marco-MiniLM-L-6-v2),
                                no API call (RERANK_ENABLED + RERANK_CANDIDATE_COUNT)
    rag/prompt.py               build_prompt() -> numbered context block +
                                section_title citations, cite-only-from-context
    rag/generator.py           generate_answer() -> Gemini gemini-flash-lite-latest
    rag/pipeline.py            Pipeline.index(directory) / Pipeline.retrieve()
                                / Pipeline.ask() -- wires all of the above

`rag/config.py` is the single place that reads environment variables (via
`.env`); no other module calls `os.environ` directly. `RAG_DEBUG=true` makes
`pipeline.py` print every intermediate value per the Debugging Requirement.
`rag/evaluation.py`'s `evaluate_retrieval()` takes a generic
`retrieve_fn(question, k)` closure, so `run_eval.py` (V2), `compare_chunking.py`
(V3), `compare_retrieval.py` (V4), and `compare_reranking.py` (V5) all reuse
the same Recall@K/Precision@K/MRR/nDCG@K harness against `eval/qa_dataset.py`
(24 questions, file-level ground truth, includes deliberately not-in-corpus
questions) — see `eval/METRICS.md` for what the metrics mean.

**V6 (Query Transformation) is deliberately skipped** — explicit user
decision, not an oversight. Explained conceptually but not built.

**V7 (Production RAG) has been discussed but not started.** Prioritized plan
(most pain-motivated first): persistence (save/load the index across
restarts — V1-V5 all re-embed everything on every run) → API layer (FastAPI)
→ document updates/deletions → reactive retry/backoff on API calls → model
provider abstraction. Async/job queues deliberately deprioritized (no scale
justification yet, per the Architecture Rule). **Open decision, not yet
made: start with persistence or the API layer.**

`reference/langchain_equivalent.py` shows the same pipeline built with
LangChain (LCEL chains, DirectoryLoader/RecursiveCharacterTextSplitter/
GoogleGenerativeAIEmbeddings/InMemoryVectorStore/ChatGoogleGenerativeAI),
annotated with what each call replaces and its hidden costs (double retrieval
for sources, lost default observability, 34 extra dependencies). This exists
per the Framework Rule's "unless explicitly requested" exception, purely for
comparison — it is not a project direction change.

### Known issues / open gaps

- The Gemini free-tier quota that blocked live testing earlier has been
  confirmed reset (a cheap `embed_query()` call succeeded). However,
  **`compare_chunking.py`, `compare_retrieval.py`, and `compare_reranking.py`
  have still never been run end-to-end against the real corpus with live
  embeddings** — BM25 and the cross-encoder reranker are verified offline
  only (e.g. a standalone rerank test correctly scored the SSTF-relevant
  chunk +9.47 vs -4.05/-8.69 for wrong chunks). Running these three scripts
  for real numbers is explicitly deferred by the user ("will run this later").
- No automated test suite exists yet. `requirements.txt` lists `pytest` but
  no test files exist — writing V1-V5's per-stage tests (see Testing
  Philosophy below) remains an open gap, repeatedly deferred in favor of
  moving forward on the version ladder.

### Next steps to continue from

1. Run `compare_chunking.py`, `compare_retrieval.py`, `compare_reranking.py`
   for real against the live API now that quota is confirmed working, to get
   actual V3/V4/V5 numbers (currently only structurally/offline verified).
2. Decide V7's starting point (persistence vs. API layer) and implement it.
3. Write the pytest suite for V1-V5 stages (long-deferred).

### Commands

    pip install -r requirements.txt     # setup (or install into .venv/ if using a venv)
    # .env holds GEMINI_API_KEY (embedding/generation models, chunk size/
    # overlap, top_k, and RAG_DEBUG are also configured there)
    python main.py documents/           # index a directory once, then ask
                                         # questions interactively against it
    python run_eval.py documents/       # V2: retrieval-only evaluation against
                                         # eval/qa_dataset.py -- Recall@K,
                                         # Precision@K, MRR, nDCG@K
    python compare_chunking.py documents/  # V3: same eval, once per chunking
                                         # strategy (fixed/paragraph/recursive)
    python compare_retrieval.py documents/ # V4: vector-only vs hybrid retrieval
    python compare_reranking.py documents/ # V5: plain vs reranked retrieval

See `eval/METRICS.md` for what Recall@K/Precision@K/MRR/nDCG@K actually mean.

Model names in `.env`/`rag/config.py` (`GEMINI_EMBEDDING_MODEL`,
`GEMINI_GENERATION_MODEL`) need periodic updating as Google deprecates
models — if you hit a 404 "model not found" or "no longer available to new
users", list current options with
`client.models.list()` (filter by `supported_actions` containing
`embedContent` or `generateContent`) rather than guessing a new name.

No test suite exists yet. Once test files are added, run them with `pytest`
(or `pytest path/to/test_file.py::test_name` for a single test).

## Reference Material

`rag_architecture.html` is a general "how big companies do this" reference —
useful for vocabulary and for seeing what a fully scaled system eventually
looks like. It is NOT a template to build toward directly: it names tools
(LangChain, Kubernetes, Pinecone, Kafka, etc.) and a fully-productionized
architecture that directly conflict with the Framework Rule and Architecture
Rule below at the early versions (V1–V6). Only reach for concepts in that file
once the version ladder actually gets there, and only after justifying them
via the Architecture Rule questions — never adopt something from it just
because it's listed there.

## Objective

I am learning RAG by progressively building one RAG system from a basic implementation into an industry-grade production system.

The priority is NOT simply getting the application working.

The priority is understanding:

- why every component exists
- what happens internally
- what data flows between components
- what tradeoffs each design introduces
- how the component evolves in production systems

I want to understand the engineering underneath RAG frameworks.

---

# Core Learning Philosophy

DO NOT hide important concepts behind frameworks.

Before using a library abstraction, explain what it is doing conceptually.

For example, don't simply use:

    retriever.invoke(query)

Explain that retrieval involves:

    query
    → query embedding
    → similarity search
    → ranking
    → Top-K chunks

Then implement it.

The same principle applies to:

- embeddings
- vector databases
- chunking
- reranking
- hybrid search
- query rewriting
- caching
- evaluation
- observability
- authorization
- agents

---

# Development Strategy

We will evolve the SAME project rather than repeatedly creating toy projects.

## V1 — Basic RAG

Build:

    Documents
       ↓
    Loading
       ↓
    Preprocessing
       ↓
    Chunking
       ↓
    Embeddings
       ↓
    Vector Index
       ↓
    Query Embedding
       ↓
    Similarity Search
       ↓
    Top-K Chunks
       ↓
    Prompt Construction
       ↓
    LLM
       ↓
    Answer + Sources

V1 should be intentionally simple.

Do NOT add production complexity prematurely.

---

## V2 — Retrieval Fundamentals + Evaluation

Learn and implement:

- cosine similarity
- dot product
- distance metrics
- Top-K
- chunk-size experiments
- chunk-overlap experiments
- Recall@K
- Precision@K
- MRR
- nDCG
- ground-truth evaluation dataset

---

## V3 — Better Chunking + Metadata

Explore:

- paragraph chunking
- sentence chunking
- recursive chunking
- semantic chunking
- metadata
- document hierarchy
- source tracking
- document versions

---

## V4 — Hybrid Search

Add:

- keyword search
- BM25
- semantic search
- hybrid retrieval
- score fusion
- metadata filtering

Understand WHY hybrid search can outperform pure vector search.

---

## V5 — Reranking

Add:

    Initial retrieval
        ↓
    Top-N candidates
        ↓
    Reranker
        ↓
    Top-K final context

Understand:

- bi-encoder retrieval
- cross-encoder reranking
- latency/quality tradeoffs
- candidate count
- final context size

---

## V6 — Query Transformation

Explore:

- query rewriting
- query expansion
- multi-query retrieval
- HyDE
- decomposition
- routing

Do not add these just because they are popular.

Measure whether they actually improve retrieval.

---

## V7 — Production RAG

Gradually introduce:

- API architecture
- async processing
- background ingestion
- document updates/deletions
- caching
- database separation
- queues
- failure handling
- retries
- rate limiting
- model abstractions
- configuration management

---

## V8 — Security

Implement concepts such as:

- authentication
- authorization
- document-level permissions
- metadata-based access filtering
- tenant isolation
- audit logging
- PII considerations
- secure document ingestion

Never assume that retrieving a document means the current user is allowed to see it.

---

## V9 — Evaluation + Observability

Implement:

- retrieval evaluation
- answer evaluation
- faithfulness
- citation accuracy
- relevance
- latency measurement
- token usage
- cost tracking
- structured logging
- tracing
- failure analysis
- regression tests

Every major improvement should ideally be measurable.

---

## V10 — Scaling

Explore:

- Redis/cache layers
- vector database scaling
- indexing strategies
- asynchronous workers
- message queues
- horizontal scaling
- batching
- rate limiting
- model routing
- cost optimization

Do not introduce infrastructure unless we understand the problem it solves.

---

## V11 — Agentic RAG

Only after understanding conventional RAG thoroughly, explore:

- tool calling
- agents
- workflow orchestration
- LangGraph
- multi-step retrieval
- SQL + RAG
- API + RAG
- agent routing
- planning
- state management

---

# Framework Rule

Do NOT use LangChain, LangGraph, LlamaIndex, or similar frameworks during the early learning stages unless explicitly requested.

The goal is to understand the primitives first.

Frameworks can be introduced later to understand how they package these primitives.

---

# Architecture Rule

Do NOT assume that there is one "correct" production RAG architecture.

For every architectural decision, discuss:

1. What problem does it solve?
2. What is the simplest solution?
3. When does the simple solution stop being sufficient?
4. What is the production alternative?
5. What are the costs?
6. What are the latency implications?
7. What are the failure modes?
8. What scale justifies the added complexity?

Avoid cargo-cult architecture.

For example:

Do not add Kubernetes merely because it appears in a production architecture diagram.

---

# Code Quality

Write production-quality code progressively.

Early versions should prioritize:

- readability
- explicit data flow
- modularity
- testability
- observability
- understandable abstractions

Avoid unnecessary abstraction.

Do not create classes/interfaces solely to make the project look "enterprise."

---

# Debugging Requirement

The RAG pipeline should be inspectable.

When debugging is enabled, expose:

    Query

    Query embedding dimensions

    Retrieved chunks

    Similarity scores

    Metadata

    Reranker scores (when implemented)

    Final context

    Prompt

    Model

    Token usage

    Latency

    Final answer

    Sources

I should be able to understand WHY the system produced an answer.

---

# Testing Philosophy

Do not only test whether the application runs.

Test the individual RAG stages.

Examples:

- document loading
- chunking
- metadata preservation
- embedding generation
- retrieval
- ranking
- prompt construction
- answer generation
- citation/source mapping

Create known questions with known relevant documents.

Also create questions whose answers are NOT present.

---

# Important Principle

RAG quality is not determined only by the LLM.

Think in terms of:

    Data quality
         ↓
    Chunking quality
         ↓
    Embedding quality
         ↓
    Retrieval quality
         ↓
    Reranking quality
         ↓
    Context quality
         ↓
    Prompt quality
         ↓
    LLM quality
         ↓
    Final answer quality

When something goes wrong, investigate the pipeline rather than immediately blaming the LLM.

---

# Teaching Format

When implementing a new major component:

1. Explain the problem.
2. Explain the underlying concept.
3. Show the data flow.
4. Explain the simplest implementation.
5. Implement it.
6. Test it.
7. Inspect/debug it.
8. Explain its limitations.
9. Explain how production systems improve it.
10. Only then move to the next component.

Do not dump the entire project at once.

Teach and build incrementally.

---

# Current Project Rule

At any point, maintain a clear distinction between:

- what we have already implemented
- what we are implementing now
- what we will implement later
- why we are deliberately NOT implementing certain features yet

Never silently jump ahead to future levels.

---

# Ultimate Goal

By the end of this project, I should be able to look at a production RAG architecture and explain:

- every major component
- why it exists
- how information flows through it
- its alternatives
- its tradeoffs
- its failure modes
- how to evaluate it
- how to scale it
- how to secure it
- how to debug it

I should understand RAG as a software engineering system, not just as an LLM application.
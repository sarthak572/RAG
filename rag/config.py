"""Central place for API keys, model names, and tunables.

Everything here is read once at import time from environment variables
(loaded from .env). No other module should call os.environ directly.
"""

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Model used to turn text into vectors (indexing AND querying must use the
# same embedding model, otherwise the vectors live in different spaces and
# similarity scores are meaningless).
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# Model used to generate the final answer from retrieved chunks.
GENERATION_MODEL = os.environ.get("GEMINI_GENERATION_MODEL", "gemini-flash-lite-latest")

# Chunking parameters (see rag/chunker.py for why these two exist together).
CHUNK_SIZE_CHARS = int(os.environ.get("CHUNK_SIZE_CHARS", "1000"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", "150"))

# Which chunking strategy to use by default: "fixed", "paragraph", or
# "recursive" (see rag/chunker.py). compare_chunking.py overrides this
# per-run to evaluate all three against the same corpus.
CHUNK_STRATEGY = os.environ.get("CHUNK_STRATEGY", "fixed")

# Which retrieval mode to use: "vector" (semantic only, V1) or "hybrid"
# (vector + BM25 keyword search, fused via RRF -- V4). See
# rag/hybrid_search.py for why hybrid can catch what vector search misses.
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "vector")

# Whether to rerank retrieved candidates with a cross-encoder before
# building the prompt (V5). See rag/reranker.py.
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "false").lower() == "true"

# How many candidates to retrieve BEFORE reranking (N). Only used when
# RERANK_ENABLED is true -- should be noticeably larger than TOP_K, since
# the point of reranking is picking a better final K out of a wider N.
RERANK_CANDIDATE_COUNT = int(os.environ.get("RERANK_CANDIDATE_COUNT", "20"))

# How many chunks to retrieve per query.
TOP_K = int(os.environ.get("TOP_K", "5"))

# When true, pipeline.py prints every intermediate value (embeddings dims,
# similarity scores, final prompt, token usage, latency, ...) so the system
# stays inspectable rather than a black box.
DEBUG = os.environ.get("RAG_DEBUG", "false").lower() == "true"

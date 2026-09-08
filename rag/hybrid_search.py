"""Stage 4c (V4): Hybrid Search -- Score Fusion.

Combines vector search (rag/vector_store.py, meaning-based) and BM25
(rag/bm25.py, exact-term-based) into one ranked result, so each covers the
other's blind spot: BM25 rescues an exact acronym/term that gets diluted
in a generic embedding; vector search rescues a paraphrase with no shared
vocabulary that BM25 would find nothing for. See CLAUDE.md's V4 notes --
the goal is to MEASURE this against the eval harness, not assume it helps.

Fusion uses Reciprocal Rank Fusion (RRF), not a weighted sum of raw
scores, because cosine similarity (-1..1) and BM25 scores (unbounded,
corpus-dependent) live on incomparable scales -- there's no principled way
to add them directly. RRF sidesteps that entirely by using each chunk's
RANK POSITION in each list instead of its raw score.
"""

from rag.bm25 import BM25Index
from rag.chunker import Chunk
from rag.embedder import embed_query
from rag.vector_store import ScoredChunk, VectorStore

_RRF_K = 60  # smoothing constant from the original RRF paper -- dampens
# how much a rank-1 hit dominates over a rank-2 hit; not sensitive to
# small changes, 60 is the standard default rather than a tuned value.


def _chunk_key(chunk: Chunk) -> tuple:
    """Identity for a chunk across independently-produced ranked lists --
    (source, chunk_index) uniquely identifies a chunk within one index."""
    return (chunk.source, chunk.chunk_index)


def reciprocal_rank_fusion(ranked_lists: list[list[ScoredChunk]],
                            k: int = _RRF_K) -> list[ScoredChunk]:
    """Merge multiple independently-ranked ScoredChunk lists into one.

    fused_score(chunk) = sum, over every list containing it, of 1/(k + rank)

    A chunk that appears near the top of EITHER list gets a large
    contribution; a chunk appearing in BOTH lists (even at middling ranks)
    can outscore a chunk that's rank-1 in only one list -- rewarding
    agreement between the two signals, not just a single strong opinion.
    The returned ScoredChunk.score is this fused value, not a similarity
    or a BM25 score -- it's only meaningful for re-ranking within this
    fused list, not comparable across calls.
    """
    fused_scores: dict[tuple, float] = {}
    chunk_by_key: dict[tuple, Chunk] = {}

    for ranked in ranked_lists:
        for rank, scored in enumerate(ranked, start=1):
            key = _chunk_key(scored.chunk)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            chunk_by_key[key] = scored.chunk

    ordered_keys = sorted(fused_scores, key=lambda key: -fused_scores[key])
    return [ScoredChunk(chunk=chunk_by_key[key], score=fused_scores[key]) for key in ordered_keys]


def hybrid_search(query: str, vector_store: VectorStore, bm25_index: BM25Index,
                   top_k: int, candidate_k: int | None = None) -> list[ScoredChunk]:
    """Run vector search and BM25 independently over the same corpus, fuse
    via RRF, return the top_k fused results.

    candidate_k (how many each individual method retrieves before fusion)
    defaults to a multiple of top_k -- wider than what you'll actually use,
    so RRF has enough overlap between the two lists to do meaningful
    fusion instead of just concatenating two short, disjoint lists.
    """
    candidate_k = candidate_k or max(top_k * 4, 20)

    query_vector = embed_query(query)
    vector_results = vector_store.search(query_vector, top_k=candidate_k)
    keyword_results = bm25_index.search(query, top_k=candidate_k)

    fused = reciprocal_rank_fusion([vector_results, keyword_results])
    return fused[:top_k]

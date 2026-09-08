"""Stage 5 (V5): Reranking.

The second stage of retrieval: rag/vector_store.py and rag/bm25.py (via
rag/hybrid_search.py) are BI-ENCODER methods -- query and chunk are scored
independently, cheaply, which is what lets them scan the whole corpus.
Reranking uses a CROSS-ENCODER instead: query and ONE candidate chunk are
fed into the model TOGETHER, so every query token can attend to every
chunk token in the same forward pass. This catches relevance signals a
bi-encoder structurally cannot (see CLAUDE.md's V5 notes for the concrete
"starvation" example) -- at the cost of not being precomputable, so it can
only run on a small candidate set (N), not the whole corpus.

Runs entirely locally via sentence-transformers -- no Gemini API call, no
effect on that quota. This is a real, if small (~80MB), download and a
real dependency (sentence-transformers + torch), not a config toggle.
"""

from sentence_transformers import CrossEncoder

from rag.vector_store import ScoredChunk

# ms-marco-MiniLM-L-6-v2: a small, fast cross-encoder trained specifically
# on human relevance judgments (MS MARCO passage ranking) -- i.e. trained
# on "does this passage answer this query," which is a narrower and more
# specific objective than the general-purpose semantic similarity our
# embedding model was trained for. That's WHY it can reorder results an
# embedding-only comparison can't, not just because it's slower.
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None  # lazy-loaded -- downloads/loads once per process


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    """Re-score `candidates` against `query` with a cross-encoder, return
    the top_k by the NEW score.

    Cost scales with len(candidates), not corpus size -- this only ever
    runs on the N chunks retrieval already narrowed things down to. The
    returned ScoredChunk.score is the cross-encoder's raw relevance logit,
    not a cosine similarity or a BM25 score -- comparable only to other
    scores from this same rerank() call, not across it.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, sc.chunk.text) for sc in candidates]
    scores = model.predict(pairs)

    reranked = sorted(zip(candidates, scores), key=lambda pair: -pair[1])
    return [
        ScoredChunk(chunk=sc.chunk, score=float(score))
        for sc, score in reranked[:top_k]
    ]

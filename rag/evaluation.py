"""Stage 8 (V2): Retrieval Evaluation.

Turns "does the answer look right?" into actual numbers. Every metric here
is computed against the RETRIEVAL step only -- which source files ended up
in the top-K -- not against the LLM's final wording. That split is
deliberate: retrieval quality and generation quality are different failure
modes (see the Important Principle in CLAUDE.md: a wrong answer might mean
the right chunk was never retrieved, OR it was retrieved but the LLM still
got it wrong). These metrics isolate the first failure mode.

Ground truth here is FILE-level, not chunk-level: a question is tagged with
which source file(s) SHOULD show up in the results, not which exact chunk.
This is deliberate too -- file-level ground truth stays valid even when
CHUNK_SIZE_CHARS/CHUNK_OVERLAP_CHARS change (chunk boundaries move, files
don't), which matters because V2 also wants to run chunk-size experiments
against this same dataset.

All four metrics take the same two inputs:
    retrieved: list[str]  -- source filenames, ranked best-to-worst
    relevant:  set[str]   -- source filenames that SHOULD have been found
"""

from __future__ import annotations

import math
from typing import Callable

from rag.vector_store import ScoredChunk

# Rate limiting for embed_query calls is handled inside rag/embedder.py's
# shared _throttle(), not here -- it needs to see embed_chunks calls too.

# A retrieval function: (question, k) -> ranked ScoredChunk list. This is
# how evaluate_retrieval() stays agnostic to WHICH retrieval mode is being
# tested -- plain vector search, hybrid (V4), or reranked (V5) all fit this
# same shape, so one evaluation loop works for all of them.
RetrieveFn = Callable[[str, int], list[ScoredChunk]]


def ranked_sources(scored_chunks: list[ScoredChunk]) -> list[str]:
    """Collapse ranked chunks into a ranked, de-duplicated list of sources.

    Several chunks from the same file collapse into one entry, placed at
    the position of that file's first (best-scoring) chunk.
    """
    seen: list[str] = []
    for sc in scored_chunks:
        if sc.chunk.source not in seen:
            seen.append(sc.chunk.source)
    return seen


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """Of everything that SHOULD be found, what fraction was found in top-k?

    Returns None (not applicable) for questions with no relevant source at
    all -- e.g. a question whose answer isn't in the corpus.
    """
    if not relevant:
        return None
    hits = set(retrieved[:k]) & relevant
    return len(hits) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of the k sources retrieved, what fraction were actually relevant?

    Always defined, including when `relevant` is empty (correct precision
    is then 0.0: nothing retrieved could have been relevant).
    """
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for source in top_k if source in relevant)
    return hits / len(top_k)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float | None:
    """1 / (rank of the first relevant source), 1-indexed; 0.0 if never found.

    None when there is no relevant source to find.
    """
    if not relevant:
        return None
    for i, source in enumerate(retrieved, start=1):
        if source in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """Normalized Discounted Cumulative Gain @ k.

    Like recall, but rewards relevant sources appearing EARLIER (via a
    log2 rank discount) and accounts for questions with multiple relevant
    sources. None when there is no relevant source to find.
    """
    if not relevant:
        return None

    def dcg(sources: list[str]) -> float:
        return sum(
            1.0 / math.log2(i + 1)
            for i, source in enumerate(sources[:k], start=1)
            if source in relevant
        )

    ideal = dcg(list(relevant))  # best possible arrangement: all relevant first
    actual = dcg(retrieved)
    return actual / ideal if ideal > 0 else 0.0


def evaluate_retrieval(retrieve_fn: RetrieveFn, questions: list[dict], k: int) -> list[dict]:
    """Run every question in `questions` through `retrieve_fn` (retrieval
    only, no LLM call) and return one result dict per question.

    `retrieve_fn(question, k) -> list[ScoredChunk]` -- pass a closure over
    whatever retrieval mode you're testing (plain vector search, hybrid,
    reranked...). Each question dict must have "question" (str) and
    "relevant_sources" (list[str]), matching eval/qa_dataset.py's shape.
    """
    results = []
    for item in questions:
        relevant = set(item["relevant_sources"])
        scored = retrieve_fn(item["question"], k)
        retrieved = ranked_sources(scored)

        results.append({
            "question": item["question"],
            "relevant": relevant,
            "retrieved": retrieved,
            "recall": recall_at_k(retrieved, relevant, k),
            "precision": precision_at_k(retrieved, relevant, k),
            "rr": reciprocal_rank(retrieved, relevant),
            "ndcg": ndcg_at_k(retrieved, relevant, k),
        })
    return results


def aggregate(results: list[dict]) -> dict[str, float]:
    """Average each metric across results, skipping None (not-applicable) values."""

    def avg(key: str) -> float:
        values = [r[key] for r in results if r[key] is not None]
        return sum(values) / len(values) if values else float("nan")

    return {
        "recall": avg("recall"),
        "precision": avg("precision"),
        "mrr": avg("rr"),
        "ndcg": avg("ndcg"),
    }

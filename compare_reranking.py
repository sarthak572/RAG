"""V5: Reranking comparison (with vs without).

Builds one index from documents/, then evaluates the same 24 ground-truth
questions (eval/qa_dataset.py) two ways: plain retrieval (top_k=TOP_K
directly), and retrieve-then-rerank (retrieve RERANK_CANDIDATE_COUNT
candidates via the current RETRIEVAL_MODE, then narrow to TOP_K with the
local cross-encoder in rag/reranker.py) -- printing Recall@K, Precision@K,
MRR, and nDCG@K side by side.

The reranking step itself makes no Gemini API calls (it's a local
sentence-transformers model) -- only the underlying retrieval step
(vector/hybrid) does.

Usage:
    python compare_reranking.py documents/
"""

import sys

from eval.qa_dataset import QUESTIONS
from rag import config
from rag.bm25 import BM25Index
from rag.chunker import chunk_documents
from rag.embedder import embed_chunks, embed_query
from rag.evaluation import aggregate, evaluate_retrieval
from rag.hybrid_search import hybrid_search
from rag.loader import load_documents
from rag.reranker import rerank
from rag.vector_store import VectorStore


def build_index(directory: str) -> tuple[VectorStore, BM25Index, int]:
    documents = load_documents(directory)
    chunks = chunk_documents(documents)
    vectors = embed_chunks([c.text for c in chunks])

    store = VectorStore()
    store.add(chunks, vectors)

    bm25 = BM25Index()
    bm25.add(chunks)

    return store, bm25, len(chunks)


def _retrieve(store: VectorStore, bm25: BM25Index, query: str, top_k: int):
    if config.RETRIEVAL_MODE == "hybrid":
        return hybrid_search(query, store, bm25, top_k=top_k)
    return store.search(embed_query(query), top_k=top_k)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    store, bm25, n_chunks = build_index(directory)
    k = config.TOP_K
    candidate_k = config.RERANK_CANDIDATE_COUNT
    print(f"Indexed {n_chunks} chunks from {directory} (retrieval_mode={config.RETRIEVAL_MODE!r})\n")

    modes = {
        "plain": lambda q, top_k: _retrieve(store, bm25, q, top_k),
        "reranked": lambda q, top_k: rerank(q, _retrieve(store, bm25, q, candidate_k), top_k),
    }

    rows = []
    for name, retrieve_fn in modes.items():
        results = evaluate_retrieval(retrieve_fn, QUESTIONS, k)
        metrics = aggregate(results)
        rows.append((name, metrics))
        print(f"[{name}] evaluated {len(QUESTIONS)} questions")

    print()
    header = f"{'mode':<10} {'recall@' + str(k):>10} {'prec@' + str(k):>9} {'mrr':>6} {'ndcg@' + str(k):>9}"
    print(header)
    print("-" * len(header))
    for name, metrics in rows:
        print(
            f"{name:<10} {metrics['recall']:>10.3f} {metrics['precision']:>9.3f} "
            f"{metrics['mrr']:>6.3f} {metrics['ndcg']:>9.3f}"
        )


if __name__ == "__main__":
    main()

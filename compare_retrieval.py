"""V4: Retrieval mode comparison (vector-only vs hybrid).

Builds ONE index (embeddings + BM25) from documents/, then evaluates the
SAME 24 ground-truth questions (eval/qa_dataset.py) two ways: pure vector
search, and hybrid (vector + BM25 fused via RRF) -- printing Recall@K,
Precision@K, MRR, and nDCG@K side by side.

Unlike compare_chunking.py, this only needs to build the index ONCE --
retrieval mode is a property of how a QUERY is searched, not how the
corpus was chunked/embedded, so both modes share the same VectorStore and
BM25Index.

Usage:
    python compare_retrieval.py documents/
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
from rag.vector_store import VectorStore


def build_index(directory: str) -> tuple[VectorStore, BM25Index, int]:
    documents = load_documents(directory)
    chunks = chunk_documents(documents)
    vectors = embed_chunks([c.text for c in chunks])

    store = VectorStore()
    store.add(chunks, vectors)

    bm25 = BM25Index()
    bm25.add(chunks)  # no API cost -- BM25 only needs chunk text

    return store, bm25, len(chunks)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    store, bm25, n_chunks = build_index(directory)
    k = config.TOP_K
    print(f"Indexed {n_chunks} chunks from {directory}\n")

    modes = {
        "vector": lambda q, top_k: store.search(embed_query(q), top_k=top_k),
        "hybrid": lambda q, top_k: hybrid_search(q, store, bm25, top_k=top_k),
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

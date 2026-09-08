"""V3: Chunking strategy comparison.

Builds one index per chunking strategy (fixed / paragraph / recursive)
from the SAME documents/ corpus, evaluates each against the SAME
ground-truth questions (eval/qa_dataset.py), and prints Recall@K,
Precision@K, MRR, and nDCG@K side by side -- plus chunk count and average
chunk length, since two strategies can tie on retrieval quality while
differing meaningfully in indexing cost (more/fewer chunks = more/fewer
embedding calls and more/less context stuffed into the LLM's prompt).

This works because eval/qa_dataset.py's ground truth is file-level, not
chunk-level -- it doesn't care how a file got cut into chunks, only which
file should come back. See rag/evaluation.py's docstring.

Usage:
    python compare_chunking.py documents/
"""

import sys
import time

from eval.qa_dataset import QUESTIONS
from rag import config
from rag.chunker import Chunk, chunk_documents
from rag.embedder import embed_chunks, embed_query
from rag.evaluation import aggregate, evaluate_retrieval
from rag.loader import load_documents
from rag.vector_store import VectorStore

STRATEGIES = ["fixed", "paragraph", "recursive"]


def build_index(documents, strategy: str) -> tuple[VectorStore, list[Chunk]]:
    chunks = chunk_documents(documents, strategy=strategy)
    vectors = embed_chunks([c.text for c in chunks])
    store = VectorStore()
    store.add(chunks, vectors)
    return store, chunks


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    documents = load_documents(directory)
    k = config.TOP_K

    # One-time startup grace period, not a per-strategy cooldown: protects
    # against quota already used by a DIFFERENT recent process (e.g. a
    # script that crashed a minute ago) that this process's in-memory
    # throttle has no way to see. Once running, the shared throttle in
    # rag/embedder.py correctly paces every call within this process,
    # across all three strategies, on its own.
    STARTUP_GRACE_SECONDS = 20
    print(f"(startup grace: {STARTUP_GRACE_SECONDS}s, in case a previous run left residual quota usage)")
    time.sleep(STARTUP_GRACE_SECONDS)

    rows = []
    for strategy in STRATEGIES:
        store, chunks = build_index(documents, strategy)
        results = evaluate_retrieval(
            lambda question, top_k: store.search(embed_query(question), top_k=top_k),
            QUESTIONS, k,
        )
        metrics = aggregate(results)
        avg_len = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0.0
        rows.append((strategy, len(chunks), avg_len, metrics))
        print(f"[{strategy}] indexed {len(chunks)} chunks (avg {avg_len:.0f} chars/chunk)")

    print()
    header = (
        f"{'strategy':<12} {'n_chunks':>9} {'avg_len':>8} "
        f"{'recall@' + str(k):>10} {'prec@' + str(k):>9} {'mrr':>6} {'ndcg@' + str(k):>9}"
    )
    print(header)
    print("-" * len(header))
    for strategy, n_chunks, avg_len, metrics in rows:
        print(
            f"{strategy:<12} {n_chunks:>9} {avg_len:>8.0f} "
            f"{metrics['recall']:>10.3f} {metrics['precision']:>9.3f} "
            f"{metrics['mrr']:>6.3f} {metrics['ndcg']:>9.3f}"
        )


if __name__ == "__main__":
    main()

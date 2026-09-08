"""V2: Retrieval evaluation.

Indexes documents/ once (using the default chunking strategy from
rag/config.py), then answers every question in eval/qa_dataset.py via
RETRIEVAL ONLY -- no LLM generation call -- and reports Recall@K,
Precision@K, MRR, and nDCG@K, per-question and averaged.

For comparing chunking strategies against each other, use
compare_chunking.py instead -- this script only evaluates whatever
CHUNK_STRATEGY is currently set to.

Usage:
    python run_eval.py documents/
"""

import sys

from eval.qa_dataset import QUESTIONS
from rag import config
from rag.chunker import chunk_documents
from rag.embedder import embed_chunks, embed_query
from rag.evaluation import aggregate, evaluate_retrieval
from rag.loader import load_documents
from rag.vector_store import VectorStore


def build_index(directory: str) -> VectorStore:
    documents = load_documents(directory)
    chunks = chunk_documents(documents)
    vectors = embed_chunks([c.text for c in chunks])
    store = VectorStore()
    store.add(chunks, vectors)
    return store


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    store = build_index(directory)
    print(f"Indexed {len(store)} chunks from {directory} (strategy={config.CHUNK_STRATEGY!r})")
    print(f"Evaluating {len(QUESTIONS)} questions at top_k={config.TOP_K}\n")

    k = config.TOP_K
    results = evaluate_retrieval(
        lambda question, top_k: store.search(embed_query(question), top_k=top_k),
        QUESTIONS, k,
    )

    for r in results:
        print(f"Q: {r['question']}")
        print(f"   expected:  {sorted(r['relevant']) or '(none -- not in corpus)'}")
        print(f"   retrieved: {r['retrieved']}")
        print(
            f"   recall@{k}={r['recall']}  precision@{k}={r['precision']:.2f}  "
            f"RR={r['rr']}  nDCG@{k}={r['ndcg']}"
        )
        print()

    metrics = aggregate(results)
    n_answerable = sum(1 for r in results if r["recall"] is not None)
    print("=== Aggregate ===")
    print(f"Recall@{k}:    {metrics['recall']:.3f}  (n={n_answerable} answerable questions)")
    print(f"Precision@{k}: {metrics['precision']:.3f}  (n={len(results)} questions, incl. not-in-corpus)")
    print(f"MRR:          {metrics['mrr']:.3f}  (n={n_answerable} answerable questions)")
    print(f"nDCG@{k}:      {metrics['ndcg']:.3f}  (n={n_answerable} answerable questions)")


if __name__ == "__main__":
    main()

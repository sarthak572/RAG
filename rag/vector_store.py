"""Stage 4: Vector Index.

The simplest possible "vector database": keep every chunk's vector in
memory and, on search, compute cosine similarity against all of them,
then return the highest-scoring ones.

This is brute-force exact search: O(n) per query, 100% recall. Real
vector DBs (Pinecone, Qdrant, Milvus) replace this scan with an
approximate index (HNSW, IVF) that trades a small amount of recall for
much better speed at millions/billions of vectors. At V1 scale (a
handful of documents) brute force is both simpler AND fast enough --
there is no reason yet to reach for an approximation.
"""

from dataclasses import dataclass

import numpy as np

from rag.chunker import Chunk
from rag.embedder import Vector


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # cosine similarity, -1..1, higher = more similar


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None  # shape (n_chunks, dim)

    def add(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        """Add chunks and their embeddings to the index."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")

        new_vectors = np.array(vectors, dtype=np.float32)
        # normalize now so search is a plain dot product, not a full
        # cosine-similarity recomputation on every query
        new_vectors /= np.linalg.norm(new_vectors, axis=1, keepdims=True)

        self._chunks.extend(chunks)
        self._vectors = (
            new_vectors
            if self._vectors is None
            else np.vstack([self._vectors, new_vectors])
        )

    def search(self, query_vector: Vector, top_k: int) -> list[ScoredChunk]:
        """Return the top_k chunks most similar to query_vector.

        Data flow: query vector -> cosine similarity against every stored
        vector -> sort descending -> top_k (chunk, score) pairs.
        """
        if self._vectors is None or len(self._chunks) == 0:
            return []

        query = np.array(query_vector, dtype=np.float32)
        query /= np.linalg.norm(query)

        # vectors are pre-normalized, so this dot product IS cosine similarity
        scores = self._vectors @ query

        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(-scores)[:k]

        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._chunks)

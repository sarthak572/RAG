"""Stage 7: Pipeline.

Wires stages 1-6 together into the two operations a RAG system actually
exposes:

  index(directory)  -- offline: Documents -> Loading -> Chunking ->
                        Embeddings -> Vector Index
  ask(query)        -- online:  Query -> Query Embedding -> Similarity
                        Search -> Top-K Chunks -> Prompt -> LLM -> Answer

When config.DEBUG is on, every intermediate value is printed: query
embedding dims, retrieved chunks + similarity scores, the final prompt,
model, token usage, latency, answer, and sources. The goal is to always
be able to explain WHY the system produced a given answer, not just that
it produced one.
"""

from dataclasses import dataclass

from rag import config
from rag.bm25 import BM25Index
from rag.chunker import chunk_documents
from rag.embedder import embed_chunks, embed_query
from rag.generator import generate_answer
from rag.hybrid_search import hybrid_search
from rag.loader import load_documents
from rag.prompt import build_prompt
from rag.reranker import rerank
from rag.vector_store import ScoredChunk, VectorStore


@dataclass
class AnswerResult:
    answer: str
    sources: list[str]
    scored_chunks: list[ScoredChunk]


class Pipeline:
    def __init__(self) -> None:
        self.store = VectorStore()
        self.bm25 = BM25Index()

    def index(self, directory: str) -> int:
        """Load, chunk, and embed every document in `directory`.

        Returns the number of chunks indexed.
        """
        documents = load_documents(directory)
        chunks = chunk_documents(documents)
        if not chunks:
            raise ValueError(f"No indexable text found in {directory}")

        vectors = embed_chunks([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self.bm25.add(chunks)  # no API cost -- BM25 only needs chunk text

        if config.DEBUG:
            print(f"[index] {len(documents)} documents -> {len(chunks)} chunks "
                  f"-> {len(vectors)} vectors (dim={len(vectors[0])})")

        return len(chunks)

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Retrieve the top_k chunks for `query`, using whichever
        RETRIEVAL_MODE is configured ("vector" or "hybrid")."""
        if config.RETRIEVAL_MODE == "hybrid":
            return hybrid_search(query, self.store, self.bm25, top_k=top_k)
        query_vector = embed_query(query)
        return self.store.search(query_vector, top_k=top_k)

    def ask(self, query: str) -> AnswerResult:
        """Answer a query using the indexed documents."""
        retrieval_k = config.RERANK_CANDIDATE_COUNT if config.RERANK_ENABLED else config.TOP_K
        candidates = self.retrieve(query, top_k=retrieval_k)

        if config.RERANK_ENABLED:
            scored_chunks = rerank(query, candidates, top_k=config.TOP_K)
        else:
            scored_chunks = candidates

        prompt = build_prompt(query, scored_chunks)
        generation = generate_answer(prompt)

        sources = sorted({sc.chunk.source for sc in scored_chunks})

        if config.DEBUG:
            print(f"[ask] query: {query!r}")
            print(f"[ask] retrieval_mode: {config.RETRIEVAL_MODE!r}  "
                  f"rerank_enabled: {config.RERANK_ENABLED}")
            if config.RERANK_ENABLED:
                print(f"[ask] retrieved {len(candidates)} candidates, reranked to "
                      f"{len(scored_chunks)}:")
            else:
                print(f"[ask] retrieved {len(scored_chunks)} chunks:")
            for sc in scored_chunks:
                print(f"    score={sc.score:.4f} source={sc.chunk.source} "
                      f"chunk_index={sc.chunk.chunk_index}")
            print(f"[ask] prompt:\n{prompt}\n")
            print(f"[ask] model: {generation.model}")
            print(f"[ask] tokens: in={generation.input_tokens} "
                  f"out={generation.output_tokens}")
            print(f"[ask] latency: {generation.latency_seconds:.2f}s")
            print(f"[ask] answer: {generation.text}")
            print(f"[ask] sources: {sources}")

        return AnswerResult(answer=generation.text, sources=sources, scored_chunks=scored_chunks)

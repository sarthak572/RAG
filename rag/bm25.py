"""Stage 4b (V4): Keyword Search (BM25).

The complement to rag/embedder.py + rag/vector_store.py's semantic search:
where cosine similarity over embeddings captures MEANING (a paraphrase
with no shared words can still match), BM25 captures exact TERM overlap
(a rare acronym or specific name matches only if it's literally present,
with no risk of getting diluted by the surrounding topic's vocabulary).
See CLAUDE.md's V4 notes for the concrete example of why this matters:
an embedding of "What is SSTF in disk scheduling?" can end up nearly as
close to CPU-scheduling chunks as to the one chunk that actually defines
SSTF, because "scheduling" dominates the semantic signal. BM25 has no such
confusion -- it just checks which chunks contain the literal token "sstf".

This is implemented from scratch (no rank_bm25 or similar library) per the
Framework Rule -- BM25 is a well-defined, learnable algorithm, not
infrastructure. Needs zero API calls: it only ever looks at chunk text
you already have locally.

The algorithm, in one pass: BM25 scores a chunk against a query by summing,
for each query term present in the chunk, (term's rarity across the whole
corpus) x (a saturating function of how often the term appears in THIS
chunk, normalized against chunk length so long chunks don't win purely by
containing more words).
"""

import math
import re
from collections import Counter

from rag.chunker import Chunk
from rag.vector_store import ScoredChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric runs. Deliberately simple --
    no stemming, no stopword removal. Good enough to demonstrate exact-term
    matching; a production system would add both."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index over a fixed set of chunks.

    k1 controls how quickly extra occurrences of a term stop adding score
    (term-frequency saturation -- the 5th occurrence of a word matters much
    less than the 1st). b controls how much chunk length is penalized (0 =
    ignore length entirely, 1 = fully normalize by length). 1.5 / 0.75 are
    the standard defaults from the original BM25 literature.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_term_freqs: list[Counter] = []  # term -> count, per chunk
        self._doc_lengths: list[int] = []
        self._doc_freq: Counter = Counter()  # term -> number of chunks containing it
        self._avg_doc_length = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index, rebuilding corpus-wide statistics
        (document frequency, average length) that BM25's formula needs."""
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            term_freq = Counter(tokens)
            self._chunks.append(chunk)
            self._doc_term_freqs.append(term_freq)
            self._doc_lengths.append(len(tokens))
            for term in term_freq:
                self._doc_freq[term] += 1

        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )

    def _idf(self, term: str) -> float:
        """Inverse document frequency: rare terms across the corpus score
        higher than common ones. The +1 inside the log keeps this
        non-negative even for a term appearing in most of the corpus
        (the classic TF-IDF idf can go negative there; this is BM25's
        standard fix)."""
        n = len(self._chunks)
        df = self._doc_freq.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Score every chunk against the query's terms, return the top_k
        with score > 0 (a chunk sharing zero terms with the query has no
        keyword relevance at all -- excluding it is the point of keyword
        search, unlike vector search where everything gets SOME score)."""
        if not self._chunks:
            return []

        query_terms = _tokenize(query)
        scores = [0.0] * len(self._chunks)

        for i, term_freq in enumerate(self._doc_term_freqs):
            doc_length = self._doc_lengths[i]
            score = 0.0
            for term in query_terms:
                freq = term_freq.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf(term)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * doc_length / self._avg_doc_length
                )
                score += idf * numerator / denominator
            scores[i] = score

        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [
            ScoredChunk(chunk=self._chunks[i], score=scores[i])
            for i in ranked[:top_k]
            if scores[i] > 0
        ]

    def __len__(self) -> int:
        return len(self._chunks)

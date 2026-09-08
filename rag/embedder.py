"""Stage 3: Embeddings.

Converts text into a dense vector: a fixed-length list of floats such that
texts with similar meaning end up close together in vector space. This is
the property that makes "similarity search" (Stage 5) meaningful at all.

CRITICAL: indexing (chunks) and querying (user question) must use the same
embedding model. Different models produce incomparable vector spaces --
comparing them would be like measuring distance in miles against a map
scaled in kilometers.

Gemini's embedding API also takes a `task_type`, because the model is
trained to embed "a document being stored" and "a query being searched"
slightly differently, even though both go through the same model. We tag
each call correctly instead of ignoring the distinction.
"""

import collections
import time

from google import genai
from google.genai import types

from rag.config import EMBEDDING_MODEL, GEMINI_API_KEY

_client = genai.Client(api_key=GEMINI_API_KEY)

Vector = list[float]

# The free tier caps embedContent at 100 requests/minute, and each text in
# a batch counts toward that individually. The limit is per API key,
# shared across EVERY call this process makes -- embed_chunks (indexing)
# AND embed_query (evaluation) alike -- so both funnel through the same
# throttle below rather than each having its own local delay. An earlier
# version gave embed_chunks a fixed sleep between its OWN batches only,
# which still blew the quota the moment two different calls (e.g. the
# last batch of one chunking strategy and the first batch of the next in
# compare_chunking.py) landed in the same rolling 60s window.
_MAX_ITEMS_PER_MINUTE = 95  # the confirmed cap is 100/min; small margin only
_EMBED_BATCH_SIZE = 95
_recent_calls: collections.deque[tuple[float, int]] = collections.deque()


def _throttle(n_items: int) -> None:
    """Block until issuing a call for n_items more texts stays under the
    shared per-minute cap, based on actual recent call history.

    This only sees calls THIS process made -- it can't see quota used by a
    different process (e.g. a script that ran moments ago). Callers doing
    a fresh run shortly after a previous one should add their own short
    startup grace delay; see compare_chunking.py.
    """
    now = time.monotonic()
    while _recent_calls and _recent_calls[0][0] < now - 60:
        _recent_calls.popleft()
    used = sum(count for _, count in _recent_calls)
    while _recent_calls and used + n_items > _MAX_ITEMS_PER_MINUTE:
        wait = _recent_calls[0][0] + 61 - now
        if wait > 0:
            time.sleep(wait)
        now = time.monotonic()
        while _recent_calls and _recent_calls[0][0] < now - 60:
            _recent_calls.popleft()
        used = sum(count for _, count in _recent_calls)
    _recent_calls.append((time.monotonic(), n_items))


def embed_chunks(texts: list[str]) -> list[Vector]:
    """Embed chunk texts for storage in the vector index, batched and
    throttled to stay under the embedding API's free-tier rate limit.

    Data flow: list[str] -> Gemini embedding API (in batches) -> list[vector]
    """
    vectors: list[Vector] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        _throttle(len(batch))
        response = _client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        vectors.extend(embedding.values for embedding in response.embeddings)
    return vectors


def embed_query(text: str) -> Vector:
    """Embed a single user query for similarity search against the index.

    Uses task_type="RETRIEVAL_QUERY" -- the counterpart to
    RETRIEVAL_DOCUMENT used when embedding chunks.
    """
    _throttle(1)
    response = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values

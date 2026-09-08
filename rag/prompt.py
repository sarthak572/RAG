"""Stage 5: Prompt Construction.

Turns (query, retrieved chunks) into the actual text sent to the LLM.

The instruction to answer only from context, and to cite chunk numbers,
is the main defense against hallucination at this stage -- the model has
no way to check facts itself, so we constrain what it's allowed to draw
from and make its claims traceable back to a source chunk.
"""

from rag.vector_store import ScoredChunk

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below. If the answer is not contained in the "
    "context, say you don't know -- do not use outside knowledge. "
    "When you state a fact, cite it with the matching [N] chunk number."
)


def build_prompt(query: str, scored_chunks: list[ScoredChunk]) -> str:
    """Assemble the final prompt string sent to the LLM.

    Data flow: query + scored chunks -> numbered context block -> full prompt
    """
    context_block = "\n\n".join(
        f"[{i + 1}] (source: {sc.chunk.source}"
        f"{f', section: {sc.chunk.section_title}' if sc.chunk.section_title else ''})\n"
        f"{sc.chunk.text}"
        for i, sc in enumerate(scored_chunks)
    )

    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

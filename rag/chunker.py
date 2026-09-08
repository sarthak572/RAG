"""Stage 2: Chunking.

Splits each Document's text into smaller pieces, one embedding per Chunk.
V1 shipped a single strategy (fixed-size sliding window). V3 adds two more
so they can be measured against each other with the V2 eval harness instead
of picked by intuition (see CLAUDE.md's Architecture Rule).

Three strategies, in increasing order of structural awareness:

  fixed     -- cut every `chunk_size` characters, with `overlap` characters
               repeated between consecutive chunks. Simplest possible
               approach; can (and will) slice mid-sentence.
  paragraph -- split on blank lines, then greedily pack consecutive
               paragraphs up to `chunk_size` so a chunk never cuts a
               paragraph in half. A single paragraph longer than
               chunk_size is kept whole (no further splitting) -- simplest
               fix for the "cuts mid-sentence" problem, at the cost of
               variable, sometimes oversized chunk lengths.
  recursive -- prefer the largest structural unit first: split by
               heading-delimited sections, and only descend to paragraphs
               (then, as a last resort, a fixed-size cut) for any section
               still larger than chunk_size. This is the standard
               production approach (what LangChain's
               RecursiveCharacterTextSplitter does conceptually).

All three also extract a best-effort `section_title` per chunk, from
lines that look like a numbered heading (e.g. "6. Virtual Memory" or
OSTEP-style "26.1 Why Use Threads?"). This is genuinely best-effort: it's
a regex heuristic, not real document structure, so it will miss headings
in formats that don't follow "number(s) + short title on their own line"
(a real limitation, not an oversight -- proper structure extraction needs
format-aware parsing, e.g. reading DOCX heading styles or a PDF layout
parser, which is more than a regex can give).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, CHUNK_STRATEGY
from rag.loader import Document


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int  # position of this chunk within its source document
    section_title: str | None = None  # best-effort, see module docstring


# Matches a short line starting with numeric heading markers, e.g.
# "1. What is a Process?" or "26.1 Why Use Threads?". Deliberately strict:
# the title part must start with a capital letter and contain only
# letters/spaces/basic punctuation -- no code-symbol characters. Without
# that restriction this also matches numbered code-listing lines (e.g.
# PDF-extracted line numbers like "7 void *mythread(void *arg) {"), which
# it did on the first pass against the OSTEP chapters and fragmented
# recursive chunking badly (293 chunks instead of ~140). A real document
# parser would use font/layout cues instead of a regex to tell a heading
# from a numbered code line; this heuristic can't see that distinction and
# still isn't perfect, but this is enough to exclude the obvious case.
_HEADING_RE = re.compile(
    r"^\s*\d{1,2}(?:\.\d{1,2})?\.?\s+([A-Z][A-Za-z :,'\"\-?]{2,70})$",
    re.MULTILINE,
)


def _extract_headings(text: str) -> list[tuple[int, str]]:
    """Return (char_offset, title) for each heading-like line, in order."""
    return [(m.start(), m.group(1).strip()) for m in _HEADING_RE.finditer(text)]


def _section_title_at(headings: list[tuple[int, str]], position: int) -> str | None:
    """The most recent heading at or before `position`, if any."""
    title = None
    for offset, heading_title in headings:
        if offset > position:
            break
        title = heading_title
    return title


def _extract_paragraphs(text: str) -> list[tuple[int, str]]:
    """Return (start_offset, paragraph_text) for each blank-line-delimited block."""
    return [
        (m.start(), m.group().strip())
        for m in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text)
        if m.group().strip()
    ]


def _fixed_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Plain fixed-size sliding-window split, no Chunk wrapping. Shared by
    the fixed strategy and as the last-resort fallback inside recursive."""
    stride = chunk_size - overlap
    windows = []
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            windows.append(piece)
        start += stride
    return windows


def _pack_paragraphs(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into <=chunk_size pieces. A paragraph that
    alone exceeds chunk_size gets a fixed-size sliding-window cut as a
    last resort -- everything else stays whole."""
    packed: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush() -> None:
        if buffer:
            packed.append("\n\n".join(buffer).strip())

    for para in paragraphs:
        if len(para) > chunk_size:
            flush()
            buffer.clear()
            buffer_len = 0
            packed.extend(_fixed_windows(para, chunk_size, overlap))
            continue
        if buffer and buffer_len + len(para) + 2 > chunk_size:
            flush()
            buffer.clear()
            buffer_len = 0
        buffer.append(para)
        buffer_len += len(para) + 2

    flush()
    return [p for p in packed if p]


def chunk_document_fixed(document: Document, chunk_size: int = CHUNK_SIZE_CHARS,
                          overlap: int = CHUNK_OVERLAP_CHARS) -> list[Chunk]:
    """Split by raw character count, with overlap. See module docstring."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = document.text
    headings = _extract_headings(text)
    stride = chunk_size - overlap

    chunks = []
    start = 0
    index = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(Chunk(
                text=piece, source=document.source, chunk_index=index,
                section_title=_section_title_at(headings, start),
            ))
            index += 1
        start += stride

    return chunks


def chunk_document_paragraph(document: Document,
                              max_chunk_size: int = CHUNK_SIZE_CHARS) -> list[Chunk]:
    """Split on paragraph boundaries. See module docstring."""
    text = document.text
    headings = _extract_headings(text)
    paragraphs = _extract_paragraphs(text)

    chunks = []
    index = 0
    buffer_texts: list[str] = []
    buffer_start: int | None = None
    buffer_len = 0

    def flush() -> None:
        nonlocal index
        if buffer_texts:
            combined = "\n\n".join(buffer_texts).strip()
            if combined:
                chunks.append(Chunk(
                    text=combined, source=document.source, chunk_index=index,
                    section_title=_section_title_at(headings, buffer_start or 0),
                ))
                index += 1

    for offset, para in paragraphs:
        if buffer_texts and buffer_len + len(para) + 2 > max_chunk_size:
            flush()
            buffer_texts = []
            buffer_start = None
            buffer_len = 0
        if buffer_start is None:
            buffer_start = offset
        buffer_texts.append(para)
        buffer_len += len(para) + 2

    flush()
    return chunks


def chunk_document_recursive(document: Document, chunk_size: int = CHUNK_SIZE_CHARS,
                              overlap: int = CHUNK_OVERLAP_CHARS) -> list[Chunk]:
    """Split by section first, descending to paragraphs/fixed-cuts only
    where a section is too large -- then pack small adjacent sections
    back together up to chunk_size, so a run of short sections doesn't
    each become its own tiny chunk. See module docstring."""
    text = document.text
    headings = _extract_headings(text)
    boundaries = sorted({0, *(offset for offset, _ in headings), len(text)})

    # Pass 1: break into (title, piece) units no larger than chunk_size.
    # A section that already fits stays whole; an oversized section is
    # pre-split via paragraph packing.
    units: list[tuple[str | None, str]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        section_text = text[start:end].strip()
        if not section_text:
            continue
        title = _section_title_at(headings, start)

        if len(section_text) <= chunk_size:
            units.append((title, section_text))
        else:
            paragraphs = [p for _, p in _extract_paragraphs(section_text)]
            for piece in _pack_paragraphs(paragraphs, chunk_size, overlap):
                units.append((title, piece))

    # Pass 2: greedily pack consecutive units (every unit is already
    # <=chunk_size after pass 1) back up to chunk_size.
    chunks = []
    index = 0
    buffer_texts: list[str] = []
    buffer_title: str | None = None
    buffer_len = 0

    def flush() -> None:
        nonlocal index
        if buffer_texts:
            chunks.append(Chunk(
                text="\n\n".join(buffer_texts).strip(), source=document.source,
                chunk_index=index, section_title=buffer_title,
            ))
            index += 1

    for title, unit_text in units:
        if buffer_texts and buffer_len + len(unit_text) + 2 > chunk_size:
            flush()
            buffer_texts = []
            buffer_title = None
            buffer_len = 0
        if buffer_title is None:
            buffer_title = title
        buffer_texts.append(unit_text)
        buffer_len += len(unit_text) + 2

    flush()
    return chunks


_STRATEGIES = {
    "fixed": chunk_document_fixed,
    "paragraph": chunk_document_paragraph,
    "recursive": chunk_document_recursive,
}


def chunk_document(document: Document, strategy: str | None = None) -> list[Chunk]:
    """Chunk one document using the named strategy (default: config.CHUNK_STRATEGY)."""
    strategy = strategy or CHUNK_STRATEGY
    try:
        chunk_fn = _STRATEGIES[strategy]
    except KeyError:
        raise ValueError(f"Unknown chunk strategy {strategy!r}; choose from {list(_STRATEGIES)}")
    return chunk_fn(document)


def chunk_documents(documents: list[Document], strategy: str | None = None) -> list[Chunk]:
    """Chunk every document and flatten the result into one list."""
    all_chunks = []
    for document in documents:
        all_chunks.extend(chunk_document(document, strategy=strategy))
    return all_chunks

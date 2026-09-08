"""Stage 1: Loading.

Reads raw documents off disk and turns each one into a Document: the raw
text plus enough metadata to trace an answer back to its source later.

V1 deliberately only handles plain text (.txt, .md). Real-world formats
(PDF, HTML, DOCX) need format-specific extraction -- that's a V3+ problem,
not something to solve before the core pipeline even works end to end.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    text: str
    source: str  # relative file path, used for citations later


def load_documents(directory: str) -> list[Document]:
    """Read every .txt/.md file under `directory` into a Document.

    Data flow: filesystem -> raw bytes -> decoded text -> Document
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    documents = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue  # empty file, nothing to index
        documents.append(Document(text=text, source=str(path.relative_to(root))))

    return documents

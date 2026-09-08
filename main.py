"""CLI entry point for V1.

Usage:
    python main.py documents/

Indexes the given directory once, then drops into a loop where you can
ask multiple questions against that same in-memory index -- avoids
re-embedding every chunk on every question. V1 has no persistence across
process restarts; that's a deliberate V7 (production) concern, not a V1 one.
"""

import sys

from rag.pipeline import Pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    pipeline = Pipeline()

    n_chunks = pipeline.index(directory)
    print(f"Indexed {n_chunks} chunks from {directory}\n")

    print("Type a question and press Enter. Empty line or Ctrl+C to quit.\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break

        result = pipeline.ask(query)
        print(f"\nAnswer:\n{result.answer}")
        print(f"\nSources: {', '.join(result.sources)}\n")


if __name__ == "__main__":
    main()

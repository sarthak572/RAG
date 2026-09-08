"""LangChain equivalent of this project's from-scratch V1 RAG pipeline.

REFERENCE ONLY -- not part of the actual project. main.py still uses
rag/*.py, unchanged. This file exists to answer one specific question,
asked directly: "how much code does LangChain actually save, and what is
each call hiding underneath?" Every LangChain call below is commented with
which of our own files/functions it replaces.

Per the Framework Rule in CLAUDE.md, this is not a template to build
toward -- it exists purely as a side-by-side comparison, by explicit
request, not as a direction change for the project.

--- Line count ---
Our from-scratch pipeline (rag/loader.py + chunker.py + embedder.py +
vector_store.py + prompt.py + generator.py + pipeline.py) is roughly 260
lines including comments. The actual pipeline logic below (excluding this
docstring) is roughly 35 lines. That gap is real -- but see the caveats in
the comments below before treating it as "LangChain is just better":
several things our version does by default (RAG_DEBUG's token usage,
latency, and per-chunk similarity scores; a single retrieval pass instead
of two) require extra code here to get back.

--- Dependency count ---
Installing langchain-core, langchain-text-splitters, langchain-community,
and langchain-google-genai pulled in 34 additional packages (aiohttp,
sqlalchemy, langsmith, pydantic-settings, greenlet, ...). Our own
project's entire dependency list is 4 packages: google-genai,
python-dotenv, numpy, pytest.

Usage:
    python reference/langchain_equivalent.py documents/
"""

import sys

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, EMBEDDING_MODEL, GENERATION_MODEL, TOP_K


def build_chain(directory: str):
    # --- Loading -----------------------------------------------------
    # Replaces rag/loader.py's load_documents(): walks `directory`, reads
    # every matching file, wraps each in a LangChain Document (their
    # version of our own Document dataclass -- text + metadata dict,
    # where "source" is set automatically from the file path).
    loader = DirectoryLoader(
        directory, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    # --- Chunking ------------------------------------------------------
    # Replaces ALL THREE of our rag/chunker.py strategies (fixed,
    # paragraph, recursive) with one call. RecursiveCharacterTextSplitter
    # tries separators from coarsest to finest ("\n\n", "\n", " ", "") --
    # conceptually the same "prefer natural boundaries, fall back to a
    # hard cut" idea we built by hand in chunk_document_recursive(), just
    # generalized. It does NOT do our section-heading-aware splitting or
    # section_title metadata extraction -- that part of our V3 work has
    # no direct one-line equivalent here.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS,
    )
    chunks = splitter.split_documents(documents)

    # --- Embeddings + Vector Index ---------------------------------------
    # GoogleGenerativeAIEmbeddings replaces rag/embedder.py entirely --
    # INCLUDING the RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY task_type split
    # we implemented by hand as two separate functions. Verified by
    # reading the installed package's source: .embed_documents() defaults
    # task_type to "RETRIEVAL_DOCUMENT", .embed_query() defaults to
    # "RETRIEVAL_QUERY", automatically. It also batches internally (the
    # underlying API caps a request at 100 texts -- the same constraint
    # our _EMBED_BATCH_SIZE works around) but has NO equivalent to our
    # rate-limit _throttle() for the free-tier per-MINUTE cap -- that's
    # still something you would have to add yourself, framework or not.
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # InMemoryVectorStore replaces rag/vector_store.py's VectorStore class
    # -- same idea (brute-force cosine similarity over an in-memory list
    # of vectors -- see its docstring for the same O(n) scaling ceiling
    # ours has), just implemented for you.
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)  # <- this line makes real embedding API calls

    # as_retriever() bundles "embed the query, then similarity-search the
    # store" into one object -- replaces the embed_query() + store.search()
    # pairing that rag/pipeline.py's ask() does explicitly.
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    # --- Prompt Construction ---------------------------------------------
    # Replaces rag/prompt.py's build_prompt() + SYSTEM_INSTRUCTION verbatim.
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant that answers questions using ONLY the "
        "context provided below. If the answer is not contained in the "
        "context, say you don't know -- do not use outside knowledge. "
        "When you state a fact, cite it with the matching [N] chunk number.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    def format_docs(docs) -> str:
        return "\n\n".join(
            f"[{i + 1}] (source: {d.metadata.get('source')})\n{d.page_content}"
            for i, d in enumerate(docs)
        )

    # --- LLM ---------------------------------------------------------------
    # Replaces rag/generator.py's generate_answer() -- but note what's
    # MISSING compared to our version: this bare call gives you no token
    # usage, no latency, and (as wired below) no per-chunk similarity
    # scores. Getting that back is real, non-trivial code on top (a custom
    # callback handler, or LangSmith tracing) -- it isn't free just
    # because the framework is shorter for the happy path. CLAUDE.md's
    # Debugging Requirement gets all of that for free in our own
    # rag/pipeline.py because we wrote the orchestration ourselves and
    # could print anything we wanted at any step.
    llm = ChatGoogleGenerativeAI(model=GENERATION_MODEL)

    # --- Pipeline (LCEL) -----------------------------------------------
    # This `|`-chained pipeline replaces rag/pipeline.py's Pipeline class
    # entirely. Reading right to left through what happens to a question
    # string: it's piped in as {"question": ...} (RunnablePassthrough just
    # forwards it unchanged); "context" is produced by running that SAME
    # question through `retriever` (embed + search) then `format_docs`;
    # both dict values get substituted into the prompt template; the
    # filled prompt goes to the LLM; StrOutputParser unwraps the LLM's
    # response object down to a plain string.
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    chain, retriever = build_chain(directory)
    print("Type a question and press Enter. Empty line or Ctrl+C to quit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break

        answer = chain.invoke(query)
        # NOTE: this re-runs retrieval a SECOND time just to recover
        # sources for printing, since the chain above only exposes the
        # formatted context string, not the underlying Document objects,
        # to the caller. That's twice the embedding+search cost per
        # question compared to our rag/pipeline.py, which computes
        # scored_chunks once and reuses it for both the prompt and
        # AnswerResult.sources. Avoiding the double call here needs a
        # RunnableParallel restructuring -- a few more lines, eating
        # further into the "35 lines" headline number above.
        sources = sorted({d.metadata.get("source") for d in retriever.invoke(query)})
        print(f"\nAnswer:\n{answer}")
        print(f"\nSources: {', '.join(sources)}\n")


if __name__ == "__main__":
    main()

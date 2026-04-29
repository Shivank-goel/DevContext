"""Docs / RAG tools (stub)."""
from devcontext.rag.retriever import Retriever
from devcontext.rag.ingestion import ingest


# module-level singleton — initialized once, reused across calls
_retriever: Retriever = None


def get_retriever() -> Retriever:
    """Lazy-initialize the retriever singleton."""
    global _retriever
    if _retriever is None:
        vectorstore = ingest()
        _retriever = Retriever(vectorstore)
    return _retriever


def search_docs(query: str, top_k: int = 5) -> dict:
    """
    Search the docs/ knowledge base for content relevant to query.
    Returns formatted context string + raw chunks.
    """
    retriever = get_retriever()
    docs = retriever.retrieve(query, top_k=top_k)

    if not docs:
        return {
            "query": query,
            "context": "No relevant documentation found.",
            "chunks": [],
            "error": None
        }

    chunks = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": i
        }
        for i, doc in enumerate(docs)
    ]

    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}]\n{c['content']}"
        for c in chunks
    ])

    return {
        "query": query,
        "context": context,
        "chunks": chunks,
        "error": None
    }
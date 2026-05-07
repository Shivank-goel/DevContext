"""RAG tests (stub)."""
import pytest
from pathlib import Path
from devcontext.rag.ingestion import (
    load_documents,
    chunk_documents,
    ingest,
    load_vectorstore
)
from devcontext.rag.retriever import Retriever


# --- Ingestion tests ---

def test_load_documents_returns_list():
    docs = load_documents()
    assert isinstance(docs, list)
    assert len(docs) > 0, "Should load at least one document"


def test_load_documents_have_content():
    docs = load_documents()
    for doc in docs:
        assert doc.page_content.strip() != "", "Document content should not be empty"


def test_chunk_documents_produces_chunks():
    docs = load_documents()
    chunks = chunk_documents(docs)
    assert len(chunks) > len(docs), "Chunking should produce more chunks than documents"


def test_chunk_size_respected():
    docs = load_documents()
    chunks = chunk_documents(docs)
    for chunk in chunks:
        # allow some tolerance for overlap
        assert len(chunk.page_content) <= 600, \
            f"Chunk too large: {len(chunk.page_content)} chars"


def test_ingest_returns_vectorstore():
    vectorstore = ingest()
    assert vectorstore is not None


def test_vectorstore_has_documents():
    vectorstore = ingest()
    count = vectorstore._collection.count()
    assert count > 0, "ChromaDB collection should have documents"


# --- Retriever tests ---

def test_retriever_returns_results():
    vectorstore = ingest()
    retriever = Retriever(vectorstore)
    results = retriever.retrieve("What agents does DevContext have?")
    assert len(results) > 0, "Should retrieve at least one result"


def test_retriever_respects_top_k():
    vectorstore = ingest()
    retriever = Retriever(vectorstore)
    results = retriever.retrieve("agents", top_k=2)
    assert len(results) <= 2, "Should respect top_k limit"


def test_retriever_relevance():
    vectorstore = ingest()
    retriever = Retriever(vectorstore)
    results = retriever.retrieve("RAG pipeline chromadb embedding")
    # first result should mention RAG or pipeline
    top_content = results[0].page_content.lower()
    assert any(kw in top_content for kw in ["rag", "pipeline", "chroma", "embed"]), \
        "Top result should be semantically relevant"


def test_retrieve_with_scores_returns_tuples():
    vectorstore = ingest()
    retriever = Retriever(vectorstore)
    results = retriever.retrieve_with_scores("MCP server")
    assert len(results) > 0
    for doc, score in results:
        assert isinstance(score, float), "Score should be a float"
        assert score >= 0, "Cosine distance should be non-negative"


def test_irrelevant_query_still_returns_results():
    """
    RAG always returns top_k results regardless of relevance.
    This test documents that behavior explicitly.
    """
    vectorstore = ingest()
    retriever = Retriever(vectorstore)
    results = retriever.retrieve("what is the weather in tokyo today")
    # should still return results — just low quality ones
    assert len(results) > 0, "Retriever always returns results — no relevance threshold"
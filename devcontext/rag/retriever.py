"""ChromaDB retrieval (stub)."""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from devcontext.config.settings import settings
from devcontext.rag.ingestion import load_vectorstore


class Retriever:
    def __init__(self, vectorstore: Chroma = None):
        self.vectorstore = vectorstore or load_vectorstore()

    def retrieve(self, query: str, top_k: int = settings.top_k_results) -> list[Document]:
        """Retrieve top_k most relevant chunks for a query."""
        results = self.vectorstore.similarity_search(query, k=top_k)
        return results

    def retrieve_with_scores(self, query: str, top_k: int = settings.top_k_results) -> list[tuple]:
        """Retrieve chunks with relevance scores. Lower = more similar."""
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return results

    def format_context(self, docs: list[Document]) -> str:
        """Format retrieved docs into a single context string for the LLM."""
        return "\n\n---\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in docs
        )
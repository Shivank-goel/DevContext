from devcontext.config.llm import get_chat_model
from devcontext.config.settings import settings
from devcontext.rag.ingestion import ingest
from devcontext.rag.retriever import Retriever


def main():
    print("DevContext starting...")
    print(f"LLM       : Ollama ({settings.ollama_model})")
    print(f"Embeddings: {settings.embedding_model}")
    print(f"Ollama URL: {settings.ollama_base_url}")
    print(f"Docs dir  : {settings.docs_dir}")
    print(f"Chroma dir: {settings.chroma_db_dir}")
    print(f"MCP port  : {settings.mcp_server_port}\n")

    print("Testing LLM connection...")
    response = get_chat_model(settings).invoke("Reply with one word: working")
    print(f"LLM response: {response.content}\n")

    print("Testing RAG pipeline...")
    retriever = Retriever(ingest())
    query = "What agents does DevContext have?"
    docs = retriever.retrieve(query)
    print(f"Query: {query}")
    print(f"Retrieved {len(docs)} chunk(s)")
    for i, doc in enumerate(docs, start=1):
        print(f"\n[Chunk {i}] {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
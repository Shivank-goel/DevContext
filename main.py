from devcontext.config.settings import settings
from langchain_ollama import ChatOllama
from devcontext.rag.ingestion import ingest
from devcontext.rag.retriever import Retriever
from devcontext.agents.supervisor import run


def main():
    print("DevContext starting...")
    print(f"LLM       : Ollama ({settings.ollama_model})")
    print(f"Embeddings: {settings.embedding_model}")
    print(f"Ollama URL: {settings.ollama_base_url}")
    print(f"Docs dir  : {settings.docs_dir}")
    print(f"Chroma dir: {settings.chroma_db_dir}")
    print(f"MCP port  : {settings.mcp_server_port}\n")

    # condensed previous checks
    llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)
    print("LLM:", llm.invoke("one word: working").content)
    vectorstore = ingest()
    print(f"RAG: {Retriever(vectorstore).retrieve('agents', top_k=1)[0].page_content[:50]}...\n")

    print("=" * 40)
    print("Testing Supervisor Graph\n")

    # test 1 — should route to docs_agent
    q1 = "How does the RAG pipeline work?"
    r1 = run(q1)
    print(f"Q: {q1}")
    print(f"→ Routed to: {r1['agent_used']}")
    print(f"→ Response: {r1['response'][:200]}...\n")

    # test 2 — should route to code_agent
    q2 = "What does the settings.py file configure?"
    r2 = run(q2, filepath="devcontext/config/settings.py")
    print(f"Q: {q2}")
    print(f"→ Routed to: {r2['agent_used']}")
    print(f"→ Response: {r2['response'][:200]}...\n")

    # test 3 — should route to review_agent
    q3 = "Review the file_tools.py for any issues"
    r3 = run(q3, filepath="devcontext/tools/file_tools.py")
    print(f"Q: {q3}")
    print(f"→ Routed to: {r3['agent_used']}")
    print(f"→ Response: {r3['response'][:200]}...\n")



if __name__ == "__main__":
    main()
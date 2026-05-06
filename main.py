import uvicorn
import threading
from devcontext.config.settings import settings, setup_tracing
from langchain_ollama import ChatOllama
from devcontext.rag.ingestion import ingest
from devcontext.rag.retriever import Retriever
from devcontext.agents.supervisor import run
from devcontext.api.routes import app
from devcontext.rag.evaluator import run_evaluation, print_eval_report



def test_graph():
    print("=" * 40)
    print("Testing Supervisor Graph\n")

    queries = [
        ("How does the RAG pipeline work?", None),
        ("What does the settings.py file configure?", "devcontext/config/settings.py"),
        ("Review the file_tools.py for any issues", "devcontext/tools/file_tools.py"),
    ]

    for query, filepath in queries:
        result = run(query, filepath=filepath)
        print(f"Q: {query}")
        print(f"→ Routed to: {result['agent_used']}")
        print(f"→ Response: {result['response'][:150]}...\n")

def test_evaluation():
    print("=" * 40)
    print("Running RAG Evaluation (this takes ~2 mins)...")
    scores = run_evaluation()
    print_eval_report(scores)

def main():
    # setup tracing
    setup_tracing()

    print("DevContext starting...")
    print(f"LLM       : Ollama ({settings.ollama_model})")
    print(f"Ollama URL: {settings.ollama_base_url}")
    print(f"Docs dir  : {settings.docs_dir}")
    print(f"Chroma dir: {settings.chroma_db_dir}")
    print(f"MCP port  : {settings.mcp_server_port}\n")

    # smoke tests
    llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)
    print("LLM:", llm.invoke("one word: working").content)
    vectorstore = ingest()
    print(f"RAG: {Retriever(vectorstore).retrieve('agents', top_k=1)[0].page_content[:50]}...\n")

    # graph tests
    test_graph()

    # start FastAPI
    print("=" * 40)
    print(f"Starting FastAPI server on http://localhost:{settings.mcp_server_port}")
    print(f"Docs at http://localhost:{settings.mcp_server_port}/docs\n")
    test_evaluation()
    uvicorn.run(app, host=settings.mcp_server_host, port=settings.mcp_server_port)


if __name__ == "__main__":
    main()
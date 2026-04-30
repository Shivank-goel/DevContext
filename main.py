from devcontext.config.llm import get_chat_model
from devcontext.config.settings import settings
from devcontext.rag.ingestion import ingest
from devcontext.rag.retriever import Retriever
from devcontext.rag.retriever import Retriever
from devcontext.tools.file_tools import read_file, list_files
from devcontext.tools.git_tools import get_recent_commits, get_repo_summary
from devcontext.tools.docs_tools import search_docs
from devcontext.agents import AgentState
from devcontext.agents.code_agent import code_agent
from devcontext.agents.review_agent import review_agent
from devcontext.agents.docs_agent import docs_agent


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

    print("=" * 40)
    print("Testing file_tools...")
    result = read_file("docs/sample.md")
    print(f"read_file: {result['lines']} lines, {result['size_bytes']} bytes")

    files = list_files(".", extensions=[".py"])
    print(f"list_files: {files['file_count']} Python files found")

    print("\nTesting git_tools...")
    summary = get_repo_summary()
    if summary.get("error"):
        print(f"git: {summary['error']}")
    else:
        print(f"Branch: {summary['branch']}")
        print(f"Latest commit: {summary['latest_commit']['hash']} — {summary['latest_commit']['message']}")
        print(f"Tracked files: {summary['tracked_files']}")

    commits = get_recent_commits(n=3)
    if not commits.get("error"):
        print(f"Recent commits: {len(commits['commits'])} fetched")

    print("\nTesting docs_tools...")
    result = search_docs("How does the RAG pipeline work?")
    print(f"search_docs: {len(result['chunks'])} chunks retrieved")
    print(f"First chunk preview: {result['chunks'][0]['content'][:100]}...")


    # --- Code Agent ---
    print("Testing Code Agent...")
    state: AgentState = {
        "query": "What settings are configured in this file?",
        "filepath": "devcontext/config/settings.py",
        "agent": "",
        "file_content": None,
        "diff": None,
        "retrieved_context": None,
        "response": None,
        "error": None
    }
    result = code_agent(state)
    print(f"Agent: {result['agent']}")
    print(f"Response:\n{result['response']}\n")

    # --- Review Agent ---
    print("Testing Review Agent...")
    state2: AgentState = {
        "query": "Review this file",
        "filepath": "devcontext/tools/file_tools.py",
        "agent": "",
        "file_content": None,
        "diff": None,
        "retrieved_context": None,
        "response": None,
        "error": None
    }
    result2 = review_agent(state2)
    print(f"Agent: {result2['agent']}")
    print(f"Response:\n{result2['response']}\n")

    # --- Docs Agent ---
    print("Testing Docs Agent...")
    state3: AgentState = {
        "query": "How does the RAG pipeline work?",
        "filepath": None,
        "agent": "",
        "file_content": None,
        "diff": None,
        "retrieved_context": None,
        "response": None,
        "error": None
    }
    result3 = docs_agent(state3)
    print(f"Agent: {result3['agent']}")
    print(f"Response:\n{result3['response']}\n")


if __name__ == "__main__":
    main()
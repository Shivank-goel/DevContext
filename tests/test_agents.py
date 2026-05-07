"""Agent tests (stub)."""
import pytest
from devcontext.agents import AgentState
from devcontext.agents.code_agent import code_agent
from devcontext.agents.review_agent import review_agent
from devcontext.agents.docs_agent import docs_agent
from devcontext.agents.supervisor import run, supervisor_node


def make_state(query: str, filepath: str = None) -> AgentState:
    """Helper to build a clean state for testing."""
    return {
        "query": query,
        "filepath": filepath,
        "agent": "",
        "file_content": None,
        "diff": None,
        "retrieved_context": None,
        "response": None,
        "error": None
    }


# --- Code Agent tests ---

def test_code_agent_returns_response():
    state = make_state(
        "What settings are configured?",
        filepath="devcontext/config/settings.py"
    )
    result = code_agent(state)
    assert result["response"] is not None
    assert result["agent"] == "code_agent"
    assert result["error"] is None


def test_code_agent_with_invalid_file():
    state = make_state("explain this", filepath="nonexistent/file.py")
    result = code_agent(state)
    assert result["error"] is not None, "Should return error for missing file"


def test_code_agent_preserves_state_fields():
    state = make_state("explain this", filepath="devcontext/config/settings.py")
    result = code_agent(state)
    assert result["query"] == state["query"], "Query should be preserved in state"


# --- Review Agent tests ---

def test_review_agent_returns_response():
    state = make_state(
        "review this file",
        filepath="devcontext/tools/file_tools.py"
    )
    result = review_agent(state)
    assert result["response"] is not None
    assert result["agent"] == "review_agent"
    assert result["error"] is None


def test_review_agent_without_filepath():
    state = make_state("review something")
    result = review_agent(state)
    assert result["error"] is not None, "Should error without filepath"
    assert result["response"] is None


def test_review_agent_response_has_structure():
    state = make_state("review this", filepath="devcontext/tools/file_tools.py")
    result = review_agent(state)
    response = result["response"].lower()
    # review responses should mention code quality concepts
    assert any(kw in response for kw in ["bug", "error", "issue", "suggest",
                                          "improve", "code", "function"]), \
        "Review response should contain code review terminology"


# --- Docs Agent tests ---

def test_docs_agent_returns_response():
    state = make_state("How does the RAG pipeline work?")
    result = docs_agent(state)
    assert result["response"] is not None
    assert result["agent"] == "docs_agent"
    assert result["error"] is None


def test_docs_agent_sets_retrieved_context():
    state = make_state("What agents does DevContext have?")
    result = docs_agent(state)
    assert result["retrieved_context"] is not None, \
        "Docs agent should populate retrieved_context in state"


def test_docs_agent_unknown_query():
    state = make_state("What is the capital of France?")
    result = docs_agent(state)
    # should respond but indicate it couldn't find it
    assert result["response"] is not None
    assert result["error"] is None  # not an error — just no relevant context


# --- Supervisor routing tests ---

def test_supervisor_routes_docs_query():
    state = make_state("How does the MCP server work?")
    result = supervisor_node(state)
    assert result["agent"] == "docs_agent"


def test_supervisor_routes_code_query_with_filepath():
    state = make_state("explain this file", filepath="devcontext/config/settings.py")
    result = supervisor_node(state)
    assert result["agent"] == "code_agent"


def test_supervisor_routes_review_query_with_filepath():
    state = make_state("review this for bugs", filepath="devcontext/tools/file_tools.py")
    result = supervisor_node(state)
    assert result["agent"] == "review_agent"


def test_full_graph_run_docs():
    result = run("What technologies does DevContext use?")
    assert result["agent_used"] == "docs_agent"
    assert result["response"] is not None
    assert result["error"] is None


def test_full_graph_run_code():
    result = run(
        "What does this file configure?",
        filepath="devcontext/config/settings.py"
    )
    assert result["agent_used"] == "code_agent"
    assert result["response"] is not None


def test_run_always_returns_response_string():
    """run() should never return None for response."""
    result = run("some random query")
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0
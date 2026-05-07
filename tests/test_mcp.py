"""MCP server tests (stub)."""
import pytest
from devcontext.mcp_server.server import ask_codebase, review_file, search_docs


# --- MCP tool tests ---

def test_ask_codebase_returns_string():
    result = ask_codebase(
        query="What settings are configured?",
        filepath="devcontext/config/settings.py"
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_ask_codebase_invalid_file():
    result = ask_codebase(
        query="explain this",
        filepath="does/not/exist.py"
    )
    # should return error string, not raise exception
    assert isinstance(result, str)
    assert "error" in result.lower() or len(result) > 0


def test_review_file_returns_string():
    result = review_file(filepath="devcontext/tools/file_tools.py")
    assert isinstance(result, str)
    assert len(result) > 0


def test_search_docs_returns_string():
    result = search_docs(query="How does the RAG pipeline work?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_search_docs_unknown_query():
    result = search_docs(query="what is the weather today")
    # should return a string response, not crash
    assert isinstance(result, str)


def test_mcp_tools_never_raise():
    """
    MCP tools must NEVER raise exceptions — Claude Desktop
    has no way to handle Python exceptions gracefully.
    All errors must be returned as strings.
    """
    # these would normally cause errors
    try:
        ask_codebase("explain", "fake/path.py")
        review_file("another/fake.py")
        search_docs("")
    except Exception as e:
        pytest.fail(f"MCP tool raised an exception: {e}")
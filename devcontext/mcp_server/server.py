"""MCP stdio server (stub)."""
from mcp.server.fastmcp import FastMCP
from devcontext.agents.supervisor import run
from devcontext.config.settings import settings

mcp = FastMCP(
    name="devcontext",
    instructions="""DevContext is an agentic developer assistant.
It can answer questions about code files, review code for issues,
and search internal documentation. Always provide a filepath when
asking about specific files."""
)


@mcp.tool()
def ask_codebase(query: str, filepath: str) -> str:
    """
    Ask a question about a specific code file in the repository.
    
    Args:
        query: Your question about the code
        filepath: Path to the file relative to repo root e.g. devcontext/config/settings.py
    
    Returns:
        Detailed answer grounded on the actual file content
    """
    result = run(query, filepath=filepath)
    if result.get("error"):
        return f"Error: {result['error']}"
    return result["response"]


@mcp.tool()
def review_file(filepath: str) -> str:
    """
    Review a code file for bugs, quality issues, and improvement suggestions.
    
    Args:
        filepath: Path to the file to review e.g. devcontext/tools/file_tools.py
    
    Returns:
        Structured code review with specific suggestions
    """
    result = run(f"review this file for issues", filepath=filepath)
    if result.get("error"):
        return f"Error: {result['error']}"
    return result["response"]


@mcp.tool()
def search_docs(query: str) -> str:
    """
    Search the internal documentation knowledge base.
    Use this for questions about system architecture, how features work,
    configuration, and tech stack decisions.
    
    Args:
        query: Your question about the system or documentation
    
    Returns:
        Answer grounded strictly on internal documentation
    """
    result = run(query, filepath=None)
    if result.get("error"):
        return f"Error: {result['error']}"
    return result["response"]


def start_mcp_server():
    """Start the MCP server."""
    mcp.run(transport="stdio")
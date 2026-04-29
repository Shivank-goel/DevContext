"""LangGraph agents (supervisor and sub-agents)."""
# devcontext/agents/__init__.py
from typing import TypedDict, Optional

class AgentState(TypedDict):
    query: str                        # original user query
    agent: str                        # which agent was routed to
    filepath: Optional[str]           # for code/review agents
    file_content: Optional[str]       # read by code agent
    diff: Optional[str]               # read by review agent
    retrieved_context: Optional[str]  # filled by docs agent
    response: Optional[str]           # final LLM response
    error: Optional[str]              # any error that occurred
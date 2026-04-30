"""Code review agent (stub)."""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from devcontext.agents import AgentState
from devcontext.tools.git_tools import get_file_diff
from devcontext.tools.file_tools import read_file
from devcontext.config.settings import settings

llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=0.2
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior software engineer conducting a code review.
Analyze the provided code or diff and give structured feedback covering:
1. Potential bugs or logic errors
2. Code quality and readability issues  
3. Security concerns if any
4. Specific improvement suggestions

Be direct and specific. Reference line numbers or function names where possible."""),
    ("human", """File: {filepath}

{content_label}:
{content}

Provide a thorough code review:""")
])


def review_agent(state: AgentState) -> AgentState:
    """
    Reviews a file using its git diff or full content.
    Prefers diff if available, falls back to full file.
    """
    query = state["query"]
    filepath = state.get("filepath")

    if not filepath:
        return {
            **state,
            "agent": "review_agent",
            "error": "No filepath provided for review. Please specify a file.",
            "response": None
        }

    # try diff first
    diff_result = get_file_diff(filepath)
    if not diff_result["error"] and diff_result["diff"] != "No changes detected.":
        content = diff_result["diff"]
        content_label = "Git Diff"
    else:
        # fall back to full file
        file_result = read_file(filepath)
        if file_result["error"]:
            return {**state, "agent": "review_agent",
                    "error": file_result["error"], "response": None}
        content = file_result["content"]
        content_label = "Full File Content"

    # trim for context window
    max_chars = 6000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n... [truncated]"

    chain = PROMPT | llm
    result = chain.invoke({
        "filepath": filepath,
        "content_label": content_label,
        "content": content
    })

    return {
        **state,
        "agent": "review_agent",
        "filepath": filepath,
        "diff": content if content_label == "Git Diff" else None,
        "response": result.content,
        "error": None
    }
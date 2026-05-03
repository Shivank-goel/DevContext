"""LangGraph supervisor graph (stub)."""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from devcontext.agents import AgentState
from devcontext.agents.code_agent import code_agent
from devcontext.agents.review_agent import review_agent
from devcontext.agents.docs_agent import docs_agent
from devcontext.config.settings import settings


llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=0.0  # zero temp — routing must be deterministic
)

ROUTING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a routing agent. Your only job is to classify the 
user query into exactly one category. Reply with ONLY that single word.

code_agent   — user wants to understand what a SPECIFIC FILE does, 
               what a function returns, how code is written.
               Keywords: "what does X.py do", "explain this file", 
               "how does function X work", "what does X configure"
               
review_agent — user wants feedback, issues, bugs found in code.
               Keywords: "review", "check for issues", "audit", "find bugs"
               
docs_agent   — user asks about the SYSTEM overall, architecture, 
               how features work conceptually, tech stack, config values.
               Keywords: "how does the system", "what is the architecture",
               "how does RAG work", "what technologies"

CRITICAL RULE: If the query mentions a specific filename like settings.py, 
file_tools.py, ingestion.py — ALWAYS choose code_agent or review_agent, 
NEVER docs_agent.

Reply with exactly one of: code_agent, review_agent, docs_agent"""),
    ("human", "Query: {query}")
])


def supervisor_node(state: AgentState) -> AgentState:
    """
    Classifies the query and sets state['agent'] for routing.
    Filepath presence overrides LLM routing for code/review queries.
    """
    query = state["query"]
    filepath = state.get("filepath")

    # hard rule — if filepath given and query has review intent, force review_agent
    review_keywords = {"review", "check", "audit", "issues", "bugs", "problems", "fix"}
    if filepath and any(kw in query.lower() for kw in review_keywords):
        return {**state, "agent": "review_agent"}

    # hard rule — if filepath given and no review intent, force code_agent
    if filepath:
        return {**state, "agent": "code_agent"}

    # no filepath — use LLM to classify
    chain = ROUTING_PROMPT | llm
    result = chain.invoke({"query": query})
    agent_choice = result.content.strip().lower()

    valid_agents = {"code_agent", "review_agent", "docs_agent"}
    if agent_choice not in valid_agents:
        agent_choice = "docs_agent"

    print(f"  [supervisor] routed to: {agent_choice}")
    return {**state, "agent": agent_choice}


def route(state: AgentState) -> str:
    """
    Conditional edge function — returns the next node name.
    LangGraph calls this after supervisor_node to decide where to go.
    """
    return state["agent"]


def build_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph agent graph.
    """
    graph = StateGraph(AgentState)

    # add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("code_agent", code_agent)
    graph.add_node("review_agent", review_agent)
    graph.add_node("docs_agent", docs_agent)

    # entry point
    graph.add_edge(START, "supervisor")

    # conditional routing after supervisor
    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "code_agent": "code_agent",
            "review_agent": "review_agent",
            "docs_agent": "docs_agent"
        }
    )

    # all agents go to END
    graph.add_edge("code_agent", END)
    graph.add_edge("review_agent", END)
    graph.add_edge("docs_agent", END)

    return graph.compile()


# module-level compiled graph — initialize once
graph = build_graph()


def run(query: str, filepath: str = None) -> dict:
    """
    Public interface to the graph.
    Call this from API, MCP server, or main.py.
    """
    initial_state: AgentState = {
        "query": query,
        "filepath": filepath,
        "agent": "",
        "file_content": None,
        "diff": None,
        "retrieved_context": None,
        "response": None,
        "error": None
    }

    result = graph.invoke(initial_state)
    return {
        "agent_used": result["agent"],
        "response": result["response"],
        "error": result.get("error")
    }
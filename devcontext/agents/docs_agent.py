"""RAG-based docs agent (stub)."""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from devcontext.agents import AgentState
from devcontext.tools.docs_tools import search_docs
from devcontext.config.settings import settings

llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=0.1
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a documentation assistant. Answer questions strictly 
based on the provided documentation context. 
If the answer is not in the context, say: 'I could not find this in the documentation.'
Do not make up information."""),
    ("human", """Documentation Context:
{context}

Question: {query}

Answer based only on the documentation above:""")
])


def docs_agent(state: AgentState) -> AgentState:
    """
    Searches docs knowledge base and answers grounded on retrieved context.
    """
    query = state["query"]

    search_result = search_docs(query)

    if search_result["error"]:
        return {
            **state,
            "agent": "docs_agent",
            "error": search_result["error"],
            "response": None
        }

    context = search_result["context"]

    chain = PROMPT | llm
    result = chain.invoke({
        "context": context,
        "query": query
    })

    return {
        **state,
        "agent": "docs_agent",
        "retrieved_context": context,
        "response": result.content,
        "error": None
    }
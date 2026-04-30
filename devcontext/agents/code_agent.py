"""Code Q&A agent (stub)."""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from devcontext.agents import AgentState
from devcontext.tools.file_tools import read_file, list_files
from devcontext.config.settings import settings

llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=0.1  # low temp — we want precise, factual answers
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert code analyst. You answer questions about 
code strictly based on the file content provided. Do not guess or hallucinate.
If the answer is not in the provided code, say so clearly."""),
    ("human", """File: {filepath}

Content:
{file_content}

Question: {query}

Answer based only on the code above:""")
])


def code_agent(state: AgentState) -> AgentState:
    """
    Reads a file and answers the user's question about it.
    Expects state['filepath'] to be set by the supervisor.
    """
    query = state["query"]
    filepath = state.get("filepath")

    # if no filepath given, list available files and ask LLM to pick
    if not filepath:
        files = list_files(".", extensions=[".py"])
        file_list = "\n".join(files["files"])

        list_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a code assistant. Given a list of files and a question, "
                      "identify which single file is most relevant. Reply with just the filepath."),
            ("human", "Files:\n{files}\n\nQuestion: {query}\n\nMost relevant file:")
        ])
        chain = list_prompt | llm
        result = chain.invoke({"files": file_list, "query": query})
        filepath = result.content.strip()

    # read the file
    file_result = read_file(filepath)
    if file_result["error"]:
        return {**state, "error": file_result["error"], "agent": "code_agent"}

    file_content = file_result["content"]

    # trim if too long for context window
    max_chars = 6000
    if len(file_content) > max_chars:
        file_content = file_content[:max_chars] + "\n... [truncated]"

    # call LLM
    chain = PROMPT | llm
    result = chain.invoke({
        "filepath": filepath,
        "file_content": file_content,
        "query": query
    })

    return {
        **state,
        "agent": "code_agent",
        "filepath": filepath,
        "file_content": file_content,
        "response": result.content,
        "error": None
    }

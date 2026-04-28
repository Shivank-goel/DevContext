# DevContext Documentation

## Architecture Overview
DevContext is a multi-agent developer assistant built using LangGraph
for orchestration and LangChain for LLM interactions. It routes queries
to specialized agents using a supervisor pattern. The supervisor receives
the user query, classifies intent, and delegates to the appropriate agent.
All agents share a common state object that gets passed through the graph.

## Agents

### Code Agent
The Code Agent answers questions about code files and functions in the
repository. It reads file contents, understands structure, and uses the
LLM to explain logic, identify patterns, and answer questions grounded
strictly on the actual code — not hallucinated assumptions.

### Review Agent
The Review Agent analyzes code diffs and pull request changes. It reads
git diffs using GitPython, identifies potential bugs, style issues, and
logic errors, and returns structured feedback with line-level suggestions.

### Docs Agent
The Docs Agent retrieves relevant documentation using RAG. It embeds
the user query, searches ChromaDB for the most semantically similar
chunks, and passes retrieved context to the LLM to generate a grounded
answer. It never answers from memory alone.

## RAG Pipeline
Documents from the docs/ folder are loaded using LangChain's
DirectoryLoader. They are split into 512-token chunks with 64-token
overlap using RecursiveCharacterTextSplitter. Each chunk is embedded
using nomic-embed-text via Ollama and stored in ChromaDB as a persistent
vector index on disk. At query time, cosine similarity search retrieves
the top-k most relevant chunks.

## MCP Server
The entire system is exposed as an MCP (Model Context Protocol) server
running on port 8000 using FastAPI and the MCP Python SDK. It exposes
three tools: ask_codebase for code questions, review_file for diff
analysis, and search_docs for documentation retrieval. Any MCP-compatible
client including Claude Desktop can connect and call these tools directly.

## Configuration
All settings are managed via pydantic-settings and loaded from a .env
file at startup. Key settings include: chunk_size=512, chunk_overlap=64,
top_k=5, ollama_base_url, model_name, and chroma_db persist directory.
Pydantic validates all values at startup and fails fast on missing keys.

## Tech Stack
- LangGraph: agent orchestration and state machine
- LangChain: LLM calls, prompt templates, document loaders
- ChromaDB: persistent vector store for RAG
- Ollama: local LLM and embedding inference
- FastAPI: REST API and MCP server layer
- GitPython: reading git history and diffs
- RAGAS: evaluating RAG retrieval quality
- LangSmith: tracing and observability
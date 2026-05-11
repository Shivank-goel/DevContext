# DevContext — Agentic Code Assistant

A production-grade multi-agent developer tool that answers questions about 
your codebase, reviews code for issues, and searches internal documentation 
— all through a **LangGraph supervisor pattern** and an **MCP server** 
integrable with Claude Desktop.

---

## Architecture
User Query (via MCP or REST API)
↓
┌─────────────────────────┐
│   Supervisor Agent      │  ← Hybrid rule-LLM router
│   (LangGraph node)      │    Hard rules for filepath queries
└────┬──────┬─────────┬───┘    LLM classification for open queries
↓      ↓         ↓
Code      Review    Docs
Agent     Agent     Agent
│         │         │
│         │         └── ChromaDB RAG retrieval
│         └──────────── Git diff + file reader
└────────────────────── File reader + LLM Q&A
↓      ↓         ↓
┌─────────────────────────┐
│   LangGraph State       │  ← Shared typed state across all nodes
└─────────────────────────┘
↓
┌───────────────┐    ┌─────────────────┐
│  MCP Server   │    │  FastAPI REST   │
│  (stdio)      │    │  (:8000)        │
└───────────────┘    └─────────────────┘
↓
Claude Desktop / any MCP client


---

## Features

- **Code Q&A** — ask questions about any file in your repo; answers grounded strictly on file content
- **Code Review** — get structured feedback on code files or git diffs with bug detection and improvement suggestions  
- **Docs Search** — RAG over your internal markdown docs with ChromaDB and nomic-embed-text embeddings
- **MCP Server** — plug directly into Claude Desktop with 3 callable tools
- **REST API** — FastAPI with Swagger UI at `/docs`
- **RAGAS Evaluation** — RAG quality measured: faithfulness 0.944, answer relevancy 0.847
- **LangSmith Tracing** — full observability across every agent hop
- **36 tests** — unit, integration, and end-to-end coverage

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM & Embeddings | Ollama (llama3.1:8b + nomic-embed-text) |
| RAG & Vector Store | LangChain + ChromaDB |
| RAG Evaluation | RAGAS |
| Observability | LangSmith |
| MCP Server | MCP Python SDK (FastMCP) |
| REST API | FastAPI |
| Git Integration | GitPython |
| Config | Pydantic Settings |

---

## Project Structure
DevContext/
├── devcontext/
│   ├── agents/
│   │   ├── supervisor.py     # LangGraph graph + hybrid router
│   │   ├── code_agent.py     # File Q&A agent
│   │   ├── review_agent.py   # Code review agent
│   │   └── docs_agent.py     # RAG-powered docs agent
│   ├── rag/
│   │   ├── ingestion.py      # Load → chunk → embed → store
│   │   ├── retriever.py      # Semantic search over ChromaDB
│   │   └── evaluator.py      # RAGAS evaluation pipeline
│   ├── tools/
│   │   ├── file_tools.py     # File reader + directory lister
│   │   ├── git_tools.py      # Git diff + commit history
│   │   └── docs_tools.py     # Retriever wrapper
│   ├── mcp_server/
│   │   └── server.py         # MCP tools: ask_codebase, review_file, search_docs
│   ├── api/
│   │   └── routes.py         # FastAPI endpoints
│   └── config/
│       └── settings.py       # Pydantic settings + tracing setup
├── docs/                     # RAG corpus — add your markdown docs here
├── tests/                    # 36 tests across unit/integration/e2e
└── main.py                   # Entry point

---

## Quickstart

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running

### 1. Clone and install
```bash
git clone https://github.com/Shivank-goel/DevContext.git
cd DevContext
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Pull models
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 3. Configure
```bash
cp .env.example .env
# Optional: add LANGSMITH_API_KEY for tracing
```

### 4. Add your docs
Drop any `.md` files into the `docs/` folder — these become your searchable knowledge base.

### 5. Run
```bash
python main.py
```

API available at `http://localhost:8000`
Swagger UI at `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/query` | Auto-route to best agent |
| POST | `/code` | Force code Q&A (requires filepath) |
| POST | `/review` | Force code review (requires filepath) |
| POST | `/docs` | Force docs search |
| POST | `/eval` | Run RAGAS evaluation |

### Example
```bash
# Auto-routed query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does the RAG pipeline work?"}'

# Code review
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{"query": "review for issues", "filepath": "devcontext/tools/file_tools.py"}'
```

---

## Claude Desktop Integration (MCP)

Add to your Claude Desktop config:

**Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "devcontext": {
      "command": "python",
      "args": ["-c", "from devcontext.mcp_server.server import start_mcp_server; start_mcp_server()"],
      "cwd": "/absolute/path/to/DevContext"
    }
  }
}
```

Restart Claude Desktop — DevContext tools appear in the tools panel automatically.

---

## RAG Evaluation

Run the RAGAS evaluation pipeline:

```bash
curl -X POST http://localhost:8000/eval
```

| Metric | Score |
|---|---|
| Faithfulness | 0.944 |
| Answer Relevancy | 0.847 |
| Overall | 0.895 |

*Evaluated on llama3.1:8b via local Ollama — no external API calls.*

---

## Running Tests

```bash
pytest tests/ -v
```
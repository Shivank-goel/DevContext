# DevContext

An agentic developer tool that reads your codebase, answers questions about it, reviews PRs, and surfaces relevant internal docs — exposed via MCP and (eventually) HTTP.

**Status:** Package layout, configuration, and dependencies are in place. Supervisor agents, MCP stdio server, FastAPI routes, and RAG pipelines are still stubs unless noted in the codebase.

## Prerequisites

- **Python 3.11+** (required by `pyproject.toml` and dependencies such as `mcp`). The system `python3` on macOS is often 3.9; use Homebrew or pyenv, for example:

  ```bash
  brew install python@3.12
  /opt/homebrew/opt/python@3.12/bin/python3.12 --version
  ```

- **[Ollama](https://ollama.com/)** for local LLM inference. Install the CLI, start the daemon, and pull a model (defaults in this repo target **`llama3.2`**):

  ```bash
  ollama serve
  ```

  In another terminal:

  ```bash
  ollama pull llama3.2
  ```

  List models already on disk: `ollama list`. Set `OLLAMA_MODEL` in `.env` to any tag you have (for example `llama3`, `mistral`, `qwen2.5`).

## Setup

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

The `[dev]` extra installs **pytest** and **pytest-asyncio** for running tests.

## Configuration

Copy the example environment file (optional; defaults match Ollama on localhost):

```bash
cp .env.example .env
```

Variables are documented in [.env.example](.env.example). The important ones are **`OLLAMA_BASE_URL`** (default `http://localhost:11434`) and **`OLLAMA_MODEL`** (default `llama3.2`).

Settings load from `.env` via [devcontext/config/settings.py](devcontext/config/settings.py). Chat models are built in [devcontext/config/llm.py](devcontext/config/llm.py) using **LangChain `ChatOllama`**.

## Run

Print resolved configuration (paths and Ollama settings):

```bash
python main.py
```

Use the **same interpreter** where you ran `pip install -e .` (typically after `source .venv/bin/activate`). A missing dependency shows a one-line install hint.

Ensure **`ollama serve`** is running and your **`OLLAMA_MODEL`** is pulled (`ollama pull …`) before agents call the model.

## Tests

```bash
pytest
```

Smoke tests verify Python version, package install, Ollama-oriented `Settings`, and `get_chat_model()`.

## Project layout

- `devcontext/` — installable Python package (config, agents, tools, RAG, MCP, API).
- `docs/` — sample documentation for future RAG ingestion.
- `chroma_db/` — default persistent Chroma path (contents are gitignored; `.gitkeep` keeps the folder).
- `main.py` — CLI entry (MCP vs API modes to be wired as implementation progresses).

## UV (optional)

If you use [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

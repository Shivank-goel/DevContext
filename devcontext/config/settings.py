"""Pydantic settings and environment."""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # LLM — local Ollama (see README for serve + pull)
    llm_provider: Literal["ollama"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"

    # Paths
    docs_dir: Path = Path("docs")
    chroma_db_dir: Path = Path("chroma_db")
    collection_name: str = "devcontext_docs"

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_results: int = 5

    # MCP
    mcp_server_host: str = "0.0.0.0"
    mcp_server_port: int = 8000

    # LangSmith
    langsmith_project: str = "devContext"
    langsmith_tracing: bool = True


import os

def setup_tracing():
    """
    Configure LangSmith tracing via environment variables.
    LangChain/LangGraph automatically picks these up.
    """
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        print(f"LangSmith tracing enabled → project: {settings.langsmith_project}")
    else:
        print("LangSmith tracing disabled — set LANGSMITH_API_KEY to enable")
settings = Settings()

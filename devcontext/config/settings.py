"""Pydantic settings and environment (stub)."""
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-20250514"

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
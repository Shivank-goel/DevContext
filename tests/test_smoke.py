"""Fast checks that the environment and install match project expectations."""

import sys
from pathlib import Path

import pytest


def test_python_version_is_supported() -> None:
    assert sys.version_info >= (3, 11), (
        "DevContext requires Python 3.11 or newer. "
        "On macOS, try: brew install python@3.12 && python3.12 -m venv .venv"
    )


def test_distribution_metadata_present() -> None:
    import importlib.metadata

    version = importlib.metadata.version("devcontext")
    assert version
    parts = version.split(".")
    assert len(parts) >= 1


def test_settings_load_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))

    from devcontext.config.settings import Settings

    s = Settings()
    assert s.llm_provider == "ollama"
    assert s.ollama_model
    assert s.ollama_base_url.startswith("http")
    assert s.docs_dir == tmp_path / "docs"
    assert s.chroma_db_dir == tmp_path / "chroma"


def test_get_chat_model_builds_ollama(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")

    from devcontext.config.llm import get_chat_model
    from devcontext.config.settings import Settings

    model = get_chat_model(Settings())
    assert model.__class__.__name__ == "ChatOllama"

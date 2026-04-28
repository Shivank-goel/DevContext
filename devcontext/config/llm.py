"""Provider-agnostic chat model factory (Ollama)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

if TYPE_CHECKING:
    from devcontext.config.settings import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    """Return the configured chat model. Today this is always Ollama."""
    if settings.llm_provider != "ollama":
        raise ValueError(f"Unsupported llm_provider: {settings.llm_provider!r}")
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.2,
    )

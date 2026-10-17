"""Runtime settings. Tests and CI stay on the fake provider unless overridden."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "auto"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-latest"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    embedding_provider: str = "auto"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 384

    vector_backend: str = "memory"
    database_url: str = ""

    load_samples: bool = True
    samples_dir: Path = Path("data/samples")
    uploads_dir: Path = Path("data/uploads")
    log_level: str = "INFO"

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "fin-doc-analyst"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    retrieval_k: int = 4
    max_upload_bytes: int = Field(default=10 * 1024 * 1024)

    def resolved_llm_provider(self) -> str:
        choice = self.llm_provider.strip().lower()
        if choice and choice != "auto":
            return choice
        if self.openai_api_key.strip():
            return "openai"
        return "fake"

    def resolved_embedding_provider(self) -> str:
        choice = self.embedding_provider.strip().lower()
        if choice and choice != "auto":
            return choice
        if self.resolved_llm_provider() == "openai" and self.openai_api_key.strip():
            return "openai"
        if self.resolved_llm_provider() == "ollama":
            return "ollama"
        return "fake"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_tracing(settings: Settings) -> None:
    """LangGraph reads LangSmith settings from the process environment."""
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"

"""Build the configured chat client."""

from __future__ import annotations

from fin_doc_analyst.config import Settings
from fin_doc_analyst.llm.anthropic_client import AnthropicClient
from fin_doc_analyst.llm.base import LLMError
from fin_doc_analyst.llm.fake import FakeLLM
from fin_doc_analyst.llm.ollama_client import OllamaClient
from fin_doc_analyst.llm.openai_client import OpenAIClient


def build_llm(settings: Settings) -> FakeLLM | OpenAIClient | AnthropicClient | OllamaClient:
    provider = settings.resolved_llm_provider()
    if provider == "fake":
        return FakeLLM()
    if provider == "openai":
        return OpenAIClient(model=settings.openai_model, api_key=settings.openai_api_key)
    if provider == "anthropic":
        return AnthropicClient(model=settings.anthropic_model, api_key=settings.anthropic_api_key)
    if provider == "ollama":
        return OllamaClient(model=settings.ollama_model, base_url=settings.ollama_base_url)
    raise LLMError(f"Unknown LLM provider: {provider}")

"""Force the offline providers before the app is imported."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ["LLM_PROVIDER"] = "fake"
os.environ["EMBEDDING_PROVIDER"] = "fake"
os.environ["VECTOR_BACKEND"] = "memory"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["LOAD_SAMPLES"] = "false"

import pytest

from fin_doc_analyst.agent.service import AnalystService
from fin_doc_analyst.config import Settings, get_settings
from fin_doc_analyst.llm.factory import build_llm
from fin_doc_analyst.retrieval.embeddings import build_embedder
from fin_doc_analyst.retrieval.factory import build_store

get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider="fake",
        embedding_provider="fake",
        vector_backend="memory",
        openai_api_key="",
        anthropic_api_key="",
        langsmith_tracing=False,
        load_samples=False,
        log_level="WARNING",
        samples_dir=ROOT / "data" / "samples",
        uploads_dir=tmp_path / "uploads",
    )


@pytest.fixture
def service(settings: Settings) -> AnalystService:
    embedder = build_embedder(settings)
    store = build_store(settings, embedder)
    analyst = AnalystService(settings, store, build_llm(settings))
    analyst.load_directory(ROOT / "data" / "samples", persist=False)
    return analyst

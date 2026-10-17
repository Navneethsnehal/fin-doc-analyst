"""Select the memory store or Postgres."""

from __future__ import annotations

from fin_doc_analyst.config import Settings
from fin_doc_analyst.retrieval.embeddings import Embedder
from fin_doc_analyst.retrieval.memory import MemoryStore
from fin_doc_analyst.retrieval.pgvector_store import PgVectorStore


def build_store(settings: Settings, embedder: Embedder) -> MemoryStore | PgVectorStore:
    backend = settings.vector_backend.strip().lower()
    if backend == "pgvector":
        return PgVectorStore(settings.database_url, embedder)
    if backend != "memory":
        raise RuntimeError(f"Unknown vector backend: {backend}")
    return MemoryStore(embedder)

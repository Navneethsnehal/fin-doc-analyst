"""Embedding providers.

The default embedder is a hashed bag-of-tokens. It is deterministic, needs no
download, and is strong enough on a small financial corpus when fused with BM25.
OpenAI text-embedding-3-small is requested at the same dimension so one pgvector
column serves both providers.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

import numpy as np

from fin_doc_analyst.textutil import tokenize


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Feature-hashing embedder. Same text always maps to the same unit vector."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float64)
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode()).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str, dim: int, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self._client = client
        self.model = model
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self.model, input=texts, dimensions=self.dim
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str, dim: int, client: Any | None = None) -> None:
        import httpx

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dim = dim
        self._client = client or httpx.Client(timeout=60.0)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        response = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        vector = list(response.json().get("embedding") or [])
        if len(vector) != self.dim:
            raise ValueError(
                f"Ollama embedding dim {len(vector)} does not match EMBEDDING_DIM={self.dim}"
            )
        array = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(array))
        if norm > 0:
            array /= norm
        return array.tolist()


def build_embedder(settings: Any) -> Embedder:
    provider = settings.resolved_embedding_provider()
    if provider == "openai":
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
    if provider == "ollama":
        return OllamaEmbedder(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            dim=settings.embedding_dim,
        )
    return HashEmbedder(dim=settings.embedding_dim)
